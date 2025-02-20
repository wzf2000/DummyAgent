import logging
import os
import json
import lmdb
from typing import Iterator
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")

class CacheInteractionTool:
    def __init__(self, data_dir: str, block_set_dir: str | None = None):
        """
        Initialize the CacheInteractionTool with the dataset directory.
        Args:
            data_dir: Path to the directory containing dataset files.
            block_set_dir: Path to the directory containing block set files.
        """
        logger.info(f"Initializing CacheInteractionTool with data directory: {data_dir}")

        # Create LMDB environments
        self.env_dir = os.path.join(data_dir, "lmdb_cache")
        os.makedirs(self.env_dir, exist_ok=True)

        self.user_env: lmdb.Environment = lmdb.open(os.path.join(self.env_dir, "users"), map_size=2 * 1024 * 1024 * 1024)
        self.item_env: lmdb.Environment = lmdb.open(os.path.join(self.env_dir, "items"), map_size=2 * 1024 * 1024 * 1024)
        self.review_env: lmdb.Environment = lmdb.open(os.path.join(self.env_dir, "reviews"), map_size=8 * 1024 * 1024 * 1024)

        block_set_items = []
        self._block_set_pairs = set()
        if block_set_dir:
            logger.info(f"Loading block set data from: {block_set_dir}")
            block_set_items = self._load_block_set_items(block_set_dir)
            self._block_set_pairs = {(item['user_id'], item['item_id']) for item in block_set_items}

        # Initialize the database if empty
        self._initialize_db(data_dir)

    def _load_block_set_items(self, block_set_dir: str) -> list[dict]:
        """Load all block set files from the block set directory."""
        block_set_data = []
        task_dir = os.path.join(block_set_dir, "tasks")
        gt_dir = os.path.join(block_set_dir, "groundtruth")

        for filename in os.listdir(task_dir):
            if filename.startswith('task_') and filename.endswith('.json'):
                task_file_path = os.path.join(task_dir, filename)
                with open(task_file_path, 'r', encoding='utf-8') as task_file:
                    task_data = json.load(task_file)
                    if task_data['type'] == 'user_behavior_simulation':
                        block_set_data.append({'user_id': task_data['user_id'], 'item_id': task_data['item_id']})
                    else:
                        gt_filename = filename.replace('task_', 'groundtruth_')
                        gt_file_path = os.path.join(gt_dir, gt_filename)
                        with open(gt_file_path, 'r', encoding='utf-8') as gt_file:
                            gt_data = json.load(gt_file)
                            for item in task_data['candidate_list']:
                                if item == gt_data['ground truth']:
                                    block_set_data.append({'user_id': task_data['user_id'], 'item_id': item})
        return block_set_data

    def _initialize_db(self, data_dir: str):
        """Initialize the LMDB databases with data if they are empty."""
        # Initialize users
        with self.user_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for user in tqdm(self._iter_file(data_dir, 'user.json')):
                        cursor.put(
                            user['user_id'].encode(),
                            json.dumps(user).encode()
                        )

        # Initialize items
        with self.item_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for item in tqdm(self._iter_file(data_dir, 'item.json')):
                        cursor.put(
                            item['item_id'].encode(),
                            json.dumps(item).encode()
                        )

        # Initialize reviews and their indices
        with self.review_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                for review in tqdm(self._iter_file(data_dir, 'review.json')):
                    # Store the review
                    txn.put(
                        review['review_id'].encode(),
                        json.dumps(review).encode()
                    )

                    # Update item reviews index (store only review_ids)
                    item_review_ids = json.loads(txn.get(f"item_{review['item_id']}".encode()) or '[]')
                    item_review_ids.append(review['review_id'])
                    txn.put(
                        f"item_{review['item_id']}".encode(),
                        json.dumps(item_review_ids).encode()
                    )

                    # Update user reviews index (store only review_ids)
                    user_review_ids = json.loads(txn.get(f"user_{review['user_id']}".encode()) or '[]')
                    user_review_ids.append(review['review_id'])
                    txn.put(
                        f"user_{review['user_id']}".encode(),
                        json.dumps(user_review_ids).encode()
                    )

    def _iter_file(self, data_dir: str, filename: str) -> Iterator[dict]:
        """Iterate through file line by line."""
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

    def get_user(self, user_id: str) -> dict | None:
        """Fetch user data based on user_id."""
        with self.user_env.begin() as txn:
            user_data = txn.get(user_id.encode())
            if user_data:
                return json.loads(user_data)
        return None

    def get_item(self, item_id: str) -> dict | None:
        """Fetch item data based on item_id."""
        if not item_id:
            return None

        with self.item_env.begin() as txn:
            item_data = txn.get(item_id.encode())
            if item_data:
                return json.loads(item_data)
        return None

    def get_reviews(
            self,
            item_id: str | None = None,
            user_id: str | None = None,
            review_id: str | None = None
    ) -> list[dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            with self.review_env.begin() as txn:
                review_data = txn.get(review_id.encode())
                if review_data:
                    review = json.loads(review_data)
                    if (review['user_id'], review['item_id']) in self._block_set_pairs:
                        return []
                    return [review]
            return []

        with self.review_env.begin() as txn:
            if item_id:
                review_ids = json.loads(txn.get(f"item_{item_id}".encode()) or '[]')
            elif user_id:
                review_ids = json.loads(txn.get(f"user_{user_id}".encode()) or '[]')
            else:
                return []

            # Fetch complete review data for each review_id
            reviews = []
            for rid in review_ids:
                review_data = txn.get(rid.encode())
                if review_data:
                    review = json.loads(review_data)
                    if (review['user_id'], review['item_id']) in self._block_set_pairs:
                        continue
                    reviews.append(review)
            return reviews

    def __del__(self):
        """Cleanup LMDB environments on object destruction."""
        self.user_env.close()
        self.item_env.close()
        self.review_env.close()
