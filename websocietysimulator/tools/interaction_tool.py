import logging
import os
import json

logger = logging.getLogger("websocietysimulator")

class InteractionTool:
    def __init__(self, data_dir: str, block_set_dir: str | None = None):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing dataset files.
            block_set_dir: Path to the directory containing block set files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        # Convert DataFrames to dictionaries for O(1) lookup
        logger.info(f"Loading item data from {os.path.join(data_dir, 'item.json')}")
        self.item_data = {item['item_id']: item for item in self._load_data(data_dir, 'item.json')}
        logger.info(f"Loading user data from {os.path.join(data_dir, 'user.json')}")
        self.user_data = {user['user_id']: user for user in self._load_data(data_dir, 'user.json')}

        # Create review indices
        logger.info(f"Loading review data from {os.path.join(data_dir, 'review.json')}")
        reviews = self._load_data(data_dir, 'review.json')
        # Load ground truth data if available
        block_set_items = []
        block_set_pairs = set()
        if block_set_dir:
            logger.info(f"Loading block set data from: {block_set_dir}")
            block_set_items = self._load_block_set_items(block_set_dir)
            block_set_pairs = {(item['user_id'], item['item_id']) for item in block_set_items}

        filtered_reviews = []
        for review in reviews:
            if (review['user_id'], review['item_id']) not in block_set_pairs:
                filtered_reviews.append(review)

        logger.info(f"Filtered {len(reviews) - len(filtered_reviews)} reviews based on block set")
        reviews = filtered_reviews
        self.review_data = {review['review_id']: review for review in reviews}

        self.item_reviews = {}
        self.user_reviews = {}

        # Build review indices
        logger.info("Building review indices")
        for review in reviews:
            # Index by item_id
            self.item_reviews.setdefault(review['item_id'], []).append(review)
            # Index by user_id
            self.user_reviews.setdefault(review['user_id'], []).append(review)

    def _load_data(self, data_dir: str, filename: str) -> list[dict]:
        """Load data as a list of dictionaries."""
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]

    def _load_block_set_items(self, block_set_dir: str) -> list[dict]:
        """Load block set items from the block set directory."""
        block_set_data = []
        task_dir = os.path.join(block_set_dir, 'tasks')
        gt_dir = os.path.join(block_set_dir, 'groundtruth')

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

    def get_user(self, user_id: str) -> dict | None:
        """Fetch user data based on user_id."""
        return self.user_data.get(user_id)

    def get_item(self, item_id: str = None) -> dict | None:
        """Fetch item data based on item_id."""
        return self.item_data.get(item_id) if item_id else None

    def get_reviews(
        self, 
        item_id: str | None = None, 
        user_id: str | None = None, 
        review_id: str | None = None
    ) -> list[dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            return [self.review_data[review_id]] if review_id in self.review_data else []

        if item_id:
            return self.item_reviews.get(item_id, [])
        elif user_id:
            return self.user_reviews.get(user_id, [])

        return []
