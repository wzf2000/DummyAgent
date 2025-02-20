import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time
import numpy as np
logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class CoTReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str, temperature: float = 0.1, n: int = 1):
        """Override the parent class's __call__ method"""
        prompt = '''#Here is the planing for recommendation system:
1. Analyze the target user's historical interactions and reviews. The goal of this step is to understand and infer user preferences and behavior patterns.
2. Analyze the candidate items' attributes. The goal of this step is to analyze the items' features.
3. Analyze the potiential user-item compatibility. The goal of this step is to quantify user-item compatibility according to the user's historical preferences and the candidate items' attributes.
4. Rank items based on the compatibility scores.

# Here is the task description:
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
            n=n
        )
        
        return reasoning_result

class CoTExampleReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str, temperature: float = 0.1, n: int = 1):
        """Override the parent class's __call__ method"""
        prompt = '''#Here is the planing for recommendation system:
1. Analyze the target user's historical interactions and reviews. The goal of this step is to understand and infer user preferences and behavior patterns.
2. Analyze the candidate items' attributes. The goal of this step is to analyze the items' features.
3. Analyze the potiential user-item compatibility. The goal of this step is to quantify user-item compatibility according to the user's historical preferences and the candidate items' attributes.
4. Rank items based on the compatibility scores.

# Here is the task description:
{task_description}

# Here is an example of how to complete the task:
Step1: I need to analyze the user's history: their past reviews and star ratings. From history review, I can infer what aspects they care about. For example, if they often mention "durable" and "long-lasting" in positive reviews, maybe they prioritize product durability. If they give high stars to items with good customer service, that's another factor.
Step2: the candidate items are listed in item list, and their info is followed. I need to go through each item's details. Let's say the items are electronics. The user's history shows they value battery life and ease of use. So, items with longer battery life and user-friendly features should rank higher.
Step3: I should compare each item's attributes against the user's preferences. If an item has features that align with the user's past positive reviews, it gets a higher spot. Also, considering the star ratings from similar users might help. If an item has many 5-star reviews mentioning aspects the user cares about, that's a good sign.
Step4: the problem says Just the ranked list. So the example should directly show the final list without explanations. The example given in the correct format is ['B07XG8C8Z9', 'B08L5WRWNW', 'B08L5VVLSL'], which matches the required structure. I need to make sure the output strictly follows that format, only the item IDs in order, no extra text.
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
            n=n
        )
        
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.cot_reasoning = CoTReasoning(profile_type_prompt='', llm=self.llm)
        self.cot_example_reasoning = CoTExampleReasoning(profile_type_prompt='', llm=self.llm)

    def parse_result(self, result: str) -> list[str]:
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                raise ValueError("No list found.")
            return eval(result)
        except:
            print('format error')
            return ['']
    
    def borda(self, results: list[list[str]], item_list: list[str]) -> list[str]:
        m = len(item_list) # number of candidates
        score = np.zeros(m)
        for result in results:
            for i, item_id in enumerate(result):
                if item_id not in item_list:
                    continue
                idx = item_list.index(item_id)
                score[idx] += m - i
        ranked_idx = np.argsort(-score)
        return [item_list[i] for i in ranked_idx]

    def multi_parse(self, task_description: str, temperature: float = 0.1, n: int = 1) -> list[str] | list[list[str]]:
        if n == 1:
            result = self.cot_reasoning(task_description=task_description, temperature=temperature, n=1)
            # result = self.cot_example_reasoning(task_description=task_description, temperature=temperature, n=1)
            return self.parse_result(result)
        else:
            results = self.cot_reasoning(task_description=task_description, temperature=temperature, n=n)
            # results = self.cot_example_reasoning(task_description=task_description, temperature=temperature, n=n)
            return [self.parse_result(r) for r in results]

    def get_item_info(self, item_id):
        item = self.interaction_tool.get_item(item_id=item_id)
        source = item['source']
        if source == 'yelp':
            keys_to_extract = list(item.keys())
        elif source == 'amazon':
            keys_to_extract = list(item.keys())
            keys_to_extract.remove('images')
            keys_to_extract.remove('videos')
        elif source == 'goodreads':
            keys_to_extract = list(item.keys())
            keys_to_extract.remove('popular_shelves')
        filtered_item = {key: item[key] for key in keys_to_extract if key in item}
        return filtered_item

    def workflow_agent(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        plan = [
            {'description': 'First, I need to find item information'},
            {'description': 'Next, I need to find review information'}
        ]
        user_meta = self.interaction_tool.get_user(user_id=self.task['user_id'])
        source = user_meta['source']

        item_list = []
        for sub_task in plan:
            if 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    filtered_item = self.get_item_info(self.task['candidate_list'][n_bus])
                    item_list.append(filtered_item)
            elif 'review' in sub_task['description']:
                history_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                history_ids = [review['item_id'] for review in history_reviews]
                if source == 'yelp':
                    keys_to_extract = ['item_id', 'funny', 'cool', 'text', 'date', 'source', 'type', 'stars', 'useful']
                elif source == 'amazon':
                    keys_to_extract = ['item_id', 'text', 'timestamp', 'source', 'type', 'sub_item_id', 'stars', 'helpful_vote', 'verified_purchase', 'title']
                elif source == 'goodreads':
                    keys_to_extract = ['item_id', 'date_added', 'date_updated', 'source', 'type', 'read_at', 'stars', 'n_votes', 'n_comments', 'text', 'started_at']
                else:
                    pass
                history_reviews = [ {key: review[key] for key in keys_to_extract if key in review} for review in history_reviews]
                history_info = [self.get_item_info(item_id) for item_id in history_ids]
                history_str = str(history_info) + str(history_reviews)
                while num_tokens_from_string(history_str) > 12000:
                    history_info = history_info[:-1]
                    history_reviews = history_reviews[:-1]
                    history_str = str(history_info) + str(history_reviews)
            else:
                pass       
        
        history_reviews = '\n'.join([f'{i}. {review}' for i, review in enumerate(history_reviews, 1)])
        history_info = '\n'.join([f'{i}. {info}' for i, info in enumerate(history_info, 1)])
        task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_reviews}.
The information of your historical items is as follows (the same order as above)
{history_info}

Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
Please rank the more interested items more front in your rank list.
The information of the above 20 candidate items is as follows: {item_list}.

Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
DO NOT output your analysis process!

The correct output format:
['item id1', 'item id2', 'item id3', ...]
'''
        # cut max_tokens_len 32768
        if num_tokens_from_string(task_description) > 31000:
            exceed_tokens = num_tokens_from_string(task_description) - 31000
            longest_item_idx = max(range(len(item_list)), key=lambda x: num_tokens_from_string(str(item_list[x])))
            encoding = tiktoken.get_encoding("cl100k_base")
            item_list[longest_item_idx] = encoding.decode(encoding.encode(str(item_list[longest_item_idx]))[:num_tokens_from_string(str(item_list[longest_item_idx])) - exceed_tokens])

            task_description = f'''You are a real user on an online platform. Your historical item review text and stars are as follows: {history_reviews}.
The information of your historical items is as follows (the same order as above)
{history_info}

Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
Please rank the more interested items more front in your rank list.
The information of the above 20 candidate items is as follows: {item_list}.

Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
DO NOT output your analysis process!

The correct output format:
['item id1', 'item id2', 'item id3', ...]
'''
        
        return task_description

    def avg_rating_sort(self, source) -> list[str]:
        item_list = self.task['candidate_list']
        item_avg_rating = []
        for n_bus in range(len(self.task['candidate_list'])):
            item_info = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
            if source == "yelp":
                item_avg_rating.append(item_info['stars'])
            elif source == "amazon":
                item_avg_rating.append(item_info['average_rating'])
            elif source == "goodreads":
                item_avg_rating.append(float(item_info['average_rating']))
        sorted_item_list = [x for _, (y, x) in sorted(enumerate(zip(item_avg_rating, item_list)), key=lambda pair: (-pair[1][0], pair[0]))]
        return sorted_item_list

    def pop_filter_sort(self, source) -> list[str]:
        item_list = self.task['candidate_list']
        item_popularity = []
        for n_bus in range(len(self.task['candidate_list'])):
            item_review = self.interaction_tool.get_reviews(item_id=self.task['candidate_list'][n_bus])
            # filter the reviews
            item_review = [review for review in item_review if review['stars'] >= 5]
            item_popularity.append(len(item_review))
        sorted_item_list = [x for _, (y, x) in sorted(enumerate(zip(item_popularity, item_list)), key=lambda pair: (-pair[1][0], pair[0]))]
        return sorted_item_list

    def workflow_prerank(self, source) -> list[str]:
        # return self.pop_filter_sort(source)
        return self.avg_rating_sort(source)

    def workflow(self):
        source = self.interaction_tool.get_user(user_id=self.task['user_id'])['source']
        pop_rankings = self.pop_filter_sort(source)
        rating_rankings = self.avg_rating_sort(source)
        
        # prerank
        self.task['candidate_list'] = self.workflow_prerank(source)
        # agent
        task_description = self.workflow_agent()
        # ensemble
        results = self.multi_parse(task_description, temperature=0.5, n=3)
        origin_result = self.borda(results, self.task['candidate_list'])
        pop_result = self.borda(results + [pop_rankings], self.task['candidate_list'])
        rating_result = self.borda(results + [rating_rankings], self.task['candidate_list'])
        return {
            'origin': origin_result,
            'pop': pop_result,
            'rating': rating_result,
            'raw': {
                'origin': results,
                'pop': results + [pop_rankings],
                'rating': results + [rating_rankings]
            }
        }
