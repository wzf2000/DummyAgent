from typing import Any, Dict, List, Optional

class RecommendationTask:
    def __init__(self, user_id: str,  
                 candidate_category: str,
                 candidate_list: List[str],
                 loc: List[float]):
        """
        Recommendation Task for the RecommendationAgent.
        Args:
            user_id: The ID of the user requesting recommendations.
            candidate_category: The category of the candidate items.
            candidate_list: List of candidate item IDs.
            loc: User's location as [latitude, longitude]. If is [-1, -1], the user is not in a specific location.
        """
        self.user_id = user_id
        self.candidate_category = candidate_category
        self.candidate_list = candidate_list
        self.loc = loc

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        Returns:
            dict: The task in dictionary format.
        """
        return {
            "description": """This is a recommendation task. 
            You are a recommendation agent that recommends items to users. 
            There is a user with id and a list of items with category and ids. 
            The location of the user is set only if it is not [-1, -1].""",
            "user_id": self.user_id,
            "candidate_category": self.candidate_category,
            "candidate_list": self.candidate_list,
            "loc": self.loc
        }