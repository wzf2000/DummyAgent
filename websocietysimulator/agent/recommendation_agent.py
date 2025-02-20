from typing import List, Dict, Any
from .agent import Agent
from ..llm import LLMBase

class RecommendationAgent(Agent):
    def __init__(self, llm: LLMBase):
        """
        RecommendationAgent initialization.
        Args:
            data_dir: Directory containing Yelp dataset files.
        """
        super().__init__(llm=llm)
        self.task = None

    def insert_task(self, task):
        """
        Insert a recommendation task.
        Args:
            task: An instance of RecommendationTask.
        """
        if not task:
            raise ValueError("The task cannot be None.")
        self.task = task.to_dict()

    def forward(self) -> List[str]:
        """
        Abstract forward method for RecommendationAgent.
        Participants must override this method to provide a sorted list of candidate POIs.
        Returns:
            list: A sorted list of candidate POIs (business dictionaries).
        """
        raise NotImplementedError("Forward method must be implemented by the participant.")