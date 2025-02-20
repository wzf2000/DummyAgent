from typing import Dict

class SimulationTask:
    def __init__(self, user_id: str, item_id: str):
        """
        Simulation Task for the SimulationAgent.
        Args:
            time: The time parameter to limit InteractionTool behavior.
            item: The item receiving the simulated review.
        """
        self.user_id = user_id
        self.item_id = item_id

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the task to a dictionary.
        Returns:
            dict: The task in dictionary format.
        """
        return {
            "description": """This is a simulation task. 
            You are a simulation agent that simulates a user's rating and review with an item. 
            There is a user with id and an item with id. """,
            "user_id": self.user_id,
            "item_id": self.item_id
        }
