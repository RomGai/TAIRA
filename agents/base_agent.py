from abc import ABC, abstractmethod
from utils.task import get_completion


class Agent(ABC):
    """Base class for all agents in TAIRA system."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def execute_task(self, task):
        """Execute the assigned task."""
        pass

    def call_gpt(self, messages, llm=None, json_format=None):
        """Call LLM for task execution."""
        return get_completion(messages, json_format, llm)
