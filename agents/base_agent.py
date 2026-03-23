from abc import ABC, abstractmethod
from utils.task import get_completion


class Agent(ABC):
    """Base class for all agents in TAIRA system."""

    def __init__(self, name, memory=None):
        self.name = name
        self.memory = memory

    @abstractmethod
    def execute_task(self, task):
        """Execute the assigned task."""
        pass

    def call_gpt(self, messages, llm=None, json_format=None):
        """Call LLM for task execution."""
        return get_completion(messages, llm=llm)
