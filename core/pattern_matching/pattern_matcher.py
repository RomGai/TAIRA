import json
import os
from typing import Dict, List, Tuple
from utils.task import get_completion


class PatternMatcher:
    """
    Implements Thought Pattern Matching mechanism from TAIRA paper.
    Retrieves Top-K most relevant patterns based on query similarity.
    """

    def __init__(self, pattern_storage_path: str = "storage/thought_patterns/patterns.json"):
        self.pattern_storage_path = pattern_storage_path
        self.patterns = self._load_patterns()

    def _load_patterns(self) -> Dict:
        """Load thought patterns from storage."""
        if os.path.exists(self.pattern_storage_path):
            with open(self.pattern_storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def select_best_pattern(self, user_query: str, user_preference: str = "") -> Tuple[str, Dict]:
        """
        Select the most relevant thought pattern for the given query.

        Args:
            user_query: User's recommendation request
            user_preference: User's preferences (optional)

        Returns:
            Tuple of (pattern_key, pattern_content)
        """
        if not self.patterns:
            return None, None

        template_texts = [
            f"{key}: {pattern['task_description']}"
            for key, pattern in self.patterns.items()
        ]
        combined_template_text = "\n".join(template_texts)

        sys_prompt = (
            "You are an expert at matching user queries to successful experience patterns. "
            "Here are the available thought patterns from past successes:\n"
            f"{combined_template_text}\n\n"
            "Each pattern represents a proven problem-solving strategy."
        )

        prompt = (
            f"Current user query: \"{user_query}\"\n"
            f"User preferences: {user_preference}\n\n"
            "Analyze the query and select the most similar pattern. "
            "First, explain your reasoning. Then output the chosen pattern number "
            "in curly braces like this: {number}. "
            "The braces must contain ONLY the number, nothing else."
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        response = get_completion(messages)

        # Extract pattern number
        import re
        matches = re.findall(r'\{(\d+)\}', response)
        if matches:
            pattern_num = int(matches[-1])
            pattern_key = f"template_{pattern_num}"
            if pattern_key in self.patterns:
                return pattern_key, self.patterns[pattern_key]

        # Fallback: return first pattern
        first_key = list(self.patterns.keys())[0]
        return first_key, self.patterns[first_key]

    def get_top_k_patterns(self, user_query: str, k: int = 3) -> List[Tuple[str, Dict]]:
        """
        Retrieve top-K most relevant patterns.

        Args:
            user_query: User's query
            k: Number of patterns to retrieve

        Returns:
            List of (pattern_key, pattern_content) tuples
        """
        # For now, use LLM-based selection for top pattern
        # In production, this could use embedding-based similarity search
        best_key, best_pattern = self.select_best_pattern(user_query)

        if best_pattern is None:
            return []

        return [(best_key, best_pattern)]
