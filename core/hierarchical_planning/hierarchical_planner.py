import json
import re
from typing import Dict, Optional
from utils.task import get_completion
from utils.Prompts import BoT_AGENTS_INSTRUCTION


class HierarchicalPlanner:
    """
    Implements Hierarchical Planning mechanism from TAIRA paper.
    Generates plans iteratively based on current information and feedback.
    """

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger
        self.method = config.get('METHOD', 'TAIRA')

    def create_initial_plan(
        self,
        user_input: str,
        preference: str,
        thought_pattern: Optional[Dict] = None
    ) -> str:
        """
        Create initial hierarchical plan based on user query and matched pattern.

        Args:
            user_input: User's query
            preference: User's preferences
            thought_pattern: Matched thought pattern (optional)

        Returns:
            JSON string of the plan
        """
        sys_prompt = (
            "You are a Manager Agent in a conversational recommendation system. "
            "You excel at hierarchical task planning and can transfer high-level "
            "thinking processes from past successes to current problems.\n\n"
            "Available agents and their functionalities:\n"
            f"{BoT_AGENTS_INSTRUCTION}"
        )

        prompt = (
            f"User query: \"{user_input}\"\n"
            f"User preferences: {preference}\n\n"
            "Create a hierarchical task plan in JSON format with sub-tasks.\n"
            "Output format:\n"
            "{\n"
            f"  \"user_input\": \"{user_input}\",\n"
            "  \"main_task\": \"...\",\n"
            "  \"sub_tasks\": {\n"
            "    \"task_1\": {\"content\": \"...\", \"agent\": \"...\"},\n"
            "    \"task_2\": {\"content\": \"...\", \"agent\": \"...\"},\n"
            "    ...\n"
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- 'content' describes what the agent should do\n"
            "- 'agent' specifies which agent executes the sub-task\n"
            "- PlannerAgent or InteractorAgent MUST be the last sub-task\n"
            "- No sub-tasks are allowed after PlannerAgent or InteractorAgent\n"
            "- Use InteractorAgent if the task can be completed in one plan\n"
            "- Use PlannerAgent if more information is needed for planning\n"
        )

        if self.method == "TAIRA" and thought_pattern:
            pattern_text = (
                f"Solution Description: {thought_pattern.get('solution_description', '')}\n"
                f"Thought Template: {thought_pattern.get('thought_template', '')}"
            )
            prompt += (
                f"\n\nFollow this proven thought pattern from similar tasks:\n"
                f"{pattern_text}\n\n"
                "Judge if this pattern fits the current problem. "
                "If suitable, follow it closely. If not fully suitable, "
                "adapt the solution_description mindset and imitate the thought_template structure."
            )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        response = get_completion(messages)
        return self._extract_json(response)

    def update_plan(
        self,
        current_goal: str,
        execution_history: list,
        preference: str,
        thought_pattern: Optional[Dict] = None
    ) -> str:
        """
        Update plan based on current execution state (Hierarchical Planning).

        Args:
            current_goal: Current planning goal
            execution_history: History of executed tasks
            preference: User preferences
            thought_pattern: Matched thought pattern (optional)

        Returns:
            JSON string of updated plan
        """
        sys_prompt = (
            "You are a Manager Agent performing hierarchical replanning. "
            "Based on execution history, generate the next phase of the plan.\n\n"
            f"Available agents:\n{BoT_AGENTS_INSTRUCTION}\n\n"
            f"Execution history so far:\n{execution_history}"
        )

        prompt = (
            f"Planning goal: \"{current_goal}\"\n"
            f"User preferences: {preference}\n\n"
            "Generate follow-up sub-tasks in JSON format:\n"
            "{\n"
            "  \"sub_tasks\": {\n"
            "    \"task_1\": {\"content\": \"...\", \"agent\": \"...\"},\n"
            "    ...\n"
            "  }\n"
            "}\n\n"
            "Rules: Same as initial planning. "
            "PlannerAgent/InteractorAgent must be last."
        )

        if self.method == "TAIRA" and thought_pattern:
            pattern_text = (
                f"Solution Description: {thought_pattern.get('solution_description', '')}\n"
                f"Thought Template: {thought_pattern.get('thought_template', '')}"
            )
            prompt += f"\n\nContinue following this pattern:\n{pattern_text}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        response = get_completion(messages)
        return self._extract_json(response)

    def _extract_json(self, text: str) -> str:
        """Extract JSON content from LLM response."""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text
