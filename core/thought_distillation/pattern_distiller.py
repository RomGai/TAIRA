import json
import os
from datetime import datetime
from typing import Dict, Optional
from utils.task import get_completion


class PatternDistiller:
    """
    Implements Thought Pattern Distillation (TPD) from TAIRA paper.
    Extracts and refines thought patterns from:
    1. Successful agent executions
    2. Expert-corrected failures
    """

    def __init__(self, storage_path: str = "storage/thought_patterns"):
        self.storage_path = storage_path
        self.patterns_file = os.path.join(storage_path, "patterns.json")
        self.log_file = os.path.join(storage_path, "distillation_log.json")

        os.makedirs(storage_path, exist_ok=True)
        self.patterns = self._load_patterns()
        self.distillation_log = self._load_log()

    def _load_patterns(self) -> Dict:
        """Load existing patterns."""
        if os.path.exists(self.patterns_file):
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_log(self) -> list:
        """Load distillation log."""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_patterns(self):
        """Persist patterns to storage."""
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, indent=2, ensure_ascii=False)

    def _save_log(self):
        """Persist distillation log."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.distillation_log, f, indent=2, ensure_ascii=False)

    def distill_from_success(
        self,
        task_route: Dict,
        user_query: str,
        similar_pattern_key: Optional[str] = None
    ) -> str:
        """
        Extract thought pattern from successful execution.

        Args:
            task_route: Execution path with tasks and results
            user_query: Original user query
            similar_pattern_key: Key of similar existing pattern (optional)

        Returns:
            Key of created/updated pattern
        """
        sys_prompt = (
            "You are a thought pattern creator specializing in distilling "
            "high-level cognitive processes from successful task executions.\n\n"
            "A thought pattern contains three parts:\n"
            "1. task_description: Abstract characterization of the task type "
            "(no specific details, only general category)\n"
            "2. solution_description: High-level problem-solving approach "
            "(conceptual guidance, like expert advice)\n"
            "3. thought_template: Step-by-step execution template "
            "(concrete action sequence)\n"
        )

        prompt = (
            f"User query: {user_query}\n"
            f"Successful task route:\n{json.dumps(task_route, indent=2, ensure_ascii=False)}\n\n"
            "Extract a reusable thought pattern from this success.\n"
            "Output JSON format:\n"
            "{\n"
            "  \"task_description\": \"...\",\n"
            "  \"solution_description\": \"...\",\n"
            "  \"thought_template\": \"...\"\n"
            "}\n"
        )

        if similar_pattern_key and similar_pattern_key in self.patterns:
            old_pattern = self.patterns[similar_pattern_key]
            prompt += (
                f"\n\nA similar pattern exists:\n"
                f"{json.dumps(old_pattern, indent=2, ensure_ascii=False)}\n\n"
                "Consider refining this pattern or creating a new one if "
                "the task differs significantly."
            )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        response = get_completion(messages)

        # Parse and save pattern
        try:
            new_pattern = json.loads(response)
            pattern_key = self._generate_pattern_key()
            self.patterns[pattern_key] = new_pattern

            # Log distillation
            self.distillation_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "pattern_key": pattern_key,
                "source_query": user_query
            })

            self._save_patterns()
            self._save_log()

            return pattern_key
        except json.JSONDecodeError:
            print(f"Failed to parse pattern from response: {response}")
            return None

    def distill_from_failure(
        self,
        task_route: Dict,
        expert_correction: str,
        failed_pattern_key: str
    ) -> str:
        """
        Refine pattern based on expert-corrected failure.

        Args:
            task_route: Failed execution path
            expert_correction: Expert's explanation of what went wrong
            failed_pattern_key: Key of the pattern that failed

        Returns:
            Key of updated pattern
        """
        if failed_pattern_key not in self.patterns:
            print(f"Pattern {failed_pattern_key} not found")
            return None

        old_pattern = self.patterns[failed_pattern_key]

        sys_prompt = (
            "You are a thought pattern refiner. You learn from failures "
            "with expert guidance to improve existing patterns.\n\n"
            "A thought pattern contains:\n"
            "1. task_description (abstract task type)\n"
            "2. solution_description (high-level approach)\n"
            "3. thought_template (execution steps)\n"
        )

        prompt = (
            f"Failed task route:\n{json.dumps(task_route, indent=2, ensure_ascii=False)}\n\n"
            f"Expert correction: {expert_correction}\n\n"
            f"Current pattern to refine:\n{json.dumps(old_pattern, indent=2, ensure_ascii=False)}\n\n"
            "Revise the pattern to address the failure. "
            "Output the improved pattern in JSON format:\n"
            "{\n"
            "  \"task_description\": \"...\",\n"
            "  \"solution_description\": \"...\",\n"
            "  \"thought_template\": \"...\"\n"
            "}\n"
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        response = get_completion(messages)

        try:
            refined_pattern = json.loads(response)
            self.patterns[failed_pattern_key] = refined_pattern

            # Log refinement
            self.distillation_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": "failure_correction",
                "pattern_key": failed_pattern_key,
                "expert_correction": expert_correction
            })

            self._save_patterns()
            self._save_log()

            return failed_pattern_key
        except json.JSONDecodeError:
            print(f"Failed to parse refined pattern: {response}")
            return None

    def _generate_pattern_key(self) -> str:
        """Generate unique pattern key."""
        existing_nums = []
        for key in self.patterns.keys():
            if key.startswith("template_"):
                try:
                    num = int(key.split("_")[1])
                    existing_nums.append(num)
                except:
                    pass

        next_num = max(existing_nums) + 1 if existing_nums else 0
        return f"template_{next_num}"

    def export_patterns_to_code(self, output_path: str = "utils/thought_template.py"):
        """Export patterns as Python code for backward compatibility."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("thought_templates = ")
            f.write(json.dumps(self.patterns, indent=4, ensure_ascii=False))
            f.write("\n")
