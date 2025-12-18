import json
import logging
from typing import Dict, Optional
from core.pattern_matching.pattern_matcher import PatternMatcher
from core.hierarchical_planning.hierarchical_planner import HierarchicalPlanner
from core.thought_distillation.pattern_distiller import PatternDistiller


class TAIRAManager:
    """
    Core Manager implementing TAIRA's complete workflow:
    1. Thought Pattern Matching
    2. Hierarchical Planning
    3. Multi-Agent Execution
    4. Thought Pattern Distillation (Learning)
    """

    def __init__(
        self,
        memory,
        user_input: str,
        target_product: str,
        targets: str,
        target_count: int,
        preference: str,
        config: Dict,
        logger: Optional[logging.Logger] = None
    ):
        self.memory = memory
        self.user_input = user_input
        self.target_product = target_product
        self.targets = targets
        self.target_count = target_count
        self.preference = preference
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize TAIRA core components
        self.pattern_matcher = PatternMatcher()
        self.hierarchical_planner = HierarchicalPlanner(config, logger)
        self.pattern_distiller = PatternDistiller()

        self.agents = {}
        self.turn = 0
        self.execution_log = []

    def register_agent(self, agent):
        """Register an executor agent."""
        self.agents[agent.name] = agent

    def delegate_task(self):
        """
        Main TAIRA workflow execution.
        Returns: (hit_rate, mrr, ndcg, fail_flag, pattern_key)
        """
        from user_simulate.evaluate_agent import EvaluateAgent

        self.logger.debug(f"User query: {self.user_input}")
        self.memory.add_input(self.user_input)

        # Phase 1: Thought Pattern Matching
        pattern_key, thought_pattern = None, None
        if self.config.get('METHOD') == 'TAIRA':
            self.logger.debug("Phase 1: Pattern Matching...")
            pattern_key, thought_pattern = self.pattern_matcher.select_best_pattern(
                self.user_input,
                self.preference
            )
            if pattern_key:
                self.logger.debug(f"Selected pattern: {pattern_key}")
                self.logger.debug(f"Pattern: {json.dumps(thought_pattern, indent=2, ensure_ascii=False)}")

        # Phase 2: Hierarchical Planning (Initial)
        self.logger.debug("Phase 2: Creating initial plan...")
        initial_plan = self.hierarchical_planner.create_initial_plan(
            self.user_input,
            self.preference,
            thought_pattern
        )

        self.memory.add_plan(initial_plan)
        self.logger.debug(f"Initial plan: {initial_plan}")

        # Phase 3: Multi-Agent Execution with Hierarchical Re-planning
        self.logger.debug("Phase 3: Executing plan...")
        result = self._execute_hierarchical_plan(initial_plan, thought_pattern)

        # Phase 4: Thought Pattern Distillation (if enabled)
        hit_rate, mrr, ndcg, fail_flag = result

        if self.config.get('ENABLE_LEARNING', False):
            self._distill_experience(hit_rate > 0, pattern_key)

        return hit_rate, mrr, ndcg, fail_flag, pattern_key or "unknown"

    def _execute_hierarchical_plan(self, plan_json_str: str, thought_pattern: Optional[Dict]):
        """Execute plan with hierarchical re-planning support."""
        from user_simulate.evaluate_agent import EvaluateAgent

        evaluator = EvaluateAgent(self.memory, self.logger, self.config)
        task_plan = json.loads(plan_json_str)

        interpreter = self.agents.get("InterpreterAgent")
        interactor = self.agents.get("InteractorAgent")

        output = ""
        last_agent_name = ""

        while last_agent_name != "InteractorAgent":
            if self.turn >= 10:
                self.logger.debug("Max turns reached - FAIL")
                return 0, 0, 0, True

            # Execute non-terminal sub-tasks
            sub_tasks = task_plan.get("sub_tasks", {})
            task_keys = list(sub_tasks.keys())

            for i, task_key in enumerate(task_keys[:-1]):
                sub_task = sub_tasks[task_key]
                content = sub_task["content"]
                agent_name = sub_task["agent"]

                query = interpreter.process_output(content, agent_name, output)
                agent = self.agents.get(agent_name)

                if not agent:
                    self.logger.error(f"Agent {agent_name} not found")
                    return 0, 0, 0, True

                self.turn += 1
                output = str(agent.execute_task(query))

                self.memory.add_observation(agent_name, query, output)
                self.execution_log.append({
                    "turn": self.turn,
                    "agent": agent_name,
                    "task": query,
                    "output": output
                })

                self.logger.debug(f"Turn {self.turn} - Agent: {agent_name}")
                self.logger.debug(f"Task: {query}")
                self.logger.debug(f"Output: {output}")

            # Handle terminal task
            last_task = sub_tasks[task_keys[-1]]
            last_agent_name = last_task["agent"]
            last_content = last_task["content"]

            if last_agent_name == "InteractorAgent":
                # Final recommendation
                query = interpreter.process_output(last_content, last_agent_name, output)
                self.logger.debug(f"Final recommendation query: {query}")

                result = interactor.generate_response(query)
                self.logger.debug(f"Recommendation: {result}")

                # Evaluate
                import re
                items_match = re.search(r'\{.*\}', result, re.DOTALL)
                if items_match:
                    rec_json = json.loads(items_match.group(0))
                    hit_rate, mrr, ndcg, fail_flag = evaluator.evaluate(
                        self.user_input,
                        rec_json,
                        self.target_product,
                        self.targets,
                        self.target_count,
                        preference=self.preference
                    )

                    self.logger.debug(f"Hit Rate: {hit_rate}, MRR: {mrr}, NDCG: {ndcg}")
                    return hit_rate, mrr, ndcg, fail_flag or (hit_rate == 0)
                else:
                    return 0, 0, 0, True

            elif last_agent_name == "PlannerAgent":
                # Hierarchical re-planning
                query = interpreter.process_output(last_content, last_agent_name, output)
                self.logger.debug(f"Re-planning with goal: {query}")

                new_plan = self.hierarchical_planner.update_plan(
                    query,
                    self.memory.get_history(),
                    self.preference,
                    thought_pattern
                )

                self.logger.debug(f"Updated plan: {new_plan}")
                self.memory.add_plan(new_plan)

                # Merge plans
                new_tasks = json.loads(new_plan)
                task_plan["sub_tasks"].update(new_tasks["sub_tasks"])

    def _distill_experience(self, success: bool, used_pattern_key: Optional[str]):
        """
        Distill experience into thought patterns (TPD).

        Args:
            success: Whether execution succeeded
            used_pattern_key: Pattern key that was used (if any)
        """
        task_route = {
            "query": self.user_input,
            "execution_log": self.execution_log,
            "memory_history": self.memory.get_history()
        }

        if success:
            self.logger.debug("Learning from success - distilling pattern...")
            new_key = self.pattern_distiller.distill_from_success(
                task_route,
                self.user_input,
                used_pattern_key
            )
            if new_key:
                self.logger.debug(f"Created/updated pattern: {new_key}")
        else:
            # In real system, expert_correction would come from human feedback
            # For now, skip or use placeholder
            pass
