"""
Planner module for the agentic vision system.
Converts agent decisions into actionable plans.
"""
from typing import List, Dict, Any

class ActionPlan:
    """
    Represents a plan of actions to execute.
    """
    def __init__(self, actions: List[Dict[str, Any]]):
        self.actions = actions

    def __iter__(self):
        return iter(self.actions)

class SafetyPlanner:
    """
    Planner that takes agent decisions and creates a list of actions to perform.
    """
    def __init__(self):
        pass

    def plan(self, decision: Dict[str, Any]) -> ActionPlan:
        """
        Convert a single agent decision into an action plan.

        Args:
            decision: Dictionary from the agent with keys:
                'decision', 'reason', 'person_id', 'next_action', 'parameters'

        Returns:
            ActionPlan containing a list of action dictionaries.
        """
        # For now, we assume each decision maps to a single action.
        # In the future, we could decompose complex decisions into multiple steps.
        action = {
            'action_type': decision['next_action'],
            'person_id': decision['person_id'],
            'parameters': decision['parameters'],
            'reason': decision['reason']
        }
        return ActionPlan([action])

    def plan_multiple(self, decisions: List[Dict[str, Any]]) -> List[ActionPlan]:
        """
        Plan for multiple decisions.

        Args:
            decisions: List of agent decisions.

        Returns:
            List of ActionPlan objects.
        """
        return [self.plan(d) for d in decisions]