from api.core.shared_context import SharedContext, PolicyViolation


class ContextBudgetManager:
    DEFAULT_BUDGETS = {
        "orchestrator": 4096,
        "decomposition": 3072,
        "rag_agent": 5120,
        "critique": 3072,
        "synthesis": 4096,
        "compression": 2048,
        "meta_agent": 3072,
    }

    def __init__(self, context: SharedContext):
        self.context = context

    def declare_budget(self, agent_id: str, max_tokens: int):
        self.context.max_budget[agent_id] = max_tokens
        if agent_id not in self.context.context_budget:
            self.context.context_budget[agent_id] = 0

    def declare_all_defaults(self):
        for agent_id, budget in self.DEFAULT_BUDGETS.items():
            self.declare_budget(agent_id, budget)

    def check_remaining(self, agent_id: str) -> int:
        max_b = self.context.max_budget.get(agent_id, 0)
        used = self.context.context_budget.get(agent_id, 0)
        return max(0, max_b - used)

    def consume(self, agent_id: str, tokens: int) -> bool:
        """Returns True if within budget, False and logs violation if over."""
        remaining = self.check_remaining(agent_id)
        if tokens > remaining:
            self.context.policy_violations.append(
                PolicyViolation(
                    agent_id=agent_id,
                    violation_type="budget_overflow",
                    details=f"Attempted to consume {tokens} tokens, only {remaining} remaining",
                )
            )
            self.context.context_budget[agent_id] = self.context.max_budget.get(agent_id, 0)
            return False
        self.context.context_budget[agent_id] = self.context.context_budget.get(agent_id, 0) + tokens
        return True

    def consume_without_check(self, agent_id: str, tokens: int):
        """Called when agent bypassed check_remaining — always logs policy violation."""
        self.context.policy_violations.append(
            PolicyViolation(
                agent_id=agent_id,
                violation_type="budget_policy_skip",
                details=f"Agent consumed {tokens} tokens without calling check_remaining first",
            )
        )
        self.context.context_budget[agent_id] = (
            self.context.context_budget.get(agent_id, 0) + tokens
        )

    def is_over_budget(self, agent_id: str) -> bool:
        used = self.context.context_budget.get(agent_id, 0)
        max_b = self.context.max_budget.get(agent_id, float("inf"))
        return used >= max_b
