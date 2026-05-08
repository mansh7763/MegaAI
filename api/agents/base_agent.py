import hashlib
import time
from abc import ABC, abstractmethod
from langchain_groq import ChatGroq
from api.core.shared_context import SharedContext, AgentOutput
from api.core.context_budget import ContextBudgetManager
from api.config import get_settings

settings = get_settings()


def get_llm(temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
    )


class BaseAgent(ABC):
    agent_id: str = "base"
    max_budget: int = 4096

    def __init__(self, context: SharedContext, budget_manager: ContextBudgetManager):
        self.context = context
        self.budget_manager = budget_manager
        self.llm = get_llm()
        self.budget_manager.declare_budget(self.agent_id, self.max_budget)

    def _count_tokens(self, text: str) -> int:
        # Rough estimate: 1 token ≈ 4 chars
        return max(1, len(text) // 4)

    def _record_output(self, content: str, latency_ms: int, citations=None) -> AgentOutput:
        token_count = self._count_tokens(content)
        output = AgentOutput(
            agent_id=self.agent_id,
            content=content,
            citations=citations or [],
            token_count=token_count,
            latency_ms=latency_ms,
        )
        self.context.agent_outputs[self.agent_id] = output
        self.budget_manager.consume(self.agent_id, token_count)
        self.context.add_sse_event(
            "agent_done",
            agent_id=self.agent_id,
            output_hash=output.output_hash,
            token_count=token_count,
            latency_ms=latency_ms,
        )
        return output

    @abstractmethod
    def run(self) -> AgentOutput:
        pass
