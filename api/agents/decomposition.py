import time
import json
from langchain_core.messages import HumanMessage, SystemMessage
from api.agents.base_agent import BaseAgent
from api.core.shared_context import AgentOutput, SubTask
from api.core.context_budget import ContextBudgetManager
from api.core.shared_context import SharedContext

DECOMPOSITION_SYSTEM_PROMPT = """You are a decomposition agent. Break the user query into typed sub-tasks with explicit dependencies.

Output ONLY valid JSON:
{
  "sub_tasks": [
    {
      "id": "t1",
      "type": "factual|reasoning|retrieval|code",
      "description": "specific task description",
      "dependencies": []
    }
  ]
}

Rules:
- Every sub-task must have a unique id starting with "t"
- Dependencies list contains ids of tasks that MUST complete before this one starts
- Types: factual (fact lookup), reasoning (logical inference), retrieval (needs document search), code (needs execution)
- Create at least 2 sub-tasks for any non-trivial query
- Dependent tasks must reference valid task ids defined earlier in the list
- If query is ambiguous, create a clarification sub-task first"""


class DecompositionAgent(BaseAgent):
    agent_id = "decomposition"
    max_budget = 3072

    def run(self) -> AgentOutput:
        self.context.add_sse_event("agent_start", agent_id=self.agent_id)
        start = time.monotonic()

        prompt = f"Break this query into typed sub-tasks with dependencies:\n\n{self.context.original_query}"
        remaining = self.budget_manager.check_remaining(self.agent_id)
        if self._count_tokens(DECOMPOSITION_SYSTEM_PROMPT + prompt) > remaining:
            prompt = prompt[:800]

        messages = [
            SystemMessage(content=DECOMPOSITION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            # Extract JSON even if wrapped in markdown fences
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            sub_tasks = [SubTask(**t) for t in parsed.get("sub_tasks", [])]
        except Exception:
            sub_tasks = [
                SubTask(id="t1", type="retrieval",
                        description=f"Retrieve relevant information for: {self.context.original_query}",
                        dependencies=[]),
                SubTask(id="t2", type="reasoning",
                        description="Analyze findings and form a conclusion",
                        dependencies=["t1"]),
            ]

        self.context.sub_tasks = sub_tasks
        self.context.dependency_graph = {t.id: t.dependencies for t in sub_tasks}

        latency = int((time.monotonic() - start) * 1000)
        summary = (
            f"Decomposed into {len(sub_tasks)} sub-tasks: "
            + ", ".join(f"{t.id}({t.type})" for t in sub_tasks)
        )
        return self._record_output(summary, latency)
