import time
import json
from langchain_core.messages import HumanMessage, SystemMessage
from api.agents.base_agent import BaseAgent
from api.core.shared_context import AgentOutput, RoutingDecision

ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestrator agent for a multi-agent LLM pipeline.

Available agents:
- decomposition: breaks queries into typed sub-tasks with dependency graphs
- rag_agent: multi-hop retrieval + citation from knowledge base
- critique: per-claim confidence scoring + flagging of specific text spans
- synthesis: merges all outputs, resolves contradictions, builds provenance map
- compression: compresses context when budget is near exhaustion
- done: pipeline is complete, return final answer

Routing rules:
1. Start with decomposition for any complex/multi-part query
2. Use rag_agent if the query requires factual retrieval from documents
3. Use critique AFTER rag_agent or decomposition has run
4. Use synthesis as the final merging step (run once, after critique)
5. Use compression ONLY if a context budget violation has been logged
6. Once synthesis is done, route to done

Output ONLY valid JSON:
{
  "next_agent": "agent_name_from_list_above",
  "justification": "one sentence explaining why this agent is needed next",
  "context_summary": "brief summary of what has been accomplished so far"
}"""


class OrchestratorAgent(BaseAgent):
    agent_id = "orchestrator"
    max_budget = 4096

    def decide_next(self, step: int = 0) -> tuple[str, str]:
        """Returns (next_agent_id, justification). Routing decision is made via LLM."""
        state_summary = self._build_state_summary()

        prompt = (
            f"Step {step}. Query: {self.context.original_query[:300]}\n\n"
            f"Current pipeline state:\n{state_summary}"
        )

        remaining = self.budget_manager.check_remaining(self.agent_id)
        if self._count_tokens(prompt) > remaining - 400:
            prompt = prompt[:remaining * 3]

        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            next_agent = parsed.get("next_agent", "synthesis")
            justification = parsed.get("justification", "No justification provided")
        except Exception:
            next_agent = self._fallback_routing(step)
            justification = f"Fallback routing at step {step} (LLM parse failed)"

        # Log the decision
        decision = RoutingDecision(
            from_agent="orchestrator",
            to_agent=next_agent,
            justification=justification,
        )
        self.context.routing_decisions.append(decision)
        self.context.add_sse_event(
            "routing_decision",
            from_agent="orchestrator",
            to_agent=next_agent,
            justification=justification,
            step=step,
        )

        self.budget_manager.consume(self.agent_id, self._count_tokens(justification))
        return next_agent, justification

    def _build_state_summary(self) -> str:
        completed = list(self.context.agent_outputs.keys())
        violations = len(self.context.policy_violations)
        return json.dumps({
            "completed_agents": completed,
            "has_sub_tasks": len(self.context.sub_tasks) > 0,
            "sub_task_count": len(self.context.sub_tasks),
            "has_retrieval_results": len(self.context.retrieval_results) > 0,
            "retrieval_chunk_count": len(self.context.retrieval_results),
            "has_critique": self.context.critique_report is not None,
            "flagged_spans": len(self.context.critique_report.flagged_spans) if self.context.critique_report else 0,
            "has_final_answer": self.context.final_answer is not None,
            "policy_violations": violations,
            "step": 0,
        })

    def _fallback_routing(self, step: int) -> str:
        completed = set(self.context.agent_outputs.keys())
        if "decomposition" not in completed:
            return "decomposition"
        if "rag_agent" not in completed:
            return "rag_agent"
        if "critique" not in completed:
            return "critique"
        if "synthesis" not in completed:
            return "synthesis"
        return "done"

    def run(self) -> AgentOutput:
        next_agent, justification = self.decide_next()
        return self._record_output(f"Routing to {next_agent}: {justification}", 0)
