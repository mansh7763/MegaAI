import time
import json
from langchain_core.messages import HumanMessage, SystemMessage
from api.agents.base_agent import BaseAgent
from api.core.shared_context import FinalAnswer, ProvenanceEntry, AgentOutput
from api.core.context_budget import ContextBudgetManager
from api.core.shared_context import SharedContext

SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis agent. Your tasks:
1. Merge all agent outputs into one coherent, complete final answer
2. For each flagged contradiction from the critique agent, explicitly resolve it and state how
3. Build a provenance map: for each sentence in your final answer, state which agent and chunk_id it came from

Output ONLY valid JSON:
{
  "final_answer": "The complete, coherent answer. Each sentence should be traceable.",
  "contradictions_resolved": [
    "Contradiction: 'X said Y'. Resolution: Based on source Z, the correct statement is W."
  ],
  "provenance_map": [
    {
      "sentence": "exact sentence from final_answer",
      "source_agent": "agent_id",
      "source_chunk_id": "chunk_id_or_null"
    }
  ]
}

Rules:
- Final answer must be self-contained and complete
- Resolve EVERY flagged span from the critique — do not surface unresolved contradictions
- Provenance map must cover all key sentences (at least one per paragraph)"""


class SynthesisAgent(BaseAgent):
    agent_id = "synthesis"
    max_budget = 4096

    def run(self) -> AgentOutput:
        self.context.add_sse_event("agent_start", agent_id=self.agent_id)
        start = time.monotonic()

        agent_summaries = "\n\n".join([
            f"[{aid}]: {out.content[:700]}"
            for aid, out in self.context.agent_outputs.items()
            if aid not in ("synthesis", "orchestrator")
        ])

        flagged_text = ""
        if self.context.critique_report and self.context.critique_report.flagged_spans:
            flagged_text = "=== Contradictions to resolve ===\n" + "\n".join([
                f"- Span: '{f['span'][:100]}' | Reason: {f['reason']}"
                for f in self.context.critique_report.flagged_spans
            ])

        prompt = (
            f"Original query: {self.context.original_query}\n\n"
            f"Agent outputs to merge:\n{agent_summaries}\n\n"
            f"{flagged_text}"
        )

        remaining = self.budget_manager.check_remaining(self.agent_id)
        if self._count_tokens(prompt) > remaining - 900:
            prompt = prompt[: remaining * 3]

        messages = [SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            provenance = [ProvenanceEntry(**p) for p in parsed.get("provenance_map", [])]
            final = FinalAnswer(
                content=parsed.get("final_answer", content),
                provenance_map=provenance,
                contradictions_resolved=parsed.get("contradictions_resolved", []),
            )
        except Exception:
            # Fallback: concatenate agent outputs
            content_parts = [
                f"{out.content[:400]}"
                for aid, out in self.context.agent_outputs.items()
                if aid not in ("synthesis", "orchestrator")
            ]
            combined = " ".join(content_parts)
            final = FinalAnswer(
                content=combined,
                provenance_map=[
                    ProvenanceEntry(
                        sentence=out.content[:80],
                        source_agent=aid,
                        source_chunk_id=out.citations[0].chunk_id if out.citations else None,
                    )
                    for aid, out in self.context.agent_outputs.items()
                    if aid not in ("synthesis", "orchestrator")
                ],
            )

        self.context.final_answer = final
        latency = int((time.monotonic() - start) * 1000)
        return self._record_output(final.content, latency)
