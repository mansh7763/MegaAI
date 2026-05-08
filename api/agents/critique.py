import time
import json
from langchain_core.messages import HumanMessage, SystemMessage
from api.agents.base_agent import BaseAgent
from api.core.shared_context import AgentOutput, CritiqueReport, ClaimScore
from api.core.context_budget import ContextBudgetManager
from api.core.shared_context import SharedContext

CRITIQUE_SYSTEM_PROMPT = """You are a critique agent. Your job is to review each agent's output and score individual claims — NOT the output as a whole.

For each claim you identify:
- Assign a confidence score (0.0–1.0): how confident are you this claim is accurate?
- Flag it (flagged: true) if you believe it is incorrect, unsupported, or contradicts another claim

Output ONLY valid JSON:
{
  "reviewed_agents": ["agent_id_1", "agent_id_2"],
  "claim_scores": [
    {
      "span": "exact text of the claim (verbatim from the agent output)",
      "confidence": 0.85,
      "flagged": false,
      "reason": null
    },
    {
      "span": "a problematic claim verbatim",
      "confidence": 0.3,
      "flagged": true,
      "reason": "This contradicts the statement in the RAG output about X"
    }
  ],
  "flagged_spans": [
    {"span": "problematic text verbatim", "reason": "specific reason why it is wrong or uncertain"}
  ],
  "overall_agreement": 0.80
}

Be precise and specific. Do NOT flag the entire output — only flag specific claims you can justify."""


class CritiqueAgent(BaseAgent):
    agent_id = "critique"
    max_budget = 3072

    def run(self) -> AgentOutput:
        self.context.add_sse_event("agent_start", agent_id=self.agent_id)
        start = time.monotonic()

        outputs_to_review = {
            aid: out for aid, out in self.context.agent_outputs.items()
            if aid not in ("critique", "orchestrator", "compression")
        }

        if not outputs_to_review:
            report = CritiqueReport(overall_agreement=1.0)
            self.context.critique_report = report
            return self._record_output(
                "No outputs to critique.", int((time.monotonic() - start) * 1000)
            )

        review_text = "\n\n".join([
            f"=== Agent: {aid} ===\n{out.content[:800]}"
            for aid, out in outputs_to_review.items()
        ])

        remaining = self.budget_manager.check_remaining(self.agent_id)
        if self._count_tokens(review_text) > remaining - 700:
            review_text = review_text[: remaining * 3]

        messages = [
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(content=f"Review these agent outputs:\n\n{review_text}"),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            claim_scores = [ClaimScore(**cs) for cs in parsed.get("claim_scores", [])]
            report = CritiqueReport(
                reviewed_agents=list(outputs_to_review.keys()),
                claim_scores=claim_scores,
                flagged_spans=parsed.get("flagged_spans", []),
                overall_agreement=float(parsed.get("overall_agreement", 0.75)),
            )
        except Exception:
            report = CritiqueReport(
                reviewed_agents=list(outputs_to_review.keys()),
                overall_agreement=0.75,
            )

        self.context.critique_report = report
        summary = (
            f"Reviewed {len(outputs_to_review)} agents. "
            f"Agreement: {report.overall_agreement:.2f}. "
            f"Flagged: {len(report.flagged_spans)} spans. "
            f"Claims scored: {len(report.claim_scores)}."
        )
        latency = int((time.monotonic() - start) * 1000)
        return self._record_output(summary, latency)
