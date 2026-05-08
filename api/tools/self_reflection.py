from api.tools.base import BaseTool, ToolResult
from api.core.shared_context import SharedContext


class SelfReflectionTool(BaseTool):
    name = "self_reflection"

    def __init__(self, context: SharedContext):
        self.context = context

    def run(self, agent_id: str) -> ToolResult:
        prior_outputs = {
            aid: out for aid, out in self.context.agent_outputs.items()
            if aid == agent_id
        }
        if not prior_outputs:
            return ToolResult(
                success=False,
                error_code="no_prior_outputs",
                error_message=f"No prior outputs found for agent {agent_id}",
                latency_ms=1,
            )

        contradictions = self._find_contradictions(list(prior_outputs.values()))
        return ToolResult(
            success=True,
            data={
                "prior_outputs": [o.model_dump() for o in prior_outputs.values()],
                "contradictions": contradictions,
            },
            latency_ms=5,
        )

    def _find_contradictions(self, outputs) -> list[dict]:
        contradictions = []
        contents = [o.content for o in outputs]
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                if self._has_contradiction(contents[i], contents[j]):
                    contradictions.append({
                        "span_a": contents[i][:120],
                        "span_b": contents[j][:120],
                        "description": "Potential conflict detected between outputs",
                    })
        return contradictions

    def _has_contradiction(self, text_a: str, text_b: str) -> bool:
        neg_indicators = ["not ", "never ", "false", "incorrect", "wrong", "unlike"]
        for indicator in neg_indicators:
            if indicator in text_a.lower() and indicator not in text_b.lower():
                key_words = [w for w in text_a.split() if len(w) > 5]
                return any(word.lower() in text_b.lower() for word in key_words[:3])
        return False
