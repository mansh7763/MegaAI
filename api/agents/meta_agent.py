import time
import json
import difflib
from langchain_core.messages import HumanMessage, SystemMessage
from api.agents.base_agent import BaseAgent
from api.core.shared_context import AgentOutput

META_AGENT_PROMPT = """You are a meta-agent that improves LLM prompts based on evaluation failures.

Analyse the failing test cases and identify the prompt that, if improved, would have the most impact on the worst-scoring dimension.

Output ONLY valid JSON:
{
  "prompt_id": "agent_name",
  "weakness": "brief description of the current prompt's weakness",
  "new_prompt": "the complete rewritten prompt text (full replacement, not a diff)",
  "justification": "why this rewrite should improve the target dimension score",
  "target_dimension": "which scoring dimension (answer_correctness|citation_accuracy|contradiction_resolution|tool_efficiency|budget_compliance|critique_agreement)",
  "affected_case_ids": ["case_id_1", "case_id_2"]
}"""

# Registry of improvable prompts
CURRENT_PROMPTS = {
    "decomposition": "You are a decomposition agent. Break the user query into typed sub-tasks with explicit dependencies...",
    "rag_agent": "You are a retrieval-augmented agent. You receive retrieved document chunks and must perform multi-hop reasoning...",
    "critique": "You are a critique agent. Your job is to review each agent's output and score individual claims...",
    "synthesis": "You are a synthesis agent. Merge all agent outputs into one coherent final answer...",
    "orchestrator": "You are an orchestrator agent. Decide which agent to invoke next based on the current state...",
}


class MetaAgent(BaseAgent):
    agent_id = "meta_agent"
    max_budget = 3072

    def run(self, failure_cases: list[dict] = None) -> AgentOutput:
        self.context.add_sse_event("agent_start", agent_id=self.agent_id)
        start = time.monotonic()

        if not failure_cases:
            return self._record_output(
                json.dumps({"error": "no failure cases provided"}),
                int((time.monotonic() - start) * 1000),
            )

        failure_summary = json.dumps(failure_cases[:5], indent=2, default=str)
        prompts_summary = json.dumps(
            {k: v[:300] for k, v in CURRENT_PROMPTS.items()}, indent=2
        )

        prompt = (
            f"Failing cases:\n{failure_summary}\n\n"
            f"Current prompts (first 300 chars each):\n{prompts_summary}"
        )

        remaining = self.budget_manager.check_remaining(self.agent_id)
        if self._count_tokens(prompt) > remaining - 500:
            prompt = prompt[: remaining * 3]

        messages = [
            SystemMessage(content=META_AGENT_PROMPT),
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

            # Build unified diff
            old_prompt = CURRENT_PROMPTS.get(parsed.get("prompt_id", ""), "")
            new_prompt = parsed.get("new_prompt", "")
            diff_lines = list(difflib.unified_diff(
                old_prompt.splitlines(),
                new_prompt.splitlines(),
                fromfile=f"{parsed.get('prompt_id', 'unknown')}_old",
                tofile=f"{parsed.get('prompt_id', 'unknown')}_new",
                lineterm="",
            ))
            parsed["unified_diff"] = "\n".join(diff_lines)
            parsed["old_text"] = old_prompt

            latency = int((time.monotonic() - start) * 1000)
            return self._record_output(json.dumps(parsed, indent=2), latency)
        except Exception as e:
            latency = int((time.monotonic() - start) * 1000)
            return self._record_output(
                json.dumps({"error": str(e), "raw_response": ""}), latency
            )
