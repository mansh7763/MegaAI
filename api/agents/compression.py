import time
from langchain_core.messages import HumanMessage, SystemMessage
from api.agents.base_agent import BaseAgent
from api.core.shared_context import AgentOutput

COMPRESSION_PROMPT = """You are a compression agent. Compress the provided context to reduce token count.

CRITICAL rules:
- PRESERVE VERBATIM: all JSON structures, numeric scores, citations, chunk_ids, provenance maps, tool outputs
- COMPRESS (summarise): narrative explanations, repetitive reasoning text, conversational filler
- Never remove structured data (lists, dicts, scores, citations)

Output the compressed context directly. No preamble."""


class CompressionAgent(BaseAgent):
    agent_id = "compression"
    max_budget = 2048

    def run(self, content_to_compress: str = "") -> AgentOutput:
        self.context.add_sse_event("agent_start", agent_id=self.agent_id)
        start = time.monotonic()

        if not content_to_compress:
            content_to_compress = "\n\n".join([
                f"[{aid}]: {out.content[:600]}"
                for aid, out in self.context.agent_outputs.items()
            ])

        messages = [
            SystemMessage(content=COMPRESSION_PROMPT),
            HumanMessage(content=f"Compress this context:\n\n{content_to_compress}"),
        ]
        try:
            response = self.llm.invoke(messages)
            compressed = response.content
        except Exception:
            compressed = content_to_compress[:len(content_to_compress) // 2]

        latency = int((time.monotonic() - start) * 1000)
        return self._record_output(compressed, latency)
