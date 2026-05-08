from api.tools.base import BaseTool, ToolResult

STUB_RESULTS = {
    "transformer": [
        {"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need",
         "snippet": "The Transformer relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.", "relevance_score": 0.97},
        {"url": "https://jalammar.github.io/illustrated-transformer/", "title": "The Illustrated Transformer",
         "snippet": "Visual guide to transformer architecture, multi-head attention, and positional encodings.", "relevance_score": 0.91},
    ],
    "attention": [
        {"url": "https://arxiv.org/abs/2205.14135", "title": "Flash Attention",
         "snippet": "FlashAttention is an IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes.", "relevance_score": 0.95},
        {"url": "https://arxiv.org/abs/2307.08691", "title": "GQA: Training Generalized Multi-Query Transformer",
         "snippet": "Grouped-query attention interpolates between multi-head and multi-query attention.", "relevance_score": 0.93},
    ],
    "rag": [
        {"url": "https://arxiv.org/abs/2005.11401", "title": "RAG: Retrieval-Augmented Generation",
         "snippet": "RAG combines parametric and non-parametric memory for knowledge-intensive NLP.", "relevance_score": 0.96},
        {"url": "https://arxiv.org/abs/2312.10997", "title": "RAPTOR: Recursive Abstractive Processing",
         "snippet": "RAPTOR builds a tree of summaries for multi-level retrieval.", "relevance_score": 0.88},
    ],
    "reasoning": [
        {"url": "https://openai.com/research/learning-to-reason-with-llms", "title": "OpenAI o1 Technical Report",
         "snippet": "o1 uses chain-of-thought reasoning trained via RL to solve complex problems step by step.", "relevance_score": 0.95},
        {"url": "https://arxiv.org/abs/2201.11903", "title": "Chain-of-Thought Prompting",
         "snippet": "Chain-of-thought prompting elicits reasoning in LLMs by providing exemplars with reasoning steps.", "relevance_score": 0.92},
    ],
    "default": [
        {"url": "https://arxiv.org/abs/2308.10792", "title": "A Survey on LLM Agents",
         "snippet": "LLM-based agents combine reasoning, tool use, and memory to complete complex tasks.", "relevance_score": 0.87},
        {"url": "https://lilianweng.github.io/posts/2023-06-23-agent/", "title": "LLM Powered Autonomous Agents",
         "snippet": "Overview of planning, memory, and tool-use in LLM agent systems.", "relevance_score": 0.84},
    ],
}


class WebSearchTool(BaseTool):
    name = "web_search"
    timeout_ms = 3000

    def run(self, query: str) -> ToolResult:
        if not query or not isinstance(query, str):
            return ToolResult(success=False, error_code="invalid_input",
                              error_message="Query must be a non-empty string", latency_ms=0)
        if len(query.strip()) < 3:
            return ToolResult(success=False, error_code="invalid_input",
                              error_message="Query too short", latency_ms=5)

        q = query.lower()
        if "transformer" in q or "architecture" in q:
            key = "transformer"
        elif "attention" in q or "flash" in q or "gqa" in q or "kv" in q:
            key = "attention"
        elif "rag" in q or "retrieval" in q:
            key = "rag"
        elif "reason" in q or "chain" in q or "cot" in q:
            key = "reasoning"
        else:
            key = "default"

        return ToolResult(success=True, data=STUB_RESULTS[key], latency_ms=120)

    def empty_result(self) -> ToolResult:
        return ToolResult(success=False, error_code="no_results", data=[], latency_ms=50)

    def timeout_result(self) -> ToolResult:
        return ToolResult(success=False, error_code="timeout", data=[], latency_ms=self.timeout_ms)
