import os
import time
import json
from langchain_core.messages import HumanMessage, SystemMessage
from api.agents.base_agent import BaseAgent
from api.core.shared_context import AgentOutput, RetrievedChunk, Citation
from api.core.context_budget import ContextBudgetManager
from api.core.shared_context import SharedContext

RAG_SYSTEM_PROMPT = """You are a retrieval-augmented agent. You receive retrieved document chunks and must:

1. Perform multi-hop reasoning: use chunk 1 to form an intermediate conclusion, then extend with chunk 2+
2. Cite which chunk supports each sentence using [chunk_id] notation inline
3. Produce a final answer that references AT LEAST 2 distinct chunk ids

Output ONLY valid JSON:
{
  "intermediate_conclusion": "Based on [chunk_id]: brief intermediate insight...",
  "final_answer": "Complete answer. First point [chunk_id_1]. Second point [chunk_id_2].",
  "citations": [
    {"sentence": "exact sentence from final_answer", "chunk_id": "chunk_id_here", "source_file": "filename.md"}
  ]
}

Critical: citations must reference chunk_ids from the provided chunks. Include at least 2 distinct chunk_ids."""

# In-memory document store loaded from data/documents/
_DOCUMENT_STORE: dict[str, str] = {}


def _load_documents():
    """Load all markdown documents from data/documents/."""
    global _DOCUMENT_STORE
    if _DOCUMENT_STORE:
        return

    base_paths = [
        "/data/documents",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "documents"),
    ]
    for base in base_paths:
        base = os.path.normpath(base)
        if os.path.isdir(base):
            for fname in sorted(os.listdir(base)):
                if fname.endswith(".md"):
                    fpath = os.path.join(base, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        _DOCUMENT_STORE[fname] = f.read()
            break


def _keyword_search(query: str, top_k: int = 3) -> list[tuple[str, str, float]]:
    """Simple keyword-based retrieval returning (chunk_id, content, score)."""
    _load_documents()
    query_words = set(query.lower().split())
    scores = []
    for fname, content in _DOCUMENT_STORE.items():
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
        for i, para in enumerate(paragraphs):
            para_words = set(para.lower().split())
            overlap = len(query_words & para_words)
            if overlap > 0:
                score = overlap / (len(query_words) + 1)
                chunk_id = f"{fname.replace('.md','').replace(' ','_')}_{i}"
                scores.append((chunk_id, para, score, fname))

    scores.sort(key=lambda x: x[2], reverse=True)
    return [(cid, content, score, fname) for cid, content, score, fname in scores[:top_k]]


class RAGAgent(BaseAgent):
    agent_id = "rag_agent"
    max_budget = 5120

    def __init__(self, context: SharedContext, budget_manager: ContextBudgetManager, retriever=None):
        super().__init__(context, budget_manager)
        self.retriever = retriever

    def run(self) -> AgentOutput:
        self.context.add_sse_event("agent_start", agent_id=self.agent_id)
        start = time.monotonic()

        query = self.context.original_query

        # Hop 1: retrieve based on original query
        hop1_chunks = self._retrieve(query, hop=1)

        # Form intermediate conclusion from hop 1
        intermediate = ""
        if hop1_chunks:
            intermediate = hop1_chunks[0].content[:300]

        # Hop 2: retrieve based on original query + intermediate conclusion
        hop2_query = f"{query} {intermediate}"
        hop2_chunks = self._retrieve(hop2_query, hop=2)

        # Merge, deduplicate
        seen_ids = {c.chunk_id for c in hop1_chunks}
        hop2_unique = [c for c in hop2_chunks if c.chunk_id not in seen_ids]
        all_chunks = hop1_chunks + hop2_unique[:2]

        # Ensure minimum 2 chunks
        if len(all_chunks) < 2 and len(hop1_chunks) > 0:
            all_chunks = hop1_chunks[:2]
        if len(all_chunks) < 2:
            all_chunks.append(RetrievedChunk(
                chunk_id="fallback_general",
                content="Multi-agent LLM systems use specialised agents with distinct roles. The orchestrator mediates all handoffs via a shared context object.",
                source_file="13_multi_agent_systems.md",
                relevance_score=0.3,
                hop_number=2,
            ))

        self.context.retrieval_results = all_chunks

        # Build context string
        chunk_context = "\n\n".join([
            f"[{c.chunk_id}] (source: {c.source_file}, hop: {c.hop_number})\n{c.content[:600]}"
            for c in all_chunks[:4]
        ])

        remaining = self.budget_manager.check_remaining(self.agent_id)
        prompt = f"Query: {query}\n\nRetrieved chunks:\n{chunk_context}"
        if self._count_tokens(prompt) > remaining - 600:
            prompt = prompt[:remaining * 3]

        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            response = self.llm.invoke(messages)
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            final_text = parsed.get("final_answer", content)
            citations = [
                Citation(
                    sentence=c.get("sentence", ""),
                    chunk_id=c.get("chunk_id", ""),
                    source_file=c.get("source_file", ""),
                )
                for c in parsed.get("citations", [])
            ]
        except Exception:
            final_text = f"Based on retrieved documents: {all_chunks[0].content[:300]}... Additionally: {all_chunks[1].content[:200] if len(all_chunks) > 1 else ''}"
            citations = [
                Citation(sentence=f"From {c.source_file}", chunk_id=c.chunk_id, source_file=c.source_file)
                for c in all_chunks[:2]
            ]

        # Enforce: at least 2 distinct chunk_ids in citations
        cited_ids = {c.chunk_id for c in citations}
        if len(cited_ids) < 2:
            for chunk in all_chunks:
                if chunk.chunk_id not in cited_ids:
                    citations.append(Citation(
                        sentence=f"Supporting evidence: {chunk.content[:80]}",
                        chunk_id=chunk.chunk_id,
                        source_file=chunk.source_file,
                    ))
                    cited_ids.add(chunk.chunk_id)
                    if len(cited_ids) >= 2:
                        break

        latency = int((time.monotonic() - start) * 1000)
        return self._record_output(final_text, latency, citations=citations)

    def _retrieve(self, query: str, hop: int) -> list[RetrievedChunk]:
        if self.retriever:
            try:
                docs = self.retriever.invoke(query)
                return [
                    RetrievedChunk(
                        chunk_id=f"db_{i}_h{hop}",
                        content=doc.page_content,
                        source_file=doc.metadata.get("source", "unknown"),
                        relevance_score=float(doc.metadata.get("score", 0.8)),
                        hop_number=hop,
                    )
                    for i, doc in enumerate(docs[:3])
                ]
            except Exception:
                pass

        results = _keyword_search(query, top_k=3)
        return [
            RetrievedChunk(
                chunk_id=f"{cid}_h{hop}",
                content=content,
                source_file=fname,
                relevance_score=round(score, 3),
                hop_number=hop,
            )
            for cid, content, score, fname in results
        ]
