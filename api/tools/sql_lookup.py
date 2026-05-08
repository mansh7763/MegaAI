import time
from typing import Optional
from api.tools.base import BaseTool, ToolResult

SAMPLE_DATA = {
    "agents": [
        {"id": 1, "name": "OrchestratorAgent", "type": "routing", "avg_latency_ms": 210},
        {"id": 2, "name": "DecompositionAgent", "type": "planning", "avg_latency_ms": 380},
        {"id": 3, "name": "RAGAgent", "type": "retrieval", "avg_latency_ms": 540},
        {"id": 4, "name": "CritiqueAgent", "type": "evaluation", "avg_latency_ms": 320},
        {"id": 5, "name": "SynthesisAgent", "type": "synthesis", "avg_latency_ms": 290},
    ],
    "eval_results": [
        {"case_id": "b1", "category": "baseline", "score": 0.91, "dimension": "answer_correctness"},
        {"case_id": "b2", "category": "baseline", "score": 0.88, "dimension": "citation_accuracy"},
        {"case_id": "adv1", "category": "adversarial", "score": 0.72, "dimension": "contradiction_resolution"},
        {"case_id": "a1", "category": "ambiguous", "score": 0.65, "dimension": "answer_correctness"},
    ],
    "documents": [
        {"id": 1, "title": "Transformer Architecture", "source_file": "transformers.md", "chunk_count": 8},
        {"id": 2, "title": "Flash Attention", "source_file": "flash_attention.md", "chunk_count": 6},
        {"id": 3, "title": "GQA & MQA", "source_file": "attention_variants.md", "chunk_count": 7},
        {"id": 4, "title": "KV Cache", "source_file": "kv_cache.md", "chunk_count": 5},
        {"id": 5, "title": "Reasoning Models", "source_file": "reasoning_models.md", "chunk_count": 9},
    ],
}


class SQLLookupTool(BaseTool):
    name = "sql_lookup"

    def run(self, natural_language_query: str) -> ToolResult:
        if not natural_language_query or not isinstance(natural_language_query, str):
            return ToolResult(success=False, error_code="invalid_input",
                              error_message="Query must be a non-empty string", latency_ms=0)

        start = time.monotonic()
        try:
            sql = self._nl_to_sql(natural_language_query)
            rows = self._execute_stub(sql)
            latency = int((time.monotonic() - start) * 1000)

            if rows is None:
                return ToolResult(success=False, error_code="sql_parse_error",
                                  data={"query_attempted": sql}, latency_ms=latency)
            if len(rows) == 0:
                return ToolResult(success=False, error_code="no_rows",
                                  data={"rows": [], "sql": sql}, latency_ms=latency)

            return ToolResult(success=True,
                              data={"rows": rows, "sql": sql, "count": len(rows)},
                              latency_ms=latency)
        except Exception as e:
            latency = int((time.monotonic() - start) * 1000)
            return ToolResult(success=False, error_code="sql_parse_error",
                              error_message=str(e), latency_ms=latency)

    def _nl_to_sql(self, query: str) -> str:
        q = query.lower()
        if "agent" in q:
            return "SELECT * FROM agents"
        if "eval" in q or "score" in q or "result" in q:
            return "SELECT * FROM eval_results"
        if "document" in q or "chunk" in q or "knowledge" in q:
            return "SELECT * FROM documents"
        return "SELECT * FROM agents LIMIT 5"

    def _execute_stub(self, sql: str) -> Optional[list]:
        sql_lower = sql.lower()
        for table_name, data in SAMPLE_DATA.items():
            if table_name in sql_lower:
                return data
        return []
