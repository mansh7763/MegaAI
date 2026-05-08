# Mega AI — Real-Time Multi-Agent LLM Orchestration

A containerized, production-grade multi-agent LLM system with dynamic routing, self-improving evaluation loop, adversarial robustness testing, and real-time SSE streaming.

## Quick Start (5 minutes)

### Prerequisites
- Docker + Docker Compose
- A [Groq API key](https://console.groq.com) (free)

### Setup

```bash
git clone https://github.com/<you>/megaai
cd megaai
git config core.hooksPath githooks   # strips Co-authored-by Cursor from commits if your IDE adds it
cp .env.example .env
# Edit .env — add your GROQ_API_KEY
docker compose up --build
```

| Service | URL | Description |
|---------|-----|-------------|
| UI + API | http://localhost:8000 | Agent chat UI |
| API Docs | http://localhost:8000/docs | Interactive Swagger |
| DB Viewer | http://localhost:8080 | Adminer (server: db, user: megaai, pass: megaai) |

---

## Architecture

See `architecture.md` for the full diagram.

```
CLIENT (Browser)
    │ SSE + REST
FastAPI API Server (:8000)
    ├── POST /query          → real-time SSE stream
    ├── GET  /trace/{job_id} → full execution trace
    ├── GET  /eval/latest    → eval summary by category+dimension
    ├── POST /eval/approve   → approve/reject MetaAgent rewrite
    └── POST /eval/rerun     → targeted re-eval on failed cases
         │
    LangGraph Orchestrator (in-process)
         ├── SharedContext (Pydantic) — single source of truth
         ├── OrchestratorAgent — LLM-based dynamic routing
         ├── DecompositionAgent — typed sub-tasks + dependency graph
         ├── RAGAgent — multi-hop retrieval + per-sentence citations
         ├── CritiqueAgent — per-claim confidence scoring
         ├── SynthesisAgent — contradiction resolution + provenance map
         ├── CompressionAgent — triggered on budget overflow
         └── MetaAgent — post-eval prompt rewrite proposals
         │
    PostgreSQL 16 + pgvector (:5432)
    Redis (:6379) — Celery broker
    Celery Worker — async eval jobs
    Adminer (:8080) — DB log query UI
```

---

## Agents

| Agent | Role | Decision Boundary |
|-------|------|-------------------|
| **OrchestratorAgent** | Decides next agent at runtime via LLM call with structured state | Never hardcoded; every routing decision is logged with justification |
| **DecompositionAgent** | Breaks queries into typed SubTasks with explicit dependency graph | Blocks dependent tasks until all their dependencies resolve |
| **RAGAgent** | Multi-hop: retrieves ≥2 chunks, reasons across them, cites every sentence | Validates ≥2 distinct chunk_ids in citations; single-hop rejected by schema |
| **CritiqueAgent** | Reviews every AgentOutput, scores individual claims 0–1, flags specific spans | Does NOT flag whole outputs — flags specific text spans only |
| **SynthesisAgent** | Merges outputs, resolves all critique flags, builds provenance map | Runs once, after critique; every sentence traced to source_agent + chunk_id |
| **CompressionAgent** | Triggered by ContextBudgetManager on budget overflow | Lossless for structured data (JSON, scores, citations); lossy for filler |
| **MetaAgent** | Post-eval: reads failure cases, proposes prompt rewrite with unified diff | Proposes only — never auto-applies; human must approve via API |

---

## Knowledge Base (14 documents)

The RAG agent retrieves from a curated corpus of state-of-the-art AI documents:

| File | Topics |
|------|--------|
| `01_transformer_architecture.md` | Encoder-decoder, encoder-only, decoder-only, scaling laws |
| `02_attention_mechanism.md` | Scaled dot-product, MHA, self/cross attention, masking |
| `03_attention_variants_mha_mqa_gqa_mla.md` | MQA, GQA, MLA (DeepSeek), comparison table |
| `04_kv_cache.md` | Prefill/decode phases, PagedAttention, quantization, prefix caching |
| `05_flash_attention.md` | IO-aware tiling, FA1/FA2/FA3, Flash Decoding |
| `06_positional_encodings_rope_alibi.md` | RoPE, ALiBi, YaRN, NTK scaling |
| `07_reasoning_models.md` | CoT, OpenAI o1/o3, DeepSeek-R1, ToT, PRMs |
| `08_conversational_memory.md` | Buffer, summarisation, entity, vector store memory |
| `09_agentic_memory.md` | Working/episodic/semantic/procedural memory, MemGPT |
| `10_rag_and_multi_hop.md` | Single-hop vs multi-hop, HyDE, RAPTOR, Self-RAG |
| `11_llm_evaluation.md` | Metrics, LLM-as-judge, hallucination detection, RAGAs |
| `12_prompt_engineering.md` | CoT, ReAct, prompt injection defences, structured output |
| `13_multi_agent_systems.md` | Orchestrator-worker, shared context, failure modes |
| `14_speculative_decoding_and_moe.md` | Draft/verify, Mixtral, DeepSeek-V3, expert routing |

---

## API Reference

### POST /query
Submit a query, receive real-time SSE stream.
```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Flash Attention?"}'
```

### GET /trace/{job_id}
Full execution trace for a completed job.
```bash
curl http://localhost:8000/trace/<job_id>
```

### GET /eval/latest
Latest eval run summary by category and dimension.
```bash
curl http://localhost:8000/eval/latest
```

### POST /eval/approve
Approve or reject a pending MetaAgent prompt rewrite.
```bash
curl -X POST http://localhost:8000/eval/approve \
  -H "Content-Type: application/json" \
  -d '{"rewrite_id": "<uuid>", "decision": "approve"}'
```

### POST /eval/rerun
Trigger targeted re-eval.
```bash
# Full suite
curl -X POST http://localhost:8000/eval/rerun -H "Content-Type: application/json" -d '{}'
# Targeted
curl -X POST http://localhost:8000/eval/rerun \
  -H "Content-Type: application/json" \
  -d '{"case_ids": ["adv1", "adv2", "b3"]}'
```

---

## Self-Improving Loop

1. `POST /eval/rerun` → runs 15 test cases through the full pipeline
2. MetaAgent reads failure cases → identifies worst-performing prompt by dimension
3. Proposes rewrite: `{prompt_id, old_text, new_text, unified_diff, justification}` → stored as `pending`
4. `POST /eval/approve {rewrite_id, decision: "approve"}` → human decides
5. If approved → targeted re-eval on failed cases → delta stored for comparison
6. Full audit trail: every rewrite, decision, timestamp queryable at `/eval/latest`

---

## Evaluation — 15 Test Cases

| Category | Count | What's tested |
|----------|-------|---------------|
| Baseline | 5 | Factual questions with known answers (Transformer, Flash Attention, GQA, KV cache, reasoning models) |
| Ambiguous | 5 | Underspecified inputs to test decomposition quality |
| Adversarial | 5 | Prompt injections, false premises, contradiction traps |

**6 Scoring Dimensions** (all custom-built, no third-party eval framework):
- `answer_correctness` — keyword match vs ground truth + injection detection
- `citation_accuracy` — validates ≥2 distinct chunk_ids cited
- `contradiction_resolution` — fraction of critique flags resolved
- `tool_efficiency` — penalises excess tool calls
- `budget_compliance` — 1 − (violations / agent_turns)
- `critique_agreement` — critique agent agreement rate

---

## Known Limitations

- **WebSearchTool is a stub** — returns deterministic results; real deployment needs Tavily/Serper
- **Code sandbox** uses basic Python exec with restricted builtins — not container-isolated
- **Groq free tier** rate limits (~30 RPM) may slow 15-case eval runs
- **MetaAgent requires human approval** — self-improvement is not autonomous by design
- **In-memory job store** — job traces are lost on restart; replace `_job_store` with PostgreSQL for production
- **pgvector embeddings** not used for live retrieval yet — RAG uses keyword search; wire up an embedding model to enable full vector search

## What We Would Build Next

1. Real embedding model (BGE-M3, `text-embedding-ada-002`) for pgvector semantic retrieval
2. Tavily/Serper real web search integration
3. Container-isolated code execution (Firecracker or Docker-in-Docker)
4. Persistent job store in PostgreSQL (replace in-memory dict)
5. LangSmith / Langfuse integration for production tracing
6. A/B testing framework for approved prompt rewrites

---

## AI Collaboration Attestation

This project was built with AI assistance (Cursor / Claude). The AI was used for:
- Scaffolding boilerplate (Docker, Pydantic schemas, FastAPI routes)
- Writing structured prompt templates for agents
- Generating the knowledge base document content

All architectural decisions (SharedContext schema, ContextBudgetManager, multi-hop citation enforcement, eval scoring logic) were designed by the engineer and implemented with AI assistance.
