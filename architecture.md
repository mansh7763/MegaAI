# Mega AI — System Architecture

## High-Level Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                    CLIENT  (Browser  :8000)                            │
│                                                                         │
│   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│   │  Query textarea  │  │  Agent pipeline │  │   Eval dashboard     │ │
│   │  Ctrl+Enter→SSE  │  │  live status    │  │  category/dimension  │ │
│   └──────────────────┘  └─────────────────┘  └──────────────────────┘ │
│   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│   │  Context budget  │  │   Event log     │  │   Final answer +     │ │
│   │  gauge per agent │  │   SSE stream    │  │   Routing trace      │ │
│   └──────────────────┘  └─────────────────┘  └──────────────────────┘ │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │  HTTP  /  SSE
┌──────────────────────────────▼─────────────────────────────────────────┐
│                   FastAPI  API  Server  (:8000)                         │
│                                                                         │
│  POST /query ──────────────── SSE stream of live agent activity         │
│  GET  /trace/{job_id} ──────── full execution trace in order            │
│  GET  /eval/latest ─────────── latest eval run by category+dimension    │
│  POST /eval/approve ────────── approve / reject MetaAgent rewrite       │
│  POST /eval/rerun ──────────── trigger targeted re-eval                 │
└──────┬───────────────────────────────────────────┬──────────────────────┘
       │  in-process                               │  background task
       │                                           │
┌──────▼──────────────────────────────┐  ┌────────▼────────────────────┐
│   LangGraph  Orchestration  Layer   │  │   Celery  Worker            │
│                                     │  │   (async eval jobs)         │
│  ┌──────────────────────────────┐   │  └─────────────────────────────┘
│  │   SharedContext  (Pydantic)  │   │
│  │                              │   │  ┌──────────────────────────────┐
│  │  job_id, original_query      │   │  │   Redis  (:6379)             │
│  │  sub_tasks + dep_graph       │   │  │   Celery broker/backend      │
│  │  retrieval_results           │   │  └──────────────────────────────┘
│  │  agent_outputs               │   │
│  │  critique_report             │   │
│  │  final_answer (provenance)   │   │
│  │  context_budget              │   │
│  │  policy_violations           │   │
│  │  tool_call_log               │   │
│  │  routing_decisions           │   │
│  │  sse_events                  │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   OrchestratorAgent          │   │
│  │   (LLM-based routing, every  │   │
│  │    step, logs justification) │   │
│  └──────────────┬───────────────┘   │
│                 │ decides next      │
│    ┌────────────┼──────────────┐    │
│    │            │              │    │
│  ┌─▼──────┐  ┌─▼──────┐  ┌───▼──┐ │
│  │ Decomp │  │  RAG   │  │Critiqu│ │
│  │ Agent  │  │ Agent  │  │Agent │ │
│  │dep.graph│ │multi-  │  │per-  │ │
│  └────────┘  │hop+cite│  │claim │ │
│              └────────┘  └──────┘ │
│  ┌─────────┐  ┌────────┐  ┌──────┐│
│  │Synthesis│  │Compress│  │ Meta ││
│  │provenance│ │lossy/  │  │Agent ││
│  │   map   │  │lossless│  │rewrit││
│  └─────────┘  └────────┘  └──────┘│
│                                     │
│  ┌──────────────────────────────┐   │
│  │   ContextBudgetManager       │   │
│  │   declare / check / consume  │   │
│  │   overflow → CompressionAgent│   │
│  │   skip → policy violation    │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   4  Tools                   │   │
│  │   WebSearchTool  (stub)      │   │
│  │   CodeExecTool  (sandbox)    │   │
│  │   SQLLookupTool  (NL→SQL)    │   │
│  │   SelfReflectionTool         │   │
│  │   each: failure contract     │   │
│  │   + 2 retries + logging      │   │
│  └──────────────────────────────┘   │
└──────────────────────┬──────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────┐
│              PostgreSQL  16  +  pgvector  (:5432)                        │
│                                                                           │
│  jobs               agent_logs          tool_logs                         │
│  eval_runs          eval_cases           eval_scores                      │
│  prompt_rewrites    document_chunks (vector(1536) + hnsw index)           │
└─────────────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────┐
│          Adminer  (:8080)               │
│   Browse jobs, eval_runs, agent_logs,   │
│   tool_logs, prompt_rewrites            │
└─────────────────────────────────────────┘
```

## Data Flow

```
User Query
    │
    ▼
POST /query → job_id assigned
    │
    ▼
run_pipeline_streaming()
    │
    ├─ OrchestratorAgent.decide_next() ──LLM call──► "decomposition"
    │      log RoutingDecision + SSE event
    │
    ├─ DecompositionAgent.run()
    │      → writes sub_tasks + dep_graph to SharedContext
    │      → SSE: agent_start, agent_done, budget_update
    │
    ├─ OrchestratorAgent.decide_next() ──LLM call──► "rag_agent"
    │
    ├─ RAGAgent.run()
    │      hop1: keyword_search(query) → RetrievedChunk[]
    │      hop2: keyword_search(query + hop1_excerpt) → more chunks
    │      LLM: reason across ≥2 chunks, produce JSON with citations
    │      → writes retrieval_results + agent_outputs["rag_agent"]
    │      → SSE: agent_start, agent_done, budget_update
    │
    ├─ OrchestratorAgent.decide_next() ──LLM call──► "critique"
    │
    ├─ CritiqueAgent.run()
    │      reviews all agent_outputs
    │      → writes critique_report (claim_scores, flagged_spans)
    │      → SSE: agent_start, agent_done
    │
    ├─ OrchestratorAgent.decide_next() ──LLM call──► "synthesis"
    │
    ├─ SynthesisAgent.run()
    │      merges outputs, resolves flagged contradictions
    │      → writes final_answer (content + provenance_map)
    │      → SSE: agent_start, agent_done
    │
    ├─ OrchestratorAgent.decide_next() ──LLM call──► "done"
    │
    └─ yield job_complete SSE event
           store SharedContext in _job_store for GET /trace
```

## Eval + Self-Improving Loop

```
POST /eval/rerun
    │
    ▼
Celery Worker: run_eval()
    │
    ├─ for each of 15 test cases:
    │      run_pipeline(case.query)
    │      score_case() → 6 DimensionScores
    │
    ├─ _compute_summary() → overall_avg, by_category, by_dimension, failing_cases
    │
    ├─ if failing_cases:
    │      MetaAgent.run(failure_cases)
    │      → proposes prompt rewrite with unified diff
    │      → stored in _rewrite_store with status="pending"
    │
GET /eval/latest → return eval results + proposed_rewrite_id
    │
POST /eval/approve {decision:"approve"}
    │      → entry["status"] = "approve", timestamp recorded
    │      → triggers targeted re-eval on affected cases
    │      → delta stored for regression detection
```
