import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import query, trace, eval as eval_router, health

app = FastAPI(
    title="Mega AI — Multi-Agent Orchestration API",
    description="""
## Real-Time Multi-Agent LLM Orchestration and Evaluation System

A production-grade multi-agent system with:
- **Dynamic runtime routing** — OrchestratorAgent decides next agent via LLM at every step
- **SharedContext** — single typed Pydantic object; agents never call each other directly
- **Multi-hop RAG** — retrieves ≥2 chunks per query with per-sentence citations
- **Self-improving eval loop** — MetaAgent proposes prompt rewrites; human approves
- **Real-time SSE streaming** — live agent activity, tool calls, and budget updates

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Submit query → SSE stream of live agent activity |
| GET | `/trace/{job_id}` | Full execution trace for a completed job |
| GET | `/eval/latest` | Latest eval run summary by category and dimension |
| POST | `/eval/approve` | Approve or reject a MetaAgent prompt rewrite |
| POST | `/eval/rerun` | Trigger targeted re-eval on previously failed cases |
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(trace.router)
app.include_router(eval_router.router)
app.include_router(health.router)

# Serve frontend if available
if os.path.exists("/frontend"):
    app.mount("/static", StaticFiles(directory="/frontend"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse("/frontend/index.html")
elif os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui_dev():
        return FileResponse("../frontend/index.html")
