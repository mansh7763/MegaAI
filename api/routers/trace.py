from fastapi import APIRouter, HTTPException
from api.models.schemas import ErrorResponse

router = APIRouter()

# In-memory job store (keyed by job_id → serialised SharedContext dict)
_job_store: dict[str, dict] = {}


def store_job(job_id: str, context_dict: dict):
    _job_store[job_id] = context_dict


@router.get(
    "/trace/{job_id}",
    summary="Get full execution trace for a completed job",
    tags=["Trace"],
)
async def get_trace(job_id: str):
    """
    Returns the full execution trace for a job, including:
    - All routing decisions in chronological order
    - Every agent output (with output hash and token count)
    - All tool calls (inputs, outputs, latency, retry count)
    - All policy violations
    - SSE event sequence
    """
    if job_id not in _job_store:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error_code="JOB_NOT_FOUND",
                message=f"No job found with id '{job_id}'",
                job_id=job_id,
            ).model_dump(),
        )

    ctx = _job_store[job_id]
    return {
        "job_id": job_id,
        "query": ctx.get("original_query", ""),
        "status": "completed" if ctx.get("metadata", {}).get("completed") else "running",
        "steps": ctx.get("metadata", {}).get("steps", 0),
        "final_answer": (
            ctx["final_answer"]["content"]
            if ctx.get("final_answer") and ctx["final_answer"]
            else ""
        ),
        "routing_decisions": ctx.get("routing_decisions", []),
        "agent_outputs": {
            aid: {
                "content_preview": out.get("content", "")[:600],
                "output_hash": out.get("output_hash", ""),
                "token_count": out.get("token_count", 0),
                "latency_ms": out.get("latency_ms", 0),
                "citations": out.get("citations", []),
            }
            for aid, out in ctx.get("agent_outputs", {}).items()
        },
        "tool_calls": ctx.get("tool_call_log", []),
        "policy_violations": ctx.get("policy_violations", []),
        "sse_events": ctx.get("sse_events", []),
        "sub_tasks": ctx.get("sub_tasks", []),
        "critique_report": ctx.get("critique_report"),
        "context_budget": ctx.get("context_budget", {}),
    }
