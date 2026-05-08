import json
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.models.schemas import QueryRequest, ErrorResponse
from api.graph.pipeline import run_pipeline_streaming

router = APIRouter()


@router.post(
    "/query",
    summary="Submit a query and receive a real-time SSE stream",
    tags=["Query"],
)
async def submit_query(request: QueryRequest):
    """
    Submit a natural language query. Returns a **Server-Sent Events** stream with real-time agent activity.

    SSE event types:
    - `job_start` — pipeline begun
    - `agent_start` — an agent has started (agent_id, justification)
    - `agent_done` — an agent finished (output_hash, token_count, latency_ms)
    - `budget_update` — current token usage per agent
    - `routing_decision` — orchestrator routing decision with justification
    - `tool_call` / `tool_result` — tool activity
    - `agent_error` — agent raised an exception
    - `job_complete` — pipeline finished (final_answer, policy_violations)
    """
    job_id = request.job_id or str(uuid.uuid4())

    async def event_generator():
        try:
            async for event in run_pipeline_streaming(request.query, job_id=job_id):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:
            error = ErrorResponse(
                error_code="PIPELINE_ERROR",
                message=str(e),
                job_id=job_id,
            )
            yield f"data: {json.dumps(error.model_dump())}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
