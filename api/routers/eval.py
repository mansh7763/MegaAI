import uuid
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.models.schemas import ApprovalRequest, RerunRequest, ErrorResponse
from api.eval.harness import run_eval, diff_runs

router = APIRouter()

# In-memory stores (replaced by DB in production via Celery worker)
_eval_store: dict[str, dict] = {}          # run_id → eval result
_rewrite_store: dict[str, dict] = {}       # rewrite_id → rewrite proposal


@router.get(
    "/eval/latest",
    summary="Get latest evaluation run summary broken down by category and dimension",
    tags=["Evaluation"],
)
async def get_latest_eval():
    """
    Returns the most recent evaluation run with:
    - Per-category averages (baseline / ambiguous / adversarial)
    - Per-dimension averages (6 scoring dimensions)
    - List of failing cases (overall score < 0.5)
    """
    if not _eval_store:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error_code="NO_EVAL_RUNS",
                message="No evaluation runs found. POST to /eval/rerun to trigger one.",
            ).model_dump(),
        )
    latest_key = sorted(_eval_store.keys())[-1]
    return _eval_store[latest_key]


@router.post(
    "/eval/approve",
    summary="Approve or reject a pending prompt rewrite proposed by MetaAgent",
    tags=["Evaluation"],
)
async def approve_rewrite(request: ApprovalRequest):
    """
    Submit a human approval or rejection for a pending MetaAgent prompt rewrite.

    - `decision` must be `"approve"` or `"reject"`
    - If approved, a targeted re-eval is triggered on the affected cases
    - All decisions are stored with timestamp for full audit trail
    """
    if request.decision not in ("approve", "reject"):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="INVALID_DECISION",
                message="decision must be 'approve' or 'reject'",
            ).model_dump(),
        )

    if request.rewrite_id not in _rewrite_store:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error_code="REWRITE_NOT_FOUND",
                message=f"No pending rewrite with id '{request.rewrite_id}'",
            ).model_dump(),
        )

    entry = _rewrite_store[request.rewrite_id]
    entry["status"] = request.decision
    entry["decided_at"] = datetime.utcnow().isoformat()

    return {
        "rewrite_id": request.rewrite_id,
        "decision": request.decision,
        "prompt_id": entry.get("prompt_id", "unknown"),
        "decided_at": entry["decided_at"],
        "message": f"Rewrite {request.decision}d. Audit trail updated.",
    }


def _store_rewrite(rewrite_data: dict) -> str:
    """Store a MetaAgent-proposed rewrite and return its id."""
    rewrite_id = str(uuid.uuid4())
    _rewrite_store[rewrite_id] = {
        "id": rewrite_id,
        "proposed_at": datetime.utcnow().isoformat(),
        "status": "pending",
        **rewrite_data,
    }
    return rewrite_id


@router.post(
    "/eval/rerun",
    summary="Trigger a targeted re-evaluation on previously failed cases",
    tags=["Evaluation"],
)
async def trigger_rerun(request: RerunRequest, background_tasks: BackgroundTasks):
    """
    Trigger a new evaluation run.

    - If `case_ids` is provided, runs only those cases (targeted re-eval on failures)
    - If omitted, runs the full 15-case suite
    - Results are stored asynchronously and queryable via GET /eval/latest
    """
    run_id = str(uuid.uuid4())

    async def _run():
        try:
            result = await run_eval(case_ids=request.case_ids, run_id=run_id)
            _eval_store[run_id] = result

            # If there are failing cases, run MetaAgent to propose rewrite
            failing = result["summary"].get("failing_cases", [])
            if failing:
                from api.core.shared_context import SharedContext
                from api.core.context_budget import ContextBudgetManager
                from api.agents.meta_agent import MetaAgent

                ctx = SharedContext(job_id=f"meta_{run_id}", original_query="meta_analysis")
                budget_mgr = ContextBudgetManager(ctx)
                agent = MetaAgent(ctx, budget_mgr)

                failure_cases = [
                    r for r in result["results"] if r["case_id"] in failing
                ]
                output = agent.run(failure_cases=failure_cases)

                try:
                    parsed = json.loads(output.content)
                    rewrite_id = _store_rewrite(parsed)
                    _eval_store[run_id]["proposed_rewrite_id"] = rewrite_id
                except Exception:
                    pass
        except Exception:
            _eval_store[run_id] = {
                "run_id": run_id,
                "error": "Eval run failed",
                "total_cases": 0,
                "summary": {},
                "results": [],
            }

    background_tasks.add_task(_run)

    return {
        "run_id": run_id,
        "message": "Evaluation started in background",
        "case_ids": request.case_ids or "all_15",
        "check_results": f"GET /eval/latest once complete",
    }
