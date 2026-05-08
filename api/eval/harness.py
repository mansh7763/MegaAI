import json
import uuid
from datetime import datetime
from api.eval.test_cases import TEST_CASES, TestCase
from api.eval.scorer import score_case, CaseScore
from api.graph.pipeline import run_pipeline
from api.core.shared_context import SharedContext


async def run_eval(case_ids: list[str] = None, run_id: str = None) -> dict:
    """
    Run the evaluation harness.
    If case_ids provided, runs only those cases. Otherwise runs all 15.
    Returns a fully serialisable dict with reproducible structure.
    """
    if not run_id:
        run_id = str(uuid.uuid4())

    cases_to_run = (
        [c for c in TEST_CASES if c.id in case_ids]
        if case_ids
        else TEST_CASES
    )

    results = []
    for test_case in cases_to_run:
        job_id = f"eval_{run_id}_{test_case.id}"
        try:
            context = await run_pipeline(test_case.query, job_id=job_id)
            case_score = score_case(context, test_case)
        except Exception as e:
            context = SharedContext(job_id=job_id, original_query=test_case.query)
            case_score = CaseScore(
                case_id=test_case.id,
                category=test_case.category,
                dimension_scores=[],
            )

        results.append({
            "case_id": test_case.id,
            "category": test_case.category,
            "adversarial_type": test_case.adversarial_type,
            "query": test_case.query,
            "ground_truth": test_case.ground_truth,
            "final_answer": context.final_answer.content if context.final_answer else "",
            "overall_score": case_score.overall,
            "dimension_scores": [
                {
                    "dimension": d.dimension,
                    "score": d.score,
                    "justification": d.justification,
                }
                for d in case_score.dimension_scores
            ],
            "agent_prompts": {
                aid: out.content[:400]
                for aid, out in context.agent_outputs.items()
            },
            "tool_calls": [tc.model_dump(default=str) for tc in context.tool_call_log],
            "routing_decisions": [rd.model_dump(default=str) for rd in context.routing_decisions],
            "policy_violations": len(context.policy_violations),
            "timestamp": datetime.utcnow().isoformat(),
        })

    summary = _compute_summary(results)

    return {
        "run_id": run_id,
        "triggered_at": datetime.utcnow().isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "total_cases": len(results),
        "results": results,
        "summary": summary,
    }


def _compute_summary(results: list[dict]) -> dict:
    if not results:
        return {}

    by_category: dict[str, list[float]] = {}
    by_dimension: dict[str, list[float]] = {}

    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, []).append(r["overall_score"])
        for ds in r.get("dimension_scores", []):
            by_dimension.setdefault(ds["dimension"], []).append(ds["score"])

    overall_scores = [r["overall_score"] for r in results]
    return {
        "overall_avg": round(sum(overall_scores) / max(len(overall_scores), 1), 3),
        "by_category": {
            cat: round(sum(scores) / max(len(scores), 1), 3)
            for cat, scores in by_category.items()
        },
        "by_dimension": {
            dim: round(sum(scores) / max(len(scores), 1), 3)
            for dim, scores in by_dimension.items()
        },
        "failing_cases": [r["case_id"] for r in results if r["overall_score"] < 0.5],
        "top_cases": sorted(
            [{"id": r["case_id"], "score": r["overall_score"]} for r in results],
            key=lambda x: x["score"],
            reverse=True,
        )[:3],
    }


def diff_runs(run_a: dict, run_b: dict) -> dict:
    """Compute score deltas between two eval runs for regression detection."""
    a_by_case = {r["case_id"]: r for r in run_a.get("results", [])}
    b_by_case = {r["case_id"]: r for r in run_b.get("results", [])}

    deltas = []
    for case_id, b_result in b_by_case.items():
        a_result = a_by_case.get(case_id)
        if not a_result:
            continue
        delta = b_result["overall_score"] - a_result["overall_score"]
        deltas.append({
            "case_id": case_id,
            "before": a_result["overall_score"],
            "after": b_result["overall_score"],
            "delta": round(delta, 3),
            "improved": delta > 0,
        })

    return {
        "run_a": run_a["run_id"],
        "run_b": run_b["run_id"],
        "case_deltas": sorted(deltas, key=lambda x: x["delta"]),
        "overall_delta": round(
            run_b["summary"].get("overall_avg", 0) - run_a["summary"].get("overall_avg", 0),
            3,
        ),
    }
