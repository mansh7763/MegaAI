import uuid
from typing import AsyncIterator
from api.core.shared_context import SharedContext
from api.core.context_budget import ContextBudgetManager
from api.agents.orchestrator import OrchestratorAgent
from api.agents.decomposition import DecompositionAgent
from api.agents.rag_agent import RAGAgent
from api.agents.critique import CritiqueAgent
from api.agents.synthesis import SynthesisAgent
from api.agents.compression import CompressionAgent

MAX_STEPS = 12
VALID_AGENTS = {"decomposition", "rag_agent", "critique", "synthesis", "compression", "done"}


def _build_agents(context: SharedContext, budget_manager: ContextBudgetManager):
    budget_manager.declare_all_defaults()
    orchestrator = OrchestratorAgent(context, budget_manager)
    agent_map = {
        "decomposition": DecompositionAgent(context, budget_manager),
        "rag_agent": RAGAgent(context, budget_manager),
        "critique": CritiqueAgent(context, budget_manager),
        "synthesis": SynthesisAgent(context, budget_manager),
        "compression": CompressionAgent(context, budget_manager),
    }
    return orchestrator, agent_map


async def run_pipeline_streaming(query: str, job_id: str = None) -> AsyncIterator[dict]:
    """Run the full agent pipeline, yielding SSE event dicts."""
    if not job_id:
        job_id = str(uuid.uuid4())

    context = SharedContext(job_id=job_id, original_query=query)
    budget_manager = ContextBudgetManager(context)
    orchestrator, agent_map = _build_agents(context, budget_manager)

    yield {"type": "job_start", "job_id": job_id, "query": query}

    step = 0
    visited: set[str] = set()

    while step < MAX_STEPS:
        next_agent, justification = orchestrator.decide_next(step)

        if next_agent == "done":
            break

        if next_agent not in VALID_AGENTS or next_agent not in agent_map:
            yield {"type": "warning", "message": f"Unknown agent requested: {next_agent}"}
            break

        # Prevent infinite loops (allow compression to re-run)
        if next_agent in visited and next_agent != "compression":
            next_agent = orchestrator._fallback_routing(step)
            if next_agent == "done" or next_agent in visited:
                break
            justification = f"Loop prevention fallback → {next_agent}"

        visited.add(next_agent)

        yield {
            "type": "agent_start",
            "agent_id": next_agent,
            "step": step,
            "justification": justification,
        }

        try:
            output = agent_map[next_agent].run()
            yield {
                "type": "agent_done",
                "agent_id": next_agent,
                "output_hash": output.output_hash,
                "token_count": output.token_count,
                "latency_ms": output.latency_ms,
            }
            yield {
                "type": "budget_update",
                "agent_id": next_agent,
                "tokens_used": context.context_budget.get(next_agent, 0),
                "tokens_remaining": budget_manager.check_remaining(next_agent),
            }
        except Exception as e:
            yield {"type": "agent_error", "agent_id": next_agent, "error": str(e)}
            break

        step += 1

    final_answer = context.final_answer.content if context.final_answer else ""
    context.metadata["completed"] = True
    context.metadata["steps"] = step

    yield {
        "type": "job_complete",
        "job_id": job_id,
        "final_answer": final_answer,
        "policy_violations": len(context.policy_violations),
        "routing_decisions": len(context.routing_decisions),
        "steps": step,
    }

    # Store context in global job store for trace endpoint
    from api.routers.trace import store_job
    store_job(job_id, context.to_dict())


async def run_pipeline(query: str, job_id: str = None) -> SharedContext:
    """Run the full pipeline and return the final SharedContext."""
    if not job_id:
        job_id = str(uuid.uuid4())

    context = SharedContext(job_id=job_id, original_query=query)
    budget_manager = ContextBudgetManager(context)
    orchestrator, agent_map = _build_agents(context, budget_manager)

    step = 0
    visited: set[str] = set()

    while step < MAX_STEPS:
        next_agent, justification = orchestrator.decide_next(step)
        if next_agent == "done":
            break
        if next_agent not in VALID_AGENTS or next_agent not in agent_map:
            break
        if next_agent in visited and next_agent != "compression":
            next_agent = orchestrator._fallback_routing(step)
            if next_agent == "done" or next_agent in visited:
                break

        visited.add(next_agent)
        try:
            agent_map[next_agent].run()
        except Exception:
            pass
        step += 1

    context.metadata["completed"] = True
    context.metadata["steps"] = step
    return context
