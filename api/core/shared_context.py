import hashlib
import json
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str  # "factual", "reasoning", "retrieval", "code"
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"  # "pending", "running", "resolved", "failed"
    output: Optional[str] = None

    def is_blocked_by(self, task_id: str) -> bool:
        return task_id in self.dependencies

    def is_blocked(self, resolved_ids: set[str]) -> bool:
        return not all(dep in resolved_ids for dep in self.dependencies)


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source_file: str
    relevance_score: float
    hop_number: int = 1


class Citation(BaseModel):
    sentence: str
    chunk_id: str
    source_file: str


class AgentOutput(BaseModel):
    agent_id: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    token_count: int = 0
    latency_ms: int = 0
    output_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.output_hash:
            self.output_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


class ClaimScore(BaseModel):
    span: str
    confidence: float  # 0.0 - 1.0
    flagged: bool
    reason: Optional[str] = None


class CritiqueReport(BaseModel):
    reviewed_agents: list[str] = Field(default_factory=list)
    claim_scores: list[ClaimScore] = Field(default_factory=list)
    flagged_spans: list[dict] = Field(default_factory=list)
    overall_agreement: float = 0.0


class ProvenanceEntry(BaseModel):
    sentence: str
    source_agent: str
    source_chunk_id: Optional[str] = None


class FinalAnswer(BaseModel):
    content: str
    provenance_map: list[ProvenanceEntry] = Field(default_factory=list)
    contradictions_resolved: list[str] = Field(default_factory=list)


class PolicyViolation(BaseModel):
    agent_id: str
    violation_type: str  # "budget_overflow", "budget_policy_skip"
    details: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolCallRecord(BaseModel):
    tool_name: str
    input_hash: str
    output_hash: str
    latency_ms: int
    accepted: bool
    retry_count: int = 0
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SSEEvent(BaseModel):
    type: str
    payload: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RoutingDecision(BaseModel):
    from_agent: str
    to_agent: str
    justification: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SharedContext(BaseModel):
    job_id: str
    original_query: str
    sub_tasks: list[SubTask] = Field(default_factory=list)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    retrieval_results: list[RetrievedChunk] = Field(default_factory=list)
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)
    critique_report: Optional[CritiqueReport] = None
    final_answer: Optional[FinalAnswer] = None
    context_budget: dict[str, int] = Field(default_factory=dict)
    max_budget: dict[str, int] = Field(default_factory=dict)
    policy_violations: list[PolicyViolation] = Field(default_factory=list)
    tool_call_log: list[ToolCallRecord] = Field(default_factory=list)
    sse_events: list[SSEEvent] = Field(default_factory=list)
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_resolved_task_ids(self) -> set[str]:
        return {t.id for t in self.sub_tasks if t.status == "resolved"}

    def get_ready_tasks(self) -> list[SubTask]:
        resolved = self.get_resolved_task_ids()
        return [t for t in self.sub_tasks if t.status == "pending" and not t.is_blocked(resolved)]

    def add_sse_event(self, event_type: str, **kwargs):
        self.sse_events.append(SSEEvent(type=event_type, payload=kwargs))

    def to_dict(self) -> dict:
        return json.loads(self.model_dump_json())
