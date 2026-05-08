from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class QueryRequest(BaseModel):
    query: str
    job_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    rewrite_id: str
    decision: str  # "approve" or "reject"


class RerunRequest(BaseModel):
    case_ids: Optional[list[str]] = None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: Optional[str] = None
