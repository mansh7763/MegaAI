import hashlib
import json
import time
from typing import Any, Optional
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: int = 0

    def to_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.data, default=str).encode()).hexdigest()[:16]


class BaseTool:
    name: str = "base_tool"
    max_retries: int = 2

    def run(self, input: Any) -> ToolResult:
        raise NotImplementedError

    def retry(self, modified_input: Any) -> ToolResult:
        return self.run(modified_input)
