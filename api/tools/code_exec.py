import io
import time
import contextlib
from api.tools.base import BaseTool, ToolResult

try:
    from RestrictedPython import compile_restricted, safe_globals, safe_builtins
    RESTRICTED_AVAILABLE = True
except ImportError:
    RESTRICTED_AVAILABLE = False

SAFE_BUILTINS = {
    "print": print, "range": range, "len": len, "str": str, "int": int,
    "float": float, "list": list, "dict": dict, "tuple": tuple, "set": set,
    "bool": bool, "abs": abs, "max": max, "min": min, "sum": sum,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "round": round,
}


class CodeExecTool(BaseTool):
    name = "code_exec"
    timeout_seconds = 5

    def run(self, code: str) -> ToolResult:
        if not code or not isinstance(code, str):
            return ToolResult(success=False, error_code="invalid_input",
                              error_message="Code must be a non-empty string", latency_ms=0)

        start = time.monotonic()
        try:
            result = self._execute(code)
            latency = int((time.monotonic() - start) * 1000)
            result["execution_time_ms"] = latency
            return ToolResult(success=result["exit_code"] == 0, data=result, latency_ms=latency)
        except SyntaxError as e:
            latency = int((time.monotonic() - start) * 1000)
            return ToolResult(success=False, error_code="invalid_input", data={
                "stdout": "", "stderr": f"SyntaxError: {e}", "exit_code": 2,
                "execution_time_ms": latency,
            }, latency_ms=latency)
        except Exception as e:
            latency = int((time.monotonic() - start) * 1000)
            return ToolResult(success=False, error_code="runtime_error", data={
                "stdout": "", "stderr": str(e), "exit_code": 1,
                "execution_time_ms": latency,
            }, latency_ms=latency)

    def _execute(self, code: str) -> dict:
        buf = io.StringIO()
        glb = {"__builtins__": SAFE_BUILTINS}
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<sandbox>", "exec"), glb)
        return {"stdout": buf.getvalue(), "stderr": "", "exit_code": 0, "execution_time_ms": 0}
