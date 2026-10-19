"""Answer-level error types. An AgentError means no predictions are accepted for that answer."""
from __future__ import annotations

# error type -> HTTP status used by POST /classify
HTTP_STATUS = {
    "invalid_input": 422,
    "unknown_brand": 502,          # requested by the model mid-run (primary brand is checked first -> 404)
    "illegal_tool": 502,
    "invalid_tool_arguments": 502,
    "invalid_output": 502,
    "turn_limit": 502,
    "provider_error": 502,
    "provider_rate_limited": 502,
    "provider_timeout": 504,
    "internal_error": 500,
}


class AgentError(Exception):
    def __init__(self, type: str, message: str):
        super().__init__(message)
        self.type = type
        self.message = message[:500]

    def as_dict(self) -> dict:
        return {"type": self.type, "message": self.message}
