"""
Shared helpers for public OMNIXAN module APIs.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, TypedDict, cast


class APIResponse(TypedDict, total=False):
    """Compatibility-friendly public response envelope."""

    status: Literal["success", "error"]
    operation: str
    error: str


def require_operation(params: Mapping[str, Any]) -> str:
    """Validate and extract the public ``operation`` field."""
    operation = params.get("operation")
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("Missing required 'operation' string in params")
    return operation


def success_response(
    operation: str,
    payload: Mapping[str, Any] | None = None,
) -> APIResponse:
    """Build a standardized success response without hiding legacy fields."""
    response: dict[str, Any] = {
        "status": "success",
        "operation": operation,
    }
    if payload:
        response.update(dict(payload))
    return cast(APIResponse, response)


def error_response(
    operation: str,
    message: str,
    payload: Mapping[str, Any] | None = None,
) -> APIResponse:
    """Build a standardized error response."""
    response: dict[str, Any] = {
        "status": "error",
        "operation": operation,
        "error": message,
    }
    if payload:
        response.update(dict(payload))
    return cast(APIResponse, response)
