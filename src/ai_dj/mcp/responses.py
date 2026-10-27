import uuid
from typing import Any, Dict, List, Optional
from ai_dj.errors import DJError


def success_response(
    result: Any,
    operation_id: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    next_actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "success": True,
        "operation_id": operation_id or f"op_{uuid.uuid4().hex[:8]}",
        "result": result,
        "warnings": warnings or [],
        "next_actions": next_actions or [],
    }


def error_response(
    error: DJError,
    next_actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    err_dict = error.to_dict()
    suggested = []
    if error.suggested_tool:
        suggested.append({
            "action": error.required_action or error.suggested_tool,
            "tool": error.suggested_tool,
            "arguments": error.suggested_arguments,
        })
    return {
        "success": False,
        "error": err_dict,
        "next_actions": next_actions or suggested,
    }


def job_response(
    job_id: str,
    status: str = "running",
    operation_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "success": True,
        "status": status,
        "job_id": job_id,
        "operation_id": operation_id or f"op_{uuid.uuid4().hex[:8]}",
        "next_actions": [
            {
                "action": "check_job_status",
                "tool": "job_status",
                "arguments": {"job_id": job_id},
            }
        ],
    }
