from typing import Any, Callable, Dict
import functools
from ai_dj.errors import DJError
from ai_dj.mcp.responses import error_response


def handle_tool_errors(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except DJError as e:
            return error_response(e)
        except Exception as e:
            generic_err = DJError(
                code="INTERNAL_ERROR",
                message=str(e),
                reason="unexpected_exception",
                recoverable=False,
            )
            return error_response(generic_err)
    return wrapper
