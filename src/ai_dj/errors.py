from typing import Any, Dict, Optional


class DJError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        reason: Optional[str] = None,
        required_action: Optional[str] = None,
        suggested_tool: Optional[str] = None,
        suggested_arguments: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.reason = reason or code.lower()
        self.required_action = required_action or suggested_tool
        self.suggested_tool = suggested_tool
        self.suggested_arguments = suggested_arguments or {}
        self.recoverable = recoverable
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "reason": self.reason,
            "recoverable": self.recoverable,
        }
        if self.required_action:
            data["required_action"] = self.required_action
        if self.suggested_tool:
            data["suggested_tool"] = self.suggested_tool
        if self.suggested_arguments:
            data["suggested_arguments"] = self.suggested_arguments
        if self.details:
            data["details"] = self.details
        return data


class PrerequisiteError(DJError):
    def __init__(
        self,
        code: str,
        message: str,
        required_action: str,
        suggested_tool: str,
        suggested_arguments: Optional[Dict[str, Any]] = None,
        reason: str = "prerequisite_missing",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            reason=reason,
            required_action=required_action,
            suggested_tool=suggested_tool,
            suggested_arguments=suggested_arguments,
            recoverable=True,
            details=details,
        )


class ResourceNotFoundError(DJError):
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        suggested_tool: Optional[str] = None,
        suggested_arguments: Optional[Dict[str, Any]] = None,
    ):
        sug_tool = suggested_tool or f"{resource_type.lower()}_create"
        sug_args = suggested_arguments or {f"{resource_type.lower()}_id": resource_id}
        super().__init__(
            code=f"{resource_type.upper()}_NOT_FOUND",
            message=f"{resource_type.capitalize()} '{resource_id}' does not exist in current session.",
            reason="resource_not_found",
            required_action=sug_tool,
            suggested_tool=sug_tool,
            suggested_arguments=sug_args,
            recoverable=True,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ValidationError(DJError):
    def __init__(
        self,
        param_name: str,
        message: str,
        suggested_value: Optional[Any] = None,
    ):
        super().__init__(
            code="INVALID_PARAMETER",
            message=f"Parameter '{param_name}' is invalid: {message}",
            reason="validation_failed",
            recoverable=True,
            details={"parameter": param_name, "suggested_value": suggested_value},
        )


class EngineUnavailableError(DJError):
    def __init__(
        self,
        engine_name: str,
        required_feature: str,
        install_command: Optional[str] = None,
    ):
        super().__init__(
            code="ENGINE_UNAVAILABLE",
            message=f"Required engine '{engine_name}' for '{required_feature}' is not available on this platform.",
            reason="engine_missing",
            required_action="install_engine",
            suggested_tool="system_capabilities",
            suggested_arguments={},
            recoverable=False,
            details={"engine": engine_name, "feature": required_feature, "install_command": install_command},
        )


class AudioProcessingError(DJError):
    def __init__(
        self,
        operation: str,
        details_msg: str,
        recoverable: bool = True,
    ):
        super().__init__(
            code="AUDIO_PROCESSING_ERROR",
            message=f"Audio processing failed during '{operation}': {details_msg}",
            reason="dsp_failure",
            recoverable=recoverable,
            details={"operation": operation, "error": details_msg},
        )
