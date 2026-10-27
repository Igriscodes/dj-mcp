from pathlib import Path
from typing import Any, List, Optional
from ai_dj.config import DJConfig, get_default_config
from ai_dj.errors import DJError, ValidationError, PrerequisiteError


def resolve_safe_path(
    user_path: str,
    config: Optional[DJConfig] = None,
    must_exist: bool = False,
    allow_absolute: bool = True,
) -> Path:
    cfg = config or get_default_config()
    path = Path(user_path).expanduser()

    if not path.is_absolute():
        path = (cfg.workspace_dir / path).resolve()
    else:
        path = path.resolve()

    if cfg.strict_workspace:
        # Check that path is within workspace_dir, cache_dir, or temp_dir
        allowed_parents = [cfg.workspace_dir.resolve(), cfg.cache_dir.resolve(), cfg.temp_dir.resolve()]
        is_safe = False
        for parent in allowed_parents:
            try:
                path.relative_to(parent)
                is_safe = True
                break
            except ValueError:
                continue

        if not is_safe and not allow_absolute:
            raise ValidationError(
                param_name="path",
                message=f"Path '{user_path}' is outside permitted workspace directory '{cfg.workspace_dir}'",
                suggested_value=str(cfg.workspace_dir / Path(user_path).name),
            )

    if must_exist and not path.exists():
        raise DJError(
            code="FILE_NOT_FOUND",
            message=f"File not found: '{path}'",
            reason="file_does_not_exist",
            suggested_tool="library_scan",
            suggested_arguments={"directory": str(cfg.workspace_dir)},
            details={"path": str(path)},
        )

    return path


def validate_sample_rate(sr: int) -> int:
    valid_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]
    if sr not in valid_rates:
        raise ValidationError(
            param_name="sample_rate",
            message=f"Sample rate {sr} Hz is unsupported. Choose from: {valid_rates}",
            suggested_value=44100,
        )
    return sr


def validate_gain(gain_db: float, min_db: float = -60.0, max_db: float = 24.0) -> float:
    if not (min_db <= gain_db <= max_db):
        raise ValidationError(
            param_name="gain_db",
            message=f"Gain {gain_db} dB is out of valid range [{min_db}, {max_db}] dB.",
            suggested_value=0.0,
        )
    return float(gain_db)


def validate_pan(pan: float) -> float:
    if not (-1.0 <= pan <= 1.0):
        raise ValidationError(
            param_name="pan",
            message=f"Pan {pan} must be between -1.0 (hard left) and +1.0 (hard right).",
            suggested_value=0.0,
        )
    return float(pan)


def validate_disk_space_for_render(est_mb: int = 50, config: Optional[DJConfig] = None) -> None:
    cfg = config or get_default_config()
    free_mb = cfg.get_free_disk_space_mb()
    if free_mb < est_mb:
        raise PrerequisiteError(
            code="INSUFFICIENT_DISK_SPACE",
            message=f"Workspace has only {free_mb} MB free, but {est_mb} MB is estimated for rendering.",
            required_action="free_disk_space",
            suggested_tool="system_capabilities",
            suggested_arguments={},
            details={"free_mb": free_mb, "required_mb": est_mb},
        )
