import os
import shutil
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


def _detect_default_workspace() -> Path:
    if "DJ_WORKSPACE" in os.environ:
        return Path(os.environ["DJ_WORKSPACE"]).expanduser().resolve()
    cwd = Path.cwd().resolve()
    cwd_str = str(cwd)
    # If launched from inside an app or extension folder (e.g. LM Studio, VS Code, site-packages)
    if any(k in cwd_str for k in [".lmstudio", "site-packages", ".vscode", "AppData"]):
        music = Path.home() / "Music"
        if music.exists():
            return music
        return Path.home()
    return cwd


class DJConfig(BaseModel):
    workspace_dir: Path = Field(default_factory=_detect_default_workspace)
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".dj_cache")
    temp_dir: Path = Field(default_factory=lambda: Path.home() / ".dj_temp")
    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    channels: int = Field(default=2, ge=1, le=8)
    bit_depth: int = Field(default=24, ge=16, le=32)
    processing_precision: str = Field(default="float32")
    max_workers: int = Field(default_factory=lambda: max(1, os.cpu_count() or 1))
    memory_limit_mb: int = Field(default=4096, ge=512)
    use_gpu: bool = Field(default=True)
    strict_workspace: bool = Field(default=True)
    auto_backup: bool = Field(default=True)
    default_export_format: str = Field(default="mp3")
    default_bit_rate_kbps: int = Field(default=320, ge=64, le=320)

    def initialize_directories(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def check_disk_space(self, min_mb: int = 500) -> bool:
        usage = shutil.disk_usage(self.workspace_dir)
        free_mb = usage.free // (1024 * 1024)
        return free_mb >= min_mb

    def get_free_disk_space_mb(self) -> int:
        usage = shutil.disk_usage(self.workspace_dir)
        return usage.free // (1024 * 1024)


_DEFAULT_CONFIG: Optional[DJConfig] = None


def get_default_config() -> DJConfig:
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = DJConfig()
        _DEFAULT_CONFIG.initialize_directories()
    return _DEFAULT_CONFIG


def set_default_config(cfg: DJConfig) -> None:
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = cfg
    _DEFAULT_CONFIG.initialize_directories()
