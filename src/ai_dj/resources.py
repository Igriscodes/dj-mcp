import os
import tempfile
import atexit
from pathlib import Path
from typing import List, Optional
import psutil

_MANAGED_TEMP_FILES: List[Path] = []


def create_managed_temp_file(suffix: str = ".wav", dir_path: Optional[Path] = None) -> Path:
    if dir_path is not None:
        dir_path.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(suffix=suffix, dir=str(dir_path))
    else:
        fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(name)
    _MANAGED_TEMP_FILES.append(path)
    return path


def cleanup_managed_temp_files() -> None:
    global _MANAGED_TEMP_FILES
    remaining: List[Path] = []
    for p in _MANAGED_TEMP_FILES:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            remaining.append(p)
    _MANAGED_TEMP_FILES = remaining


atexit.register(cleanup_managed_temp_files)


def get_current_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def check_memory_threshold(limit_mb: float = 4096.0) -> bool:
    current = get_current_process_memory_mb()
    return current < limit_mb
