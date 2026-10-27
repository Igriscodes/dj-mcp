import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from ai_dj.errors import DJError


def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}")
    try:
        with open(temp_target, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_target, target_path)
    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        raise DJError(
            code="ATOMIC_WRITE_FAILED",
            message=f"Failed to atomically write file '{target_path}': {str(e)}",
            reason="io_error",
            recoverable=True,
            details={"path": str(target_path)},
        ) from e


def atomic_write_text(target_path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(target_path, text.encode(encoding))


class StateSnapshotManager:
    def __init__(self, snapshots_dir: Path, max_snapshots: int = 50):
        self.snapshots_dir = snapshots_dir
        self.max_snapshots = max_snapshots
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.undo_stack: List[str] = []
        self.redo_stack: List[str] = []

    def save_snapshot(self, snapshot_id: str, state_dict: Dict[str, Any]) -> Path:
        target_file = self.snapshots_dir / f"{snapshot_id}.json"
        text = json.dumps(state_dict, indent=2)
        atomic_write_text(target_file, text)
        self.undo_stack.append(snapshot_id)
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_snapshots:
            oldest = self.undo_stack.pop(0)
            old_file = self.snapshots_dir / f"{oldest}.json"
            if old_file.exists():
                old_file.unlink()
        return target_file

    def load_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        target_file = self.snapshots_dir / f"{snapshot_id}.json"
        if not target_file.exists():
            raise DJError(
                code="SNAPSHOT_NOT_FOUND",
                message=f"Snapshot '{snapshot_id}' does not exist.",
                suggested_tool="project_snapshot",
            )
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise DJError(
                code="SNAPSHOT_CORRUPTED",
                message=f"Snapshot '{snapshot_id}' could not be decoded: {str(e)}",
                recoverable=True,
            ) from e

    def undo(self) -> Optional[Dict[str, Any]]:
        if len(self.undo_stack) < 2:
            return None
        current = self.undo_stack.pop()
        self.redo_stack.append(current)
        prev_id = self.undo_stack[-1]
        return self.load_snapshot(prev_id)

    def redo(self) -> Optional[Dict[str, Any]]:
        if not self.redo_stack:
            return None
        target_id = self.redo_stack.pop()
        self.undo_stack.append(target_id)
        return self.load_snapshot(target_id)
