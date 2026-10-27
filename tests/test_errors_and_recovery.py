import pytest
from ai_dj.errors import DJError, PrerequisiteError, ResourceNotFoundError, ValidationError
from ai_dj.recovery import StateSnapshotManager, atomic_write_text


def test_structured_error_dict():
    err = PrerequisiteError(
        code="TRACK_NOT_ANALYZED",
        message="Track 'drop.wav' has not been analyzed for BPM and beat positions.",
        reason="beat_grid_required",
        required_action="analyze_track",
        suggested_tool="analyze_track",
        suggested_arguments={"track_id": "drop"},
    )
    d = err.to_dict()
    assert d["code"] == "TRACK_NOT_ANALYZED"
    assert d["recoverable"] is True
    assert d["suggested_tool"] == "analyze_track"
    assert d["suggested_arguments"] == {"track_id": "drop"}


def test_atomic_write(tmp_path):
    target = tmp_path / "test.txt"
    atomic_write_text(target, "Hello Production Audio")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "Hello Production Audio"


def test_undo_redo_stack(tmp_path):
    manager = StateSnapshotManager(tmp_path / "snaps")
    state1 = {"version": 1, "value": "A"}
    state2 = {"version": 2, "value": "B"}

    manager.save_snapshot("s1", state1)
    manager.save_snapshot("s2", state2)

    undone = manager.undo()
    assert undone["value"] == "A"

    redone = manager.redo()
    assert redone["value"] == "B"
