from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from ai_dj.export import export_audio_to_file
from ai_dj.mcp.tools import (
    tool_audio_apply_effect,
    tool_audio_reverb,
    tool_audio_change_tempo,
    tool_audio_change_pitch,
    tool_separate_stems,
)
from ai_dj.session import get_active_session


@pytest.fixture
def sample_songs_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "my_music_album"
    folder.mkdir(parents=True, exist_ok=True)
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    for idx, name in enumerate(["track_alpha.wav", "track_beta.wav", "track_gamma.wav"]):
        freq = 440.0 * (idx + 1)
        sig = 0.4 * np.vstack([np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)])
        export_audio_to_file(sig, folder / name, sample_rate=sr, audio_format="wav")

    return folder


def test_audio_reverb_single_song(sample_songs_folder: Path):
    track = sample_songs_folder / "track_alpha.wav"
    out_file = sample_songs_folder / "track_alpha_verb.wav"
    res = tool_audio_reverb(
        input_files=str(track),
        output_file=str(out_file),
        room_size=0.8,
        wet_level=0.4,
    )
    assert res["success"] is True
    assert out_file.exists()
    assert res["result"]["effect_applied"] == "reverb"
    assert res["result"]["parameters"]["room_size"] == 0.8


def test_audio_reverb_multiple_mentioned_songs(sample_songs_folder: Path):
    t1 = str(sample_songs_folder / "track_alpha.wav")
    t2 = str(sample_songs_folder / "track_beta.wav")

    # Mention as comma-separated string
    res = tool_audio_reverb(
        input_files=f"{t1}, {t2}",
        room_size=0.7,
        wet_level=0.35,
    )
    assert res["success"] is True
    assert res["result"]["files_processed_count"] == 2
    for r in res["result"]["results"]:
        assert Path(r["output_file"]).exists()
        assert "_reverb" in Path(r["output_file"]).name


def test_audio_reverb_entire_folder(sample_songs_folder: Path, tmp_path: Path):
    out_dir = tmp_path / "processed_reverb"
    res = tool_audio_reverb(
        input_files=str(sample_songs_folder),
        output_dir=str(out_dir),
        format="mp3",
        room_size=0.6,
        wet_level=0.3,
    )
    assert res["success"] is True
    assert res["result"]["files_processed_count"] == 3
    for name in ["track_alpha_reverb.mp3", "track_beta_reverb.mp3", "track_gamma_reverb.mp3"]:
        assert (out_dir / name).exists()


def test_audio_apply_effect_delay_and_chorus(sample_songs_folder: Path):
    track = sample_songs_folder / "track_alpha.wav"
    res_delay = tool_audio_apply_effect(
        input_files=str(track),
        effect_type="delay",
        parameters={"delay_seconds": 0.25, "feedback": 0.4},
    )
    assert res_delay["success"] is True
    assert Path(res_delay["result"]["output_file"]).exists()

    res_chorus = tool_audio_apply_effect(
        input_files=str(track),
        effect_type="chorus",
        parameters={"rate_hz": 1.5, "depth": 0.3},
    )
    assert res_chorus["success"] is True
    assert Path(res_chorus["result"]["output_file"]).exists()


def test_stem_separation_output_dir_anchoring(sample_songs_folder: Path, monkeypatch):
    import demucs.pretrained
    import demucs.apply
    import torch

    class DummyModel:
        samplerate = 44100
        sources = ["drums", "bass", "other", "vocals"]

        def eval(self):
            pass

        def to(self, device):
            return self

    monkeypatch.setattr(demucs.pretrained, "get_model", lambda name: DummyModel())
    monkeypatch.setattr(
        demucs.apply,
        "apply_model",
        lambda model, tensor, **kwargs: torch.zeros((1, 4, 2, tensor.shape[-1]))
    )

    track = sample_songs_folder / "track_alpha.wav"

    # 1. When output_dir is omitted, it should save in track.parent / stems_track_alpha
    res1 = tool_separate_stems(audio_path=str(track))
    assert res1["success"] is True
    expected_default_dir = sample_songs_folder / "stems_track_alpha"
    assert Path(res1["result"]["output_directory"]) == expected_default_dir
    assert expected_default_dir.exists()
    assert (expected_default_dir / "vocals.mp3").exists()

    # 2. When relative output_dir is given, it should resolve relative to track.parent
    res2 = tool_separate_stems(audio_path=str(track), output_dir="custom_stems")
    assert res2["success"] is True
    expected_rel_dir = sample_songs_folder / "custom_stems"
    assert Path(res2["result"]["output_directory"]) == expected_rel_dir
    assert expected_rel_dir.exists()
    assert (expected_rel_dir / "vocals.mp3").exists()


def test_audio_change_tempo_batch_folder(sample_songs_folder: Path, tmp_path: Path):
    out_dir = tmp_path / "faster_tracks"
    res = tool_audio_change_tempo(
        input_file=str(sample_songs_folder),
        output_dir=str(out_dir),
        tempo_change_percent=25.0,
    )
    assert res["success"] is True
    assert res["result"]["files_processed_count"] == 3
    for r in res["result"]["results"]:
        assert Path(r["output_file"]).exists()


def test_workspace_get_and_set(tmp_path: Path):
    from ai_dj.mcp.tools import tool_workspace_get, tool_workspace_set
    initial = tool_workspace_get()
    assert initial["success"] is True
    assert "workspace_dir" in initial["result"]

    new_workspace = tmp_path / "custom_dj_workspace"
    update_res = tool_workspace_set(directory=str(new_workspace))
    assert update_res["success"] is True
    assert Path(update_res["result"]["active_workspace_dir"]) == new_workspace.resolve()
    assert new_workspace.exists()

    check_res = tool_workspace_get()
    assert Path(check_res["result"]["workspace_dir"]) == new_workspace.resolve()
