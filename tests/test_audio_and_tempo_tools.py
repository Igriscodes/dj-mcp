import math
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from ai_dj.mcp.tools import (
    tool_audio_change_tempo,
    tool_audio_change_pitch,
    tool_audio_change_speed,
    tool_audio_paulstretch,
    tool_audio_reverse,
    tool_audio_invert_polarity,
    tool_audio_truncate_silence,
    tool_audio_denoise,
    tool_audio_declick,
    tool_audio_vocal_remover,
    tool_timeline_set_tempo,
    tool_timeline_get_tempo,
    tool_project_create,
    tool_deck_create,
    tool_deck_load,
    tool_deck_sync,
    tool_clip_create,
)
from ai_dj.session import get_active_session


def test_audio_change_tempo(tmp_path, test_audio_file):
    out_bpm = tmp_path / "bpm_change.wav"
    res1 = tool_audio_change_tempo(
        input_file=str(test_audio_file),
        output_file=str(out_bpm),
        source_bpm=120.0,
        target_bpm=140.0,
        preserve_pitch=True,
    )
    assert res1["success"]
    assert out_bpm.exists()
    info1 = sf.info(str(out_bpm))
    orig_info = sf.info(str(test_audio_file))
    # At 140 BPM, duration should be ~120/140 of original
    assert abs(info1.duration - (orig_info.duration * 120.0 / 140.0)) < 0.2

    # Change by percent
    out_pct = tmp_path / "pct_change.wav"
    res2 = tool_audio_change_tempo(
        input_file=str(test_audio_file),
        output_file=str(out_pct),
        tempo_change_percent=20.0,
    )
    assert res2["success"]
    assert out_pct.exists()


def test_audio_change_pitch(tmp_path, test_audio_file):
    # Shift by semitones
    out_pitch = tmp_path / "pitch_shifted.wav"
    res1 = tool_audio_change_pitch(
        input_file=str(test_audio_file),
        output_file=str(out_pitch),
        semitones=2.0,
        preserve_tempo=True,
    )
    assert res1["success"]
    assert res1["result"]["semitones_applied"] == 2.0
    info = sf.info(str(out_pitch))
    orig_info = sf.info(str(test_audio_file))
    assert abs(info.duration - orig_info.duration) < 0.1

    # Shift by musical keys (e.g. G minor to A minor = +2 semitones)
    out_key = tmp_path / "key_shifted.wav"
    res2 = tool_audio_change_pitch(
        input_file=str(test_audio_file),
        output_file=str(out_key),
        from_key="G minor",
        to_key="A minor",
    )
    assert res2["success"]
    assert res2["result"]["semitones_applied"] == 2.0


def test_audio_change_speed_and_paulstretch(tmp_path, test_audio_file):
    # Change speed (varispeed)
    out_speed = tmp_path / "speed.wav"
    res1 = tool_audio_change_speed(
        input_file=str(test_audio_file),
        output_file=str(out_speed),
        speed_ratio=1.25,
    )
    assert res1["success"]
    info1 = sf.info(str(out_speed))
    orig_info = sf.info(str(test_audio_file))
    assert abs(info1.duration - (orig_info.duration / 1.25)) < 0.1

    # Vinyl RPM change: 33.33 to 45 RPM
    out_vinyl = tmp_path / "vinyl.wav"
    res2 = tool_audio_change_speed(
        input_file=str(test_audio_file),
        output_file=str(out_vinyl),
        from_rpm=33.333,
        to_rpm=45.0,
    )
    assert res2["success"]
    assert res2["result"]["speed_ratio"] > 1.3

    # Paulstretch
    out_paul = tmp_path / "paulstretch.wav"
    res3 = tool_audio_paulstretch(
        input_file=str(test_audio_file),
        output_file=str(out_paul),
        stretch_factor=3.0,
        window_size_sec=0.1,
    )
    assert res3["success"]
    info3 = sf.info(str(out_paul))
    assert abs(info3.duration - (orig_info.duration * 3.0)) < 0.3


def test_audio_reverse_and_polarity(tmp_path, test_audio_file):
    out_rev = tmp_path / "reversed.wav"
    res1 = tool_audio_reverse(str(test_audio_file), str(out_rev))
    assert res1["success"]
    data_rev, _ = sf.read(str(out_rev))
    data_orig, _ = sf.read(str(test_audio_file))
    assert np.allclose(data_rev[0], data_orig[-1], atol=1e-4)

    out_inv = tmp_path / "inverted.wav"
    res2 = tool_audio_invert_polarity(str(test_audio_file), str(out_inv), channels="both")
    assert res2["success"]
    data_inv, _ = sf.read(str(out_inv))
    assert np.allclose(data_inv, -data_orig, atol=1e-4)


def test_audio_cleanup_and_processing(tmp_path, test_audio_file):
    # Truncate silence
    out_sil = tmp_path / "silence_trunc.wav"
    res1 = tool_audio_truncate_silence(str(test_audio_file), str(out_sil), threshold_db=-60.0)
    assert res1["success"]

    # Denoise
    out_denoise = tmp_path / "denoised.wav"
    res2 = tool_audio_denoise(str(test_audio_file), str(out_denoise), reduction_db=12.0)
    assert res2["success"]

    # Declick
    out_declick = tmp_path / "declicked.wav"
    res3 = tool_audio_declick(str(test_audio_file), str(out_declick), sensitivity=0.7)
    assert res3["success"]

    # Vocal remover
    out_vocal = tmp_path / "vocal_removed.wav"
    res4 = tool_audio_vocal_remover(str(test_audio_file), str(out_vocal), mode="remove_vocals")
    assert res4["success"]


def test_timeline_tempo_control(tmp_path, test_audio_file):
    sess = get_active_session()
    tool_project_create(project_id="p_tempo", name="Tempo Project")

    # Add clip
    tool_clip_create(
        track_id="t_main",
        clip_id="c_main",
        file_path=str(test_audio_file),
        timeline_position_sec=2.0,
        duration_sec=4.0,
    )

    t_info = tool_timeline_get_tempo()
    assert t_info["result"]["bpm"] == 120.0

    # Set new tempo 150 BPM with clip stretching
    res = tool_timeline_set_tempo(bpm=150.0, stretch_clips=True)
    assert res["success"]
    assert res["result"]["new_bpm"] == 150.0

    proj = sess.require_project()
    clip = proj.timeline.tracks["t_main"].clips[0]
    # At 150 vs 120 BPM (ratio 1.25), position should be 2.0 / 1.25 = 1.6
    assert abs(clip.timeline_position_sec - 1.6) < 0.05


def test_deck_sync_smart_half_double_time(test_audio_file):
    sess = get_active_session()
    tool_project_create(project_id="p_dj_sync", name="DJ Sync Project")

    tool_deck_load(deck_id="deck_a", audio_path=str(test_audio_file))
    tool_deck_load(deck_id="deck_b", audio_path=str(test_audio_file))

    proj = sess.require_project()
    proj.decks["deck_a"].original_bpm = 85.0
    proj.decks["deck_a"].current_bpm = 85.0

    proj.decks["deck_b"].original_bpm = 180.0
    proj.decks["deck_b"].current_bpm = 180.0

    # Sync with smart half/double time enabled
    res = tool_deck_sync(deck_id="deck_b", master_deck_id="deck_a", allow_half_double_time=True)
    assert res["success"]
    # Double time (85 * 2 = 170 BPM) should be chosen over 85 BPM (-52.7% stretch)
    assert res["result"]["sync_type"] == "double_time"
    assert res["result"]["synced_bpm"] == 170.0
    assert abs(res["result"]["pitch_percent"]) < 10.0


def test_stem_separation_job_submission(tmp_path, test_audio_file):
    from ai_dj.mcp.tools import tool_separate_stems
    res = tool_separate_stems(
        audio_path=str(test_audio_file),
        output_dir=str(tmp_path / "stems_test"),
        wait=False,
    )
    assert res["success"]
    assert "job_id" in res
    assert res["status"] in ("pending", "running")


def test_audio_merge_and_stems_merge(tmp_path, test_audio_file):
    from ai_dj.mcp.tools import tool_audio_merge, tool_stems_merge
    # 1. Create simulated stems
    stems_dir = tmp_path / "fake_stems"
    stems_dir.mkdir()
    d, sr = sf.read(str(test_audio_file))
    sf.write(str(stems_dir / "vocals.wav"), d, sr)
    sf.write(str(stems_dir / "drums.wav"), d, sr)
    sf.write(str(stems_dir / "bass.wav"), d, sr)

    # 2. Merge directory
    out_merged = tmp_path / "remerged.wav"
    res = tool_stems_merge(
        stems_dir=str(stems_dir),
        output_file=str(out_merged),
        normalize=True,
        target_peak_dbfs=-0.5,
    )
    assert res["success"]
    assert out_merged.exists()
    assert res["result"]["file_count"] == 3
    assert res["result"]["peak_dbfs"] == -0.5


def test_demucs_cuda_oom_cpu_fallback(tmp_path, test_audio_file, monkeypatch):
    import torch
    from unittest.mock import MagicMock
    from ai_dj.engines.separation_engine import SeparationEngine
    import demucs.apply

    engine = SeparationEngine()
    orig_apply_model = demucs.apply.apply_model
    call_devices = []

    def mock_apply_model(model, input_tensor, **kwargs):
        dev = str(input_tensor.device)
        call_devices.append(dev)
        if "cuda" in dev:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory while allocating tensor")
        # For CPU fallback return dummy tensor with (batch=1, sources=4, channels=2, time)
        return torch.zeros((1, 4, 2, input_tensor.shape[-1]))

    monkeypatch.setattr(demucs.apply, "apply_model", mock_apply_model)

    out_dir = tmp_path / "oom_stems"
    res = engine.separate(
        audio_path=test_audio_file,
        output_dir=out_dir,
        device="cuda",
    )
    assert engine.fallback_occurred is True
    assert engine.last_device_used == "cpu"
    assert len(call_devices) > 0
    assert (out_dir / "vocals.mp3").exists()
