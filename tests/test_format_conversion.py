from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from ai_dj.config import DJConfig, get_default_config
from ai_dj.engines.ffmpeg_engine import FFmpegEngine
from ai_dj.export import export_audio_to_file
from ai_dj.mcp.tools import tool_audio_convert_format, tool_audio_merge, tool_render_project


@pytest.fixture
def test_audio_fixture(tmp_path: Path) -> Path:
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Stereo tone
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.vstack([left, right])
    wav_path = tmp_path / "test_tone.wav"
    export_audio_to_file(stereo, wav_path, sample_rate=sr, audio_format="wav")
    return wav_path


def test_ffmpeg_engine_convert_to_mp3(test_audio_fixture: Path, tmp_path: Path):
    engine = FFmpegEngine()
    assert engine.is_available() is True
    out_mp3 = tmp_path / "output.mp3"
    result = engine.convert(test_audio_fixture, out_mp3, bit_rate_kbps=320)
    assert result.exists()
    assert result.suffix.lower() == ".mp3"

    probe = engine.probe(result)
    assert probe["codec"] in ("mp3", "mp3float")
    assert probe["channels"] == 2
    assert probe["duration_seconds"] > 1.8


def test_tool_audio_convert_format_single(test_audio_fixture: Path, tmp_path: Path):
    res = tool_audio_convert_format(
        input_file=str(test_audio_fixture),
        target_format="mp3",
        bit_rate_kbps=320,
    )
    assert res["success"] is True
    out_file = Path(res["result"]["output_file"])
    assert out_file.exists()
    assert out_file.suffix == ".mp3"
    assert res["result"]["target_format"] == "mp3"
    assert res["result"]["bit_rate_kbps"] == 320
    assert "compression_ratio" in res["result"]
    assert res["result"]["space_saved_mb"] >= 0.0


def test_tool_audio_convert_format_batch(tmp_path: Path):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    batch_dir = tmp_path / "batch_stems"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for stem in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
        d = 0.3 * np.sin(2 * np.pi * 440 * t)
        stereo = np.vstack([d, d])
        export_audio_to_file(stereo, batch_dir / stem, sample_rate=sr, audio_format="wav")

    res = tool_audio_convert_format(
        input_file=str(batch_dir),
        target_format="mp3",
        bit_rate_kbps=320,
    )
    assert res["success"] is True
    assert res["result"]["files_converted_count"] == 4
    for stem in ["vocals.mp3", "drums.mp3", "bass.mp3", "other.mp3"]:
        assert (batch_dir / stem).exists()


def test_export_audio_default_format_is_mp3(tmp_path: Path):
    cfg = get_default_config()
    assert cfg.default_export_format == "mp3"

    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    data = 0.4 * np.vstack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)])

    # Output without suffix should default to mp3
    out_no_ext = tmp_path / "mix_no_ext"
    exported = export_audio_to_file(data, out_no_ext, sample_rate=sr)
    assert exported.exists()
    assert exported.suffix == ".mp3"


def test_tool_audio_convert_format_flac_and_alac(test_audio_fixture: Path, tmp_path: Path):
    # Test FLAC conversion
    res_flac = tool_audio_convert_format(
        input_file=str(test_audio_fixture),
        target_format="flac",
    )
    assert res_flac["success"] is True
    out_flac = Path(res_flac["result"]["output_file"])
    assert out_flac.exists()
    assert out_flac.suffix == ".flac"

    # Test ALAC conversion
    res_alac = tool_audio_convert_format(
        input_file=str(test_audio_fixture),
        target_format="alac",
    )
    assert res_alac["success"] is True
    out_alac = Path(res_alac["result"]["output_file"])
    assert out_alac.exists()
    assert out_alac.suffix == ".m4a"
    assert res_alac["result"]["target_format"] == "alac"
