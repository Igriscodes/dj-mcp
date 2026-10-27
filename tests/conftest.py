import math
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from ai_dj.config import DJConfig, set_default_config
from ai_dj.session import Session, set_active_session


@pytest.fixture(scope="session")
def test_audio_file(tmp_path_factory) -> Path:
    # Deterministic test audio file (stereo 44100Hz 4-beat rhythm + harmonic chord, 4.0 seconds at 120 BPM)
    sr = 44100
    bpm = 120.0
    duration = 4.0
    total_samples = int(duration * sr)
    beat_interval_samples = int((60.0 / bpm) * sr)

    t = np.linspace(0, duration, total_samples, endpoint=False)
    # Channel 1: 440 Hz chord (A4)
    ch1 = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    # Channel 2: 554.37 Hz chord (C#5)
    ch2 = 0.3 * np.sin(2 * np.pi * 554.37 * t)

    # Add sharp rhythmic clicks on every beat
    click_len = 1000
    click = np.sin(2 * np.pi * 1000.0 * np.linspace(0, 0.02, click_len)) * np.linspace(1.0, 0.0, click_len)
    for b in range(int(bpm * duration / 60.0)):
        idx = b * beat_interval_samples
        if idx + click_len < total_samples:
            ch1[idx:idx + click_len] += click * 0.5
            ch2[idx:idx + click_len] += click * 0.5

    stereo = np.stack([ch1, ch2], axis=1)
    stereo = np.clip(stereo, -0.95, 0.95).astype(np.float32)

    tmp_dir = tmp_path_factory.mktemp("audio_fixtures")
    file_path = tmp_dir / "beat_120bpm.wav"
    sf.write(str(file_path), stereo, sr, subtype="PCM_24")
    return file_path


@pytest.fixture(autouse=True)
def clean_session(tmp_path):
    cfg = DJConfig(
        workspace_dir=tmp_path,
        cache_dir=tmp_path / ".dj_cache",
        temp_dir=tmp_path / ".dj_temp",
        sample_rate=44100,
        strict_workspace=False,
    )
    cfg.initialize_directories()
    set_default_config(cfg)
    sess = Session(config=cfg)
    from ai_dj.mcp.tools import dispatch_tool
    sess.register_tool_dispatcher(dispatch_tool)
    set_active_session(sess)
    return sess
