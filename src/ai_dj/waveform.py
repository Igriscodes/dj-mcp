import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
import scipy.signal as signal


def generate_waveform_overview(
    audio_path: Path,
    num_points: int = 256,
) -> Dict[str, Any]:
    data, sr = sf.read(str(audio_path), always_2d=True)
    mono = np.mean(data, axis=1)
    total_samples = len(mono)

    if total_samples == 0:
        return {"num_points": 0, "peaks": [], "low_energy": [], "mid_energy": [], "high_energy": []}

    hop_size = max(1, total_samples // num_points)
    peaks: List[float] = []

    for i in range(num_points):
        start = i * hop_size
        end = min(total_samples, start + hop_size)
        if start >= total_samples:
            break
        chunk = mono[start:end]
        peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
        peaks.append(round(peak, 4))

    # Frequency-split waveform energy bands using butterworth filters
    # Low: < 250 Hz (bass/kick)
    # Mid: 250 Hz - 2500 Hz (vocals/synths)
    # High: > 2500 Hz (cymbals/percussion)
    nyq = sr / 2.0
    low_b, low_a = signal.butter(2, min(250.0 / nyq, 0.99), btype="low")
    mid_b, mid_a = signal.butter(2, [min(250.0 / nyq, 0.98), min(2500.0 / nyq, 0.99)], btype="band")
    high_b, high_a = signal.butter(2, min(2500.0 / nyq, 0.99), btype="high")

    low_sig = np.abs(signal.lfilter(low_b, low_a, mono))
    mid_sig = np.abs(signal.lfilter(mid_b, mid_a, mono))
    high_sig = np.abs(signal.lfilter(high_b, high_a, mono))

    low_energy: List[float] = []
    mid_energy: List[float] = []
    high_energy: List[float] = []

    for i in range(len(peaks)):
        start = i * hop_size
        end = min(total_samples, start + hop_size)
        low_val = float(np.mean(low_sig[start:end])) if end > start else 0.0
        mid_val = float(np.mean(mid_sig[start:end])) if end > start else 0.0
        high_val = float(np.mean(high_sig[start:end])) if end > start else 0.0

        low_energy.append(round(low_val, 4))
        mid_energy.append(round(mid_val, 4))
        high_energy.append(round(high_val, 4))

    return {
        "num_points": len(peaks),
        "duration_seconds": round(total_samples / sr, 3),
        "sample_rate": sr,
        "peaks": peaks,
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
    }
