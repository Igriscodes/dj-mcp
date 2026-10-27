import math
from typing import Tuple
import numpy as np
import pyloudnorm as pyln


def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    # audio is (channels, samples) -> pyloudnorm expects (samples, channels)
    data = audio.T if audio.ndim == 2 else np.expand_dims(audio, axis=1)
    meter = pyln.Meter(sample_rate)
    try:
        lufs = meter.integrated_loudness(data)
        return round(float(lufs), 2)
    except Exception:
        # Fallback to RMS calculation
        rms = np.sqrt(np.mean(np.square(data))) + 1e-9
        return round(float(20.0 * math.log10(rms) - 3.0), 2)


def calculate_lufs_gain_db(current_lufs: float, target_lufs: float) -> float:
    return round(target_lufs - current_lufs, 2)


def apply_tpdf_dither(audio: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    if bit_depth >= 32:
        return audio
    # Triangular Probability Density Function (TPDF) dither
    lsb = 1.0 / (2 ** (bit_depth - 1))
    noise = (np.random.random_sample(audio.shape) - np.random.random_sample(audio.shape)) * lsb
    dithered = audio + noise
    return np.clip(dithered, -1.0, 1.0).astype(np.float32)
