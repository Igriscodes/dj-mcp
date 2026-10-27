import math
from typing import Tuple
import numpy as np


def apply_panning(audio: np.ndarray, pan: float) -> np.ndarray:
    # pan is in [-1.0, 1.0] where -1 is left, 0 is center, +1 is right
    # Constant-power panning law:
    # theta = (pan + 1.0) * (pi / 4.0)
    # gain_left = cos(theta), gain_right = sin(theta)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)

    p = max(-1.0, min(1.0, float(pan)))
    theta = (p + 1.0) * (math.pi / 4.0)
    gain_l = math.cos(theta) * math.sqrt(2.0)
    gain_r = math.sin(theta) * math.sqrt(2.0)

    out = audio.copy()
    out[0, :] *= gain_l
    out[1, :] *= gain_r
    return out


def apply_stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
    # width: 0.0 = pure mono, 1.0 = unchanged stereo, 2.0 = extra wide
    if audio.ndim == 1 or audio.shape[0] < 2:
        return audio

    left = audio[0, :]
    right = audio[1, :]

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    side_scaled = side * max(0.0, float(width))
    new_left = mid + side_scaled
    new_right = mid - side_scaled

    out = audio.copy()
    out[0, :] = new_left
    out[1, :] = new_right
    return out


def apply_phase_inversion(audio: np.ndarray, invert_left: bool = False, invert_right: bool = False) -> np.ndarray:
    out = audio.copy()
    if invert_left and out.shape[0] >= 1:
        out[0, :] = -out[0, :]
    if invert_right and out.shape[0] >= 2:
        out[1, :] = -out[1, :]
    return out


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    # Subtract mean per channel
    mean = np.mean(audio, axis=-1, keepdims=True)
    return audio - mean


def calculate_phase_correlation(audio: np.ndarray) -> float:
    if audio.ndim == 1 or audio.shape[0] < 2:
        return 1.0
    l = audio[0, :]
    r = audio[1, :]
    denom = (np.linalg.norm(l) * np.linalg.norm(r)) + 1e-9
    if denom == 0:
        return 1.0
    corr = np.dot(l, r) / denom
    return round(float(np.clip(corr, -1.0, 1.0)), 3)
