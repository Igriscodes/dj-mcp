from typing import Dict, Any


def calculate_tempo_ratio(source_bpm: float, target_bpm: float) -> float:
    if source_bpm <= 0.0 or target_bpm <= 0.0:
        return 1.0
    return target_bpm / source_bpm


def calculate_pitch_fader_percent(source_bpm: float, target_bpm: float) -> float:
    if source_bpm <= 0.0:
        return 0.0
    return ((target_bpm - source_bpm) / source_bpm) * 100.0


def bpm_from_pitch_fader_percent(source_bpm: float, fader_percent: float) -> float:
    return source_bpm * (1.0 + (fader_percent / 100.0))
