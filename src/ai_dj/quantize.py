from typing import Optional
from ai_dj.beatgrid import BeatGrid


def quantize_time_to_beat(time_sec: float, grid: Optional[BeatGrid], quantum: str = "beat") -> float:
    if grid is None or not grid.beat_positions:
        return time_sec
    return grid.quantize_position(time_sec, quantum=quantum)


def calculate_quantized_loop_end(
    start_sec: float,
    length_beats: float,
    grid: Optional[BeatGrid],
    bpm: float = 120.0,
) -> float:
    if grid:
        interval = grid.beat_interval_sec
    else:
        interval = 60.0 / bpm
    return round(start_sec + (length_beats * interval), 4)
