import bisect
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BeatGrid(BaseModel):
    bpm: float = Field(gt=10.0, lt=400.0)
    first_beat_sec: float = Field(ge=0.0)
    time_signature_num: int = Field(default=4, ge=1, le=16)
    time_signature_denom: int = Field(default=4, ge=1, le=16)
    beat_positions: List[float] = Field(default_factory=list)
    downbeat_positions: List[float] = Field(default_factory=list)

    @property
    def beat_interval_sec(self) -> float:
        return 60.0 / self.bpm

    @classmethod
    def generate_constant_grid(
        cls,
        bpm: float,
        first_beat_sec: float,
        duration_sec: float,
        time_signature_num: int = 4,
    ) -> "BeatGrid":
        beat_interval = 60.0 / bpm
        beats: List[float] = []
        downbeats: List[float] = []

        curr = first_beat_sec
        beat_idx = 0
        while curr <= duration_sec:
            beats.append(round(curr, 4))
            if beat_idx % time_signature_num == 0:
                downbeats.append(round(curr, 4))
            curr += beat_interval
            beat_idx += 1

        return cls(
            bpm=bpm,
            first_beat_sec=first_beat_sec,
            time_signature_num=time_signature_num,
            beat_positions=beats,
            downbeat_positions=downbeats,
        )

    def get_nearest_beat(self, position_sec: float) -> float:
        if not self.beat_positions:
            return position_sec
        idx = bisect.bisect_left(self.beat_positions, position_sec)
        if idx == 0:
            return self.beat_positions[0]
        if idx >= len(self.beat_positions):
            return self.beat_positions[-1]
        before = self.beat_positions[idx - 1]
        after = self.beat_positions[idx]
        return before if (position_sec - before) < (after - position_sec) else after

    def get_next_beat(self, position_sec: float) -> float:
        if not self.beat_positions:
            return position_sec
        idx = bisect.bisect_right(self.beat_positions, position_sec)
        if idx < len(self.beat_positions):
            return self.beat_positions[idx]
        return self.beat_positions[-1]

    def get_prev_beat(self, position_sec: float) -> float:
        if not self.beat_positions:
            return position_sec
        idx = bisect.bisect_left(self.beat_positions, position_sec)
        if idx > 0:
            return self.beat_positions[idx - 1]
        return self.beat_positions[0]

    def quantize_position(self, position_sec: float, quantum: str = "beat") -> float:
        if not self.beat_positions:
            return position_sec

        interval = self.beat_interval_sec
        quantum_map = {
            "bar": interval * self.time_signature_num,
            "beat": interval,
            "1/2": interval / 2.0,
            "half": interval / 2.0,
            "1/4": interval / 4.0,
            "quarter": interval / 4.0,
            "1/8": interval / 8.0,
            "eighth": interval / 8.0,
            "1/16": interval / 16.0,
            "sixteenth": interval / 16.0,
            "1/32": interval / 32.0,
        }
        step = quantum_map.get(quantum, interval)
        # Offset relative to first_beat_sec
        diff = position_sec - self.first_beat_sec
        steps = round(diff / step)
        return round(self.first_beat_sec + (steps * step), 4)

    def shift_grid(self, offset_sec: float) -> None:
        self.first_beat_sec = max(0.0, self.first_beat_sec + offset_sec)
        self.beat_positions = [max(0.0, round(b + offset_sec, 4)) for b in self.beat_positions]
        self.downbeat_positions = [max(0.0, round(db + offset_sec, 4)) for db in self.downbeat_positions]

    def correct_bpm(self, new_bpm: float, duration_sec: Optional[float] = None) -> None:
        self.bpm = new_bpm
        max_dur = duration_sec or (self.beat_positions[-1] if self.beat_positions else 300.0)
        regen = self.generate_constant_grid(
            bpm=new_bpm,
            first_beat_sec=self.first_beat_sec,
            duration_sec=max_dur,
            time_signature_num=self.time_signature_num,
        )
        self.beat_positions = regen.beat_positions
        self.downbeat_positions = regen.downbeat_positions
