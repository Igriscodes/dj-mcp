from typing import Optional
from pydantic import BaseModel, Field


class Loop(BaseModel):
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    length_beats: float = Field(default=4.0, gt=0.0)
    is_active: bool = Field(default=True)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def double_length(self) -> None:
        self.length_beats *= 2.0
        dur = self.duration_sec
        self.end_sec = round(self.start_sec + (dur * 2.0), 4)

    def halve_length(self) -> None:
        self.length_beats = max(0.25, self.length_beats / 2.0)
        dur = self.duration_sec
        self.end_sec = round(self.start_sec + (dur / 2.0), 4)

    def shift(self, offset_sec: float) -> None:
        dur = self.duration_sec
        self.start_sec = max(0.0, round(self.start_sec + offset_sec, 4))
        self.end_sec = round(self.start_sec + dur, 4)
