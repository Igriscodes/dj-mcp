import bisect
from typing import List, Tuple
from pydantic import BaseModel, Field


class AutomationPoint(BaseModel):
    time_sec: float = Field(ge=0.0)
    value: float


class AutomationCurve(BaseModel):
    parameter_name: str
    target_id: str
    points: List[AutomationPoint] = Field(default_factory=list)

    def add_point(self, time_sec: float, value: float) -> None:
        self.points = [p for p in self.points if abs(p.time_sec - time_sec) > 1e-4]
        self.points.append(AutomationPoint(time_sec=time_sec, value=value))
        self.points.sort(key=lambda p: p.time_sec)

    def get_value_at(self, time_sec: float, default_value: float = 0.0) -> float:
        if not self.points:
            return default_value
        if time_sec <= self.points[0].time_sec:
            return self.points[0].value
        if time_sec >= self.points[-1].time_sec:
            return self.points[-1].value

        times = [p.time_sec for p in self.points]
        idx = bisect.bisect_right(times, time_sec)
        p0 = self.points[idx - 1]
        p1 = self.points[idx]

        # Linear interpolation
        ratio = (time_sec - p0.time_sec) / (p1.time_sec - p0.time_sec)
        return p0.value + ratio * (p1.value - p0.value)
