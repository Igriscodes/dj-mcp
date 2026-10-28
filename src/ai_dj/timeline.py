from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ai_dj.errors import DJError, ResourceNotFoundError, ValidationError
from ai_dj.tracks import Track


class TimelineMarker(BaseModel):
    marker_id: str
    name: str
    time_sec: float = Field(ge=0.0)
    color: Optional[str] = None


class Timeline(BaseModel):
    tracks: Dict[str, Track] = Field(default_factory=dict)
    markers: List[TimelineMarker] = Field(default_factory=list)
    bpm: float = Field(default=120.0, gt=0.0)
    time_signature: str = Field(default="4/4")
    playhead_sec: float = Field(default=0.0, ge=0.0)
    loop_start_sec: Optional[float] = None
    loop_end_sec: Optional[float] = None
    is_looping: bool = False

    @property
    def total_duration_sec(self) -> float:
        if not self.tracks:
            return 0.0
        return max((t.duration_sec for t in self.tracks.values()), default=0.0)

    def set_tempo(self, new_bpm: float, stretch_clips: bool = True) -> float:
        if new_bpm <= 0.0:
            raise ValidationError("bpm", "BPM must be strictly positive (> 0.0)")
        ratio = new_bpm / self.bpm
        if stretch_clips and abs(ratio - 1.0) > 1e-4:
            for track in self.tracks.values():
                for clip in track.clips:
                    clip.timeline_position_sec = round(clip.timeline_position_sec / ratio, 4)
                    clip.time_stretch_rate = round(clip.time_stretch_rate * ratio, 4)
                    clip.duration_sec = round(clip.duration_sec / ratio, 4)
            for m in self.markers:
                m.time_sec = round(m.time_sec / ratio, 4)
        self.bpm = round(float(new_bpm), 2)
        return self.bpm

    def add_track(self, track: Track) -> None:
        self.tracks[track.track_id] = track

    def get_track(self, track_id: str) -> Track:
        if track_id not in self.tracks:
            raise ResourceNotFoundError(resource_type="track", resource_id=track_id)
        return self.tracks[track_id]

    def remove_track(self, track_id: str) -> bool:
        if track_id in self.tracks:
            del self.tracks[track_id]
            return True
        return False

    def add_marker(self, name: str, time_sec: float, color: Optional[str] = None) -> TimelineMarker:
        marker_id = f"m_{len(self.markers) + 1}"
        m = TimelineMarker(marker_id=marker_id, name=name, time_sec=time_sec, color=color)
        self.markers.append(m)
        self.markers.sort(key=lambda x: x.time_sec)
        return m
