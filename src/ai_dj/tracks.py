from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from ai_dj.clips import AudioClip
from ai_dj.errors import DJError, ResourceNotFoundError
from ai_dj.regions import AudioRegion


class Track(BaseModel):
    track_id: str
    name: str
    channel_id: str
    clips: List[AudioClip] = Field(default_factory=list)
    regions: List[AudioRegion] = Field(default_factory=list)
    is_muted: bool = Field(default=False)
    is_solo: bool = Field(default=False)
    volume_db: float = Field(default=0.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        if not self.clips:
            return 0.0
        return max(c.end_timeline_position_sec for c in self.clips)

    def add_clip(self, clip: AudioClip) -> None:
        self.clips.append(clip)

    def get_clip(self, clip_id: str) -> AudioClip:
        for c in self.clips:
            if c.clip_id == clip_id:
                return c
        raise ResourceNotFoundError(resource_type="clip", resource_id=clip_id, suggested_tool="clip_create")

    def remove_clip(self, clip_id: str) -> bool:
        init_len = len(self.clips)
        self.clips = [c for c in self.clips if c.clip_id != clip_id]
        return len(self.clips) < init_len

    def render_track_audio(
        self,
        target_sample_rate: int = 44100,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> np.ndarray:
        if not self.clips or self.is_muted:
            total_dur = (end_sec - start_sec) if end_sec else 0.0
            total_samples = max(0, int(total_dur * target_sample_rate))
            return np.zeros((2, total_samples), dtype=np.float32)

        max_timeline_dur = end_sec or self.duration_sec
        render_duration = max(0.0, max_timeline_dur - start_sec)
        total_samples = int(render_duration * target_sample_rate)
        buffer = np.zeros((2, total_samples), dtype=np.float32)

        for clip in self.clips:
            if clip.is_muted:
                continue
            clip_start = clip.timeline_position_sec
            clip_end = clip.end_timeline_position_sec

            # Check overlap with [start_sec, max_timeline_dur]
            if clip_end <= start_sec or clip_start >= max_timeline_dur:
                continue

            clip_audio = clip.render_audio(target_sample_rate=target_sample_rate)
            clip_samples = clip_audio.shape[1]

            # Compute timeline buffer insertion bounds
            buf_start_sample = max(0, int((clip_start - start_sec) * target_sample_rate))
            buf_end_sample = min(total_samples, buf_start_sample + clip_samples)

            clip_offset = 0
            if clip_start < start_sec:
                clip_offset = int((start_sec - clip_start) * target_sample_rate)

            valid_samples = buf_end_sample - buf_start_sample
            if valid_samples > 0:
                audio_slice = clip_audio[:, clip_offset:clip_offset + valid_samples]
                # Ensure 2 channels
                if audio_slice.shape[0] == 1:
                    audio_slice = np.repeat(audio_slice, 2, axis=0)
                buffer[:, buf_start_sample:buf_end_sample] += audio_slice[:2, :]

        return buffer
