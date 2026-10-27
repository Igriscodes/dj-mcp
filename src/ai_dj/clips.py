import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf
from ai_dj.engines.rubberband_engine import RubberBandEngine
from ai_dj.errors import DJError, ValidationError


class AudioClip(BaseModel):
    clip_id: str
    name: str
    source_file: Path
    source_offset_sec: float = Field(default=0.0, ge=0.0)
    timeline_position_sec: float = Field(default=0.0, ge=0.0)
    duration_sec: float = Field(gt=0.0)
    gain_db: float = Field(default=0.0)
    fade_in_sec: float = Field(default=0.0, ge=0.0)
    fade_out_sec: float = Field(default=0.0, ge=0.0)
    fade_curve: str = Field(default="linear")
    reverse: bool = Field(default=False)
    pitch_semitones: float = Field(default=0.0)
    time_stretch_rate: float = Field(default=1.0, gt=0.0)
    is_muted: bool = Field(default=False)

    @property
    def end_timeline_position_sec(self) -> float:
        return self.timeline_position_sec + self.duration_sec

    def trim(self, start_delta_sec: float = 0.0, end_delta_sec: float = 0.0) -> "AudioClip":
        new_offset = max(0.0, self.source_offset_sec + start_delta_sec)
        new_duration = self.duration_sec - start_delta_sec - end_delta_sec
        if new_duration <= 0.001:
            raise ValidationError("duration", "Trim resulted in zero or negative duration.")
        self.source_offset_sec = round(new_offset, 4)
        self.duration_sec = round(new_duration, 4)
        return self

    def split(self, split_offset_sec: float, new_clip_id: str) -> Tuple["AudioClip", "AudioClip"]:
        if not (0.001 < split_offset_sec < (self.duration_sec - 0.001)):
            raise ValidationError("split_offset_sec", f"Split offset {split_offset_sec}s must be strictly inside clip duration [0, {self.duration_sec}]s")

        first = self.model_copy(deep=True)
        first.duration_sec = round(split_offset_sec, 4)

        second = self.model_copy(deep=True)
        second.clip_id = new_clip_id
        second.name = f"{self.name}_split"
        second.source_offset_sec = round(self.source_offset_sec + split_offset_sec, 4)
        second.timeline_position_sec = round(self.timeline_position_sec + split_offset_sec, 4)
        second.duration_sec = round(self.duration_sec - split_offset_sec, 4)

        return first, second

    def render_audio(self, target_sample_rate: int = 44100) -> np.ndarray:
        if not self.source_file.exists():
            raise DJError(
                code="SOURCE_FILE_NOT_FOUND",
                message=f"Source audio file '{self.source_file}' does not exist.",
            )

        info = sf.info(str(self.source_file))
        sr = info.samplerate
        start_frame = int(self.source_offset_sec * sr)
        # Account for time stretch rate in frames to read
        frames_to_read = int(self.duration_sec * self.time_stretch_rate * sr)

        data, read_sr = sf.read(
            str(self.source_file),
            start=start_frame,
            stop=start_frame + frames_to_read,
            always_2d=True,
            dtype="float32",
        )
        # data is (samples, channels) -> transpose to (channels, samples)
        audio = data.T

        # Reverse if requested
        if self.reverse:
            audio = np.flip(audio, axis=1)

        # Time stretch / pitch shift if needed
        rb = RubberBandEngine()
        if abs(self.time_stretch_rate - 1.0) > 1e-4:
            audio = rb.time_stretch(audio, read_sr, self.time_stretch_rate)

        if abs(self.pitch_semitones) > 1e-4:
            audio = rb.pitch_shift(audio, read_sr, self.pitch_semitones)

        # Resample to target_sample_rate if needed
        if read_sr != target_sample_rate:
            import scipy.signal as signal
            gcd = math.gcd(read_sr, target_sample_rate)
            up = target_sample_rate // gcd
            down = read_sr // gcd
            resampled = signal.resample_poly(audio, up, down, axis=1)
            audio = np.ascontiguousarray(resampled, dtype=np.float32)

        # Truncate or pad to exact target duration
        target_frames = int(self.duration_sec * target_sample_rate)
        if audio.shape[1] > target_frames:
            audio = audio[:, :target_frames]
        elif audio.shape[1] < target_frames:
            pad = np.zeros((audio.shape[0], target_frames - audio.shape[1]), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=1)

        # Apply gain
        if abs(self.gain_db) > 1e-4:
            linear_gain = math.pow(10.0, self.gain_db / 20.0)
            audio = audio * linear_gain

        # Apply fades
        if self.fade_in_sec > 0.0:
            fade_in_frames = min(int(self.fade_in_sec * target_sample_rate), audio.shape[1])
            fade_curve = np.linspace(0.0, 1.0, fade_in_frames, dtype=np.float32)
            audio[:, :fade_in_frames] *= fade_curve

        if self.fade_out_sec > 0.0:
            fade_out_frames = min(int(self.fade_out_sec * target_sample_rate), audio.shape[1])
            fade_curve = np.linspace(1.0, 0.0, fade_out_frames, dtype=np.float32)
            audio[:, -fade_out_frames:] *= fade_curve

        return audio
