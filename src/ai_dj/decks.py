import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field
from ai_dj.beatgrid import BeatGrid
from ai_dj.cues import CuePoint
from ai_dj.engines.rubberband_engine import RubberBandEngine
from ai_dj.errors import DJError, PrerequisiteError, ValidationError
from ai_dj.loops import Loop
from ai_dj.tempo import calculate_pitch_fader_percent, calculate_tempo_ratio


class Deck(BaseModel):
    deck_id: str
    loaded_track_id: Optional[str] = None
    loaded_audio_path: Optional[Path] = None
    state: str = Field(default="stopped")  # "playing", "paused", "stopped"
    position_sec: float = Field(default=0.0, ge=0.0)
    original_bpm: float = Field(default=120.0, gt=0.0)
    current_bpm: float = Field(default=120.0, gt=0.0)
    pitch_percent: float = Field(default=0.0)
    pitch_semitones: float = Field(default=0.0)
    key_lock: bool = Field(default=True)
    sync_mode: str = Field(default="off")  # "off", "tempo", "beat"
    quantize_enabled: bool = Field(default=True)
    cues: Dict[int, CuePoint] = Field(default_factory=dict)
    active_loop: Optional[Loop] = None
    beatgrid: Optional[BeatGrid] = None

    def require_loaded(self) -> None:
        if not self.loaded_track_id:
            raise PrerequisiteError(
                code="NO_TRACK_LOADED",
                message=f"Deck '{self.deck_id}' has no track loaded.",
                required_action="deck_load",
                suggested_tool="deck_load",
                suggested_arguments={"deck_id": self.deck_id, "track_id": "example_track"},
            )

    def load_track(
        self,
        track_id: str,
        audio_path: Path,
        bpm: float = 120.0,
        beatgrid: Optional[BeatGrid] = None,
    ) -> None:
        self.loaded_track_id = track_id
        self.loaded_audio_path = audio_path
        self.original_bpm = bpm
        self.current_bpm = bpm
        self.pitch_percent = 0.0
        self.pitch_semitones = 0.0
        self.position_sec = 0.0
        self.state = "stopped"
        self.beatgrid = beatgrid
        self.cues.clear()
        self.active_loop = None

    def unload(self) -> None:
        self.loaded_track_id = None
        self.loaded_audio_path = None
        self.state = "stopped"
        self.position_sec = 0.0
        self.cues.clear()
        self.active_loop = None
        self.beatgrid = None

    def play(self) -> None:
        self.require_loaded()
        self.state = "playing"

    def pause(self) -> None:
        self.require_loaded()
        self.state = "paused"

    def stop(self) -> None:
        self.require_loaded()
        self.state = "stopped"
        self.position_sec = 0.0

    def seek(self, target_sec: float, quantize: bool = False) -> float:
        self.require_loaded()
        pos = max(0.0, target_sec)
        if (quantize or self.quantize_enabled) and self.beatgrid:
            pos = self.beatgrid.quantize_position(pos, quantum="beat")
        self.position_sec = round(pos, 4)
        return self.position_sec

    def set_bpm(self, new_bpm: float) -> None:
        self.require_loaded()
        if new_bpm <= 0.0:
            raise ValidationError("bpm", "BPM must be strictly positive")
        self.current_bpm = float(new_bpm)
        self.pitch_percent = calculate_pitch_fader_percent(self.original_bpm, new_bpm)

    def set_pitch_percent(self, pitch_percent: float) -> None:
        self.require_loaded()
        self.pitch_percent = float(pitch_percent)
        self.current_bpm = self.original_bpm * (1.0 + (pitch_percent / 100.0))

    def set_key_shift(self, semitones: float) -> None:
        self.require_loaded()
        self.pitch_semitones = float(semitones)

    def sync_to(self, target_bpm: float, target_beat_phase_sec: Optional[float] = None) -> None:
        self.require_loaded()
        self.set_bpm(target_bpm)
        self.sync_mode = "beat" if target_beat_phase_sec is not None else "tempo"
        if target_beat_phase_sec is not None and self.beatgrid:
            # Beat-phase alignment
            nearest = self.beatgrid.get_nearest_beat(self.position_sec)
            self.position_sec = round(nearest, 4)

    def set_cue(self, cue_idx: int, position_sec: Optional[float] = None) -> CuePoint:
        self.require_loaded()
        pos = self.position_sec if position_sec is None else position_sec
        if self.quantize_enabled and self.beatgrid:
            pos = self.beatgrid.quantize_position(pos, quantum="beat")
        cue = CuePoint(cue_index=cue_idx, position_sec=round(pos, 4))
        self.cues[cue_idx] = cue
        return cue

    def jump_to_cue(self, cue_idx: int) -> float:
        self.require_loaded()
        cue = self.cues.get(cue_idx)
        if not cue:
            raise DJError(
                code="CUE_NOT_FOUND",
                message=f"Hot cue {cue_idx} is not set on Deck '{self.deck_id}'.",
                suggested_tool="cue_create",
                suggested_arguments={"deck_id": self.deck_id, "cue_index": cue_idx},
            )
        self.position_sec = cue.position_sec
        return self.position_sec

    def set_loop(self, length_beats: float = 4.0, start_sec: Optional[float] = None) -> Loop:
        self.require_loaded()
        start = self.position_sec if start_sec is None else start_sec
        if self.quantize_enabled and self.beatgrid:
            start = self.beatgrid.quantize_position(start, quantum="beat")
            interval = self.beatgrid.beat_interval_sec
        else:
            interval = 60.0 / self.current_bpm

        end = start + (length_beats * interval)
        loop = Loop(start_sec=round(start, 4), end_sec=round(end, 4), length_beats=length_beats, is_active=True)
        self.active_loop = loop
        return loop

    def clear_loop(self) -> None:
        self.active_loop = None

    def render_deck_audio(
        self,
        duration_sec: float,
        target_sample_rate: int = 44100,
    ) -> np.ndarray:
        target_frames = int(round(duration_sec * target_sample_rate))
        if target_frames <= 0 or not self.loaded_audio_path or not self.loaded_audio_path.exists():
            return np.zeros((2, max(0, target_frames)), dtype=np.float32)

        try:
            raw_data, file_sr = sf.read(str(self.loaded_audio_path), always_2d=True, dtype="float32")
        except Exception as e:
            raise DJError(
                code="DECK_AUDIO_READ_ERROR",
                message=f"Failed to read audio from '{self.loaded_audio_path}' for deck '{self.deck_id}': {str(e)}",
            ) from e

        source_audio = raw_data.T
        if source_audio.shape[0] == 1:
            source_audio = np.repeat(source_audio, 2, axis=0)
        elif source_audio.shape[0] > 2:
            source_audio = source_audio[:2, :]

        sr = target_sample_rate
        if file_sr != sr:
            import scipy.signal as signal
            gcd = math.gcd(file_sr, sr)
            up = sr // gcd
            down = file_sr // gcd
            source_audio = signal.resample_poly(source_audio, up, down, axis=1).astype(np.float32)

        rate = 1.0
        if self.original_bpm > 0 and self.current_bpm > 0:
            rate = float(self.current_bpm / self.original_bpm)

        pitch_semitones = float(self.pitch_semitones)
        if not self.key_lock and abs(rate - 1.0) > 1e-4 and pitch_semitones == 0.0:
            pitch_semitones = 12.0 * math.log2(rate)

        source_needed_frames = int(math.ceil(duration_sec * rate * sr)) if rate > 0 else target_frames
        source_needed_frames = max(1, source_needed_frames)

        num_source_frames = source_audio.shape[1]
        start_frame = max(0, min(int(self.position_sec * sr), num_source_frames))

        source_buf = np.zeros((2, source_needed_frames), dtype=np.float32)

        loop_active = (self.active_loop is not None and self.active_loop.is_active)
        if loop_active and num_source_frames > 0:
            l_start = max(0, min(int(self.active_loop.start_sec * sr), num_source_frames - 1))
            l_end = max(l_start + 1, min(int(self.active_loop.end_sec * sr), num_source_frames))
            loop_len = max(1, l_end - l_start)
            loop_slice = source_audio[:, l_start:l_end]

            cur_idx = 0
            if start_frame < l_start:
                pre_len = min(l_start - start_frame, source_needed_frames)
                source_buf[:, :pre_len] = source_audio[:, start_frame:start_frame + pre_len]
                cur_idx = pre_len
                cur_loop_offset = 0
            else:
                cur_loop_offset = (start_frame - l_start) % loop_len

            while cur_idx < source_needed_frames:
                chunk_len = min(loop_len - cur_loop_offset, source_needed_frames - cur_idx)
                source_buf[:, cur_idx:cur_idx + chunk_len] = loop_slice[:, cur_loop_offset:cur_loop_offset + chunk_len]
                cur_idx += chunk_len
                cur_loop_offset = 0
        else:
            avail = max(0, num_source_frames - start_frame)
            take = min(avail, source_needed_frames)
            if take > 0:
                source_buf[:, :take] = source_audio[:, start_frame:start_frame + take]

        rb = RubberBandEngine()
        out_audio = source_buf
        if abs(rate - 1.0) > 1e-4:
            out_audio = rb.time_stretch(out_audio, sr, rate)

        if abs(pitch_semitones) > 1e-4:
            out_audio = rb.pitch_shift(out_audio, sr, pitch_semitones)

        if out_audio.shape[1] > target_frames:
            out_audio = out_audio[:, :target_frames]
        elif out_audio.shape[1] < target_frames:
            pad = np.zeros((2, target_frames - out_audio.shape[1]), dtype=np.float32)
            out_audio = np.concatenate([out_audio, pad], axis=1)

        return out_audio.astype(np.float32)
