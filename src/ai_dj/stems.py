from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf
from ai_dj.engines.separation_engine import SeparationEngine
from ai_dj.errors import DJError, ResourceNotFoundError


class StemTrack(BaseModel):
    stem_name: str  # "vocals", "drums", "bass", "other"
    file_path: Path
    gain_db: float = Field(default=0.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    mute: bool = Field(default=False)
    solo: bool = Field(default=False)
    time_offset_sec: float = Field(default=0.0)


class StemSet(BaseModel):
    set_id: str
    source_audio_path: Path
    stems: Dict[str, StemTrack] = Field(default_factory=dict)

    def mute_stem(self, stem_name: str, mute: bool = True) -> None:
        if stem_name not in self.stems:
            raise ResourceNotFoundError("stem", stem_name)
        self.stems[stem_name].mute = mute

    def solo_stem(self, stem_name: str, solo: bool = True) -> None:
        if stem_name not in self.stems:
            raise ResourceNotFoundError("stem", stem_name)
        self.stems[stem_name].solo = solo

    def set_stem_gain(self, stem_name: str, gain_db: float) -> None:
        if stem_name not in self.stems:
            raise ResourceNotFoundError("stem", stem_name)
        self.stems[stem_name].gain_db = float(gain_db)

    def recombine(self, target_sample_rate: int = 44100) -> np.ndarray:
        if not self.stems:
            return np.zeros((2, 0), dtype=np.float32)

        has_solo = any(s.solo for s in self.stems.values())
        stem_signals = []
        max_len = 0

        for name, stem in self.stems.items():
            if stem.mute or (has_solo and not stem.solo):
                continue
            if not stem.file_path.exists():
                continue

            data, sr = sf.read(str(stem.file_path), always_2d=True, dtype="float32")
            sig = data.T
            if sr != target_sample_rate:
                import scipy.signal as signal
                sig = signal.resample_poly(sig, target_sample_rate, sr, axis=1)

            # Apply gain
            if abs(stem.gain_db) > 1e-4:
                import math
                lin = math.pow(10.0, stem.gain_db / 20.0)
                sig = sig * lin

            max_len = max(max_len, sig.shape[1])
            stem_signals.append(sig)

        if not stem_signals:
            return np.zeros((2, 0), dtype=np.float32)

        mix = np.zeros((2, max_len), dtype=np.float32)
        for s in stem_signals:
            mix[:, :s.shape[1]] += s[:2, :]

        return mix
