import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from ai_dj.effects import EffectChain, EffectConfig
from ai_dj.stereo import apply_panning, apply_phase_inversion, apply_stereo_width


def calculate_crossfader_gains(position: float, curve: str = "equal_power") -> Tuple[float, float]:
    # position is in [-1.0, 1.0]
    pos = max(-1.0, min(1.0, float(position)))
    # Normalized x from 0.0 (full left) to 1.0 (full right)
    x = (pos + 1.0) / 2.0

    if curve == "linear":
        gain_a = 1.0 - x
        gain_b = x
    elif curve == "cut" or curve == "scratch":
        # Sharp cut at the extreme edges
        gain_a = 1.0 if pos < 0.9 else max(0.0, (1.0 - pos) * 10.0)
        gain_b = 1.0 if pos > -0.9 else max(0.0, (pos + 1.0) * 10.0)
    elif curve == "smooth":
        # S-curve
        gain_a = 0.5 * (1.0 + math.cos(x * math.pi))
        gain_b = 0.5 * (1.0 - math.cos(x * math.pi))
    else:  # equal_power (default)
        gain_a = math.cos(x * (math.pi / 2.0))
        gain_b = math.sin(x * (math.pi / 2.0))

    return round(float(gain_a), 4), round(float(gain_b), 4)


class MixerChannel(BaseModel):
    channel_id: str
    name: str
    gain_db: float = Field(default=0.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    mute: bool = Field(default=False)
    solo: bool = Field(default=False)
    stereo_width: float = Field(default=1.0, ge=0.0, le=2.0)
    invert_phase_left: bool = Field(default=False)
    invert_phase_right: bool = Field(default=False)
    crossfader_assign: str = Field(default="thru")  # "a", "b", "thru"
    eq_low_gain_db: float = Field(default=0.0)
    eq_mid_gain_db: float = Field(default=0.0)
    eq_high_gain_db: float = Field(default=0.0)
    filter_hpf_hz: float = Field(default=20.0, ge=20.0, le=20000.0)
    filter_lpf_hz: float = Field(default=20000.0, ge=20.0, le=20000.0)

    def process_channel_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.mute or audio.size == 0:
            return np.zeros_like(audio)

        out = audio.copy()
        if out.ndim == 1:
            out = np.stack([out, out], axis=0)

        # Gain
        if abs(self.gain_db) > 1e-4:
            lin = math.pow(10.0, self.gain_db / 20.0)
            out *= lin

        # Pan
        if abs(self.pan) > 1e-4:
            out = apply_panning(out, self.pan)

        # Stereo width
        if abs(self.stereo_width - 1.0) > 1e-4:
            out = apply_stereo_width(out, self.stereo_width)

        # Phase inversion
        if self.invert_phase_left or self.invert_phase_right:
            out = apply_phase_inversion(out, self.invert_phase_left, self.invert_phase_right)

        # EQ & Filter DSP
        chain = EffectChain()
        if (
            abs(self.eq_low_gain_db) > 1e-4
            or abs(self.eq_mid_gain_db) > 1e-4
            or abs(self.eq_high_gain_db) > 1e-4
        ):
            chain.add_effect(EffectConfig(
                effect_id="ch_eq",
                effect_type="eq",
                parameters={
                    "low_gain_db": self.eq_low_gain_db,
                    "mid_gain_db": self.eq_mid_gain_db,
                    "high_gain_db": self.eq_high_gain_db,
                }
            ))

        if self.filter_hpf_hz > 25.0:
            chain.add_effect(EffectConfig(
                effect_id="ch_hpf",
                effect_type="highpass",
                parameters={"cutoff_hz": self.filter_hpf_hz}
            ))

        if self.filter_lpf_hz < 19000.0:
            chain.add_effect(EffectConfig(
                effect_id="ch_lpf",
                effect_type="lowpass",
                parameters={"cutoff_hz": self.filter_lpf_hz}
            ))

        return chain.process(out, sample_rate)


class Mixer(BaseModel):
    channels: Dict[str, MixerChannel] = Field(default_factory=dict)
    crossfader_position: float = Field(default=0.0, ge=-1.0, le=1.0)
    crossfader_curve: str = Field(default="equal_power")
    master_gain_db: float = Field(default=0.0)
    master_limiter_enabled: bool = Field(default=True)

    def add_channel(self, channel: MixerChannel) -> None:
        self.channels[channel.channel_id] = channel

    def mix_signals(self, channel_signals: Dict[str, np.ndarray], sample_rate: int) -> np.ndarray:
        if not channel_signals:
            return np.zeros((2, 0), dtype=np.float32)

        # Any solo active?
        has_solo = any(ch.solo for ch_id, ch in self.channels.items() if ch_id in channel_signals)
        gain_a, gain_b = calculate_crossfader_gains(self.crossfader_position, self.crossfader_curve)

        max_len = max(sig.shape[-1] for sig in channel_signals.values())
        master_sum = np.zeros((2, max_len), dtype=np.float32)

        for ch_id, raw_sig in channel_signals.items():
            ch = self.channels.get(ch_id)
            if not ch:
                continue
            if has_solo and not ch.solo:
                continue

            processed = ch.process_channel_audio(raw_sig, sample_rate)

            # Apply crossfader gain
            if ch.crossfader_assign == "a":
                processed *= gain_a
            elif ch.crossfader_assign == "b":
                processed *= gain_b

            # Pad if shorter than max_len
            if processed.shape[-1] < max_len:
                pad_len = max_len - processed.shape[-1]
                padded = np.pad(processed, ((0, 0), (0, pad_len)))
                master_sum += padded
            else:
                master_sum += processed[:, :max_len]

        # Master gain
        if abs(self.master_gain_db) > 1e-4:
            lin = math.pow(10.0, self.master_gain_db / 20.0)
            master_sum *= lin

        # Master limiter
        if self.master_limiter_enabled:
            chain = EffectChain([
                EffectConfig(effect_id="master_lim", effect_type="limiter", parameters={"threshold_db": -0.3})
            ])
            master_sum = chain.process(master_sum, sample_rate)

        return master_sum
