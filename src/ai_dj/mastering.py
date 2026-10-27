import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import soundfile as sf
from ai_dj.effects import EffectChain, EffectConfig
from ai_dj.errors import DJError, ValidationError
from ai_dj.loudness import apply_tpdf_dither, calculate_lufs_gain_db, measure_lufs
from ai_dj.stereo import apply_stereo_width, remove_dc_offset

MASTERING_PROFILES = {
    "streaming": {"target_lufs": -14.0, "true_peak_ceiling_db": -1.0},
    "dj_master": {"target_lufs": -9.0, "true_peak_ceiling_db": -0.3},
    "cd": {"target_lufs": -9.0, "true_peak_ceiling_db": -0.1},
    "club": {"target_lufs": -8.0, "true_peak_ceiling_db": -0.2},
    "dynamic": {"target_lufs": -18.0, "true_peak_ceiling_db": -1.0},
}


class MasteringProcessor:
    def __init__(
        self,
        profile: str = "dj_master",
        target_lufs: Optional[float] = None,
        true_peak_ceiling_db: Optional[float] = None,
        stereo_width: float = 1.05,
        apply_dither: bool = True,
        target_bit_depth: int = 24,
    ):
        prof = MASTERING_PROFILES.get(profile, MASTERING_PROFILES["dj_master"])
        self.profile = profile
        self.target_lufs = target_lufs if target_lufs is not None else prof["target_lufs"]
        self.true_peak_ceiling_db = (
            true_peak_ceiling_db if true_peak_ceiling_db is not None else prof["true_peak_ceiling_db"]
        )
        self.stereo_width = stereo_width
        self.apply_dither = apply_dither
        self.target_bit_depth = target_bit_depth

    def process_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        if audio.size == 0:
            return audio, {}

        # 1. Remove DC offset
        master = remove_dc_offset(audio)

        # 2. Gentle subsonic clean (30 Hz highpass) & air shelf
        chain = EffectChain([
            EffectConfig(effect_id="m_hpf", effect_type="highpass", parameters={"cutoff_hz": 30.0}),
            EffectConfig(effect_id="m_air", effect_type="eq", parameters={"high_gain_db": 0.8, "low_gain_db": 0.2}),
            EffectConfig(
                effect_id="m_glue",
                effect_type="compressor",
                parameters={"threshold_db": -14.0, "ratio": 1.5, "attack_ms": 30.0, "release_ms": 150.0},
            ),
        ])
        master = chain.process(master, sample_rate)

        # 3. Stereo enhancement
        if abs(self.stereo_width - 1.0) > 1e-4:
            master = apply_stereo_width(master, self.stereo_width)

        # 4. Measure pre-limiter LUFS and apply gain to reach target
        current_lufs = measure_lufs(master, sample_rate)
        gain_db = calculate_lufs_gain_db(current_lufs, self.target_lufs)
        # Limit makeup gain to avoid extreme distortion
        clamped_gain_db = max(-24.0, min(18.0, gain_db))
        lin_gain = math.pow(10.0, clamped_gain_db / 20.0)
        master = master * lin_gain

        # 5. Brickwall Limiter to enforce true peak ceiling
        lim_chain = EffectChain([
            EffectConfig(
                effect_id="m_brickwall",
                effect_type="limiter",
                parameters={"ceiling_db": self.true_peak_ceiling_db, "release_ms": 35.0},
            )
        ])
        master = lim_chain.process(master, sample_rate)

        # 6. Apply TPDF Dither
        if self.apply_dither:
            master = apply_tpdf_dither(master, bit_depth=self.target_bit_depth)

        # Enforce true peak ceiling guarantee
        ceiling_linear = math.pow(10.0, self.true_peak_ceiling_db / 20.0)
        peak_amp = float(np.max(np.abs(master)))
        if peak_amp > ceiling_linear:
            master = master * (ceiling_linear / (peak_amp + 1e-9))

        # 7. Final measurements
        final_lufs = measure_lufs(master, sample_rate)
        final_peak_db = round(float(20.0 * math.log10(np.max(np.abs(master)) + 1e-9)), 2)

        stats = {
            "profile": self.profile,
            "target_lufs": self.target_lufs,
            "initial_lufs": current_lufs,
            "final_lufs": final_lufs,
            "makeup_gain_applied_db": round(clamped_gain_db, 2),
            "true_peak_ceiling_db": self.true_peak_ceiling_db,
            "final_peak_dbfs": final_peak_db,
            "dither_bit_depth": self.target_bit_depth if self.apply_dither else None,
            "is_within_tolerance": abs(final_lufs - self.target_lufs) <= 1.0,
        }

        return master, stats
