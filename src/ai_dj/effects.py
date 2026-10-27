from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from ai_dj.errors import DJError, ValidationError


class EffectConfig(BaseModel):
    effect_id: str
    effect_type: str  # eq, compressor, limiter, reverb, delay, distortion, filter, chorus
    is_bypassed: bool = False
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EffectChain:
    def __init__(self, effects: Optional[List[EffectConfig]] = None):
        self.effects: List[EffectConfig] = effects or []

    def add_effect(self, effect: EffectConfig) -> None:
        self.effects.append(effect)

    def remove_effect(self, effect_id: str) -> bool:
        initial = len(self.effects)
        self.effects = [e for e in self.effects if e.effect_id != effect_id]
        return len(self.effects) < initial

    def set_parameter(self, effect_id: str, param_name: str, value: Any) -> None:
        for eff in self.effects:
            if eff.effect_id == effect_id:
                eff.parameters[param_name] = value
                return
        raise DJError(
            code="EFFECT_NOT_FOUND",
            message=f"Effect '{effect_id}' not found in chain.",
        )

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0 or not self.effects:
            return audio

        # Convert to pedalboard plugins if available
        try:
            import pedalboard
            plugins = []
            for eff in self.effects:
                if eff.is_bypassed:
                    continue
                plugin = self._create_pedalboard_plugin(eff)
                if plugin:
                    plugins.append(plugin)

            if plugins:
                board = pedalboard.Pedalboard(plugins)
                return board(audio.astype(np.float32), sample_rate)
        except Exception:
            pass

        return audio

    def _create_pedalboard_plugin(self, eff: EffectConfig):
        import pedalboard
        t = eff.effect_type.lower()
        p = eff.parameters

        if t in ("equalizer", "eq"):
            # 3-band EQ: low_gain_db, mid_gain_db, high_gain_db
            low = float(p.get("low_gain_db", 0.0))
            mid = float(p.get("mid_gain_db", 0.0))
            high = float(p.get("high_gain_db", 0.0))
            return pedalboard.Pedalboard([
                pedalboard.LowShelfFilter(cutoff_frequency_hz=250.0, gain_db=low),
                pedalboard.PeakFilter(cutoff_frequency_hz=1000.0, gain_db=mid, q=1.0),
                pedalboard.HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=high),
            ])
        elif t == "compressor":
            return pedalboard.Compressor(
                threshold_db=float(p.get("threshold_db", -16.0)),
                ratio=float(p.get("ratio", 4.0)),
                attack_ms=float(p.get("attack_ms", 10.0)),
                release_ms=float(p.get("release_ms", 100.0)),
            )
        elif t == "limiter":
            ceiling = float(p.get("ceiling_db", p.get("threshold_db", -0.5)))
            return pedalboard.BrickwallLimiter(
                ceiling_db=ceiling,
                release_ms=float(p.get("release_ms", 50.0)),
                true_peak=True,
            )
        elif t == "reverb":
            return pedalboard.Reverb(
                room_size=float(p.get("room_size", 0.5)),
                damping=float(p.get("damping", 0.5)),
                wet_level=float(p.get("wet_level", 0.33)),
                dry_level=float(p.get("dry_level", 0.67)),
                width=float(p.get("width", 1.0)),
            )
        elif t == "delay":
            return pedalboard.Delay(
                delay_seconds=float(p.get("delay_seconds", 0.3)),
                feedback=float(p.get("feedback", 0.35)),
                mix=float(p.get("mix", 0.3)),
            )
        elif t == "highpass":
            return pedalboard.HighpassFilter(cutoff_frequency_hz=float(p.get("cutoff_hz", 80.0)))
        elif t == "lowpass":
            return pedalboard.LowpassFilter(cutoff_frequency_hz=float(p.get("cutoff_hz", 12000.0)))
        elif t == "distortion":
            return pedalboard.Distortion(drive_db=float(p.get("drive_db", 12.0)))
        elif t == "chorus":
            return pedalboard.Chorus(
                rate_hz=float(p.get("rate_hz", 1.0)),
                depth=float(p.get("depth", 0.25)),
                mix=float(p.get("mix", 0.5)),
            )
        return None
