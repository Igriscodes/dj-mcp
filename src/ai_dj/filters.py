from pydantic import BaseModel, Field
from ai_dj.effects import EffectConfig


class DJFilter(BaseModel):
    highpass_cutoff_hz: float = Field(default=20.0, ge=20.0, le=20000.0)
    lowpass_cutoff_hz: float = Field(default=20000.0, ge=20.0, le=20000.0)

    def to_highpass_config(self, effect_id: str = "hpf") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="highpass",
            parameters={"cutoff_hz": self.highpass_cutoff_hz},
        )

    def to_lowpass_config(self, effect_id: str = "lpf") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="lowpass",
            parameters={"cutoff_hz": self.lowpass_cutoff_hz},
        )
