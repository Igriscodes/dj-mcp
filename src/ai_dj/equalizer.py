from typing import Any, Dict
from pydantic import BaseModel, Field
from ai_dj.effects import EffectConfig


class ParametricEQ(BaseModel):
    low_gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    mid_gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    high_gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    low_freq_hz: float = Field(default=250.0, ge=20.0, le=1000.0)
    mid_freq_hz: float = Field(default=1000.0, ge=200.0, le=8000.0)
    high_freq_hz: float = Field(default=4000.0, ge=1000.0, le=20000.0)
    mid_q: float = Field(default=1.0, ge=0.1, le=10.0)

    def to_effect_config(self, effect_id: str = "eq") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="eq",
            parameters={
                "low_gain_db": self.low_gain_db,
                "mid_gain_db": self.mid_gain_db,
                "high_gain_db": self.high_gain_db,
                "low_freq_hz": self.low_freq_hz,
                "mid_freq_hz": self.mid_freq_hz,
                "high_freq_hz": self.high_freq_hz,
                "mid_q": self.mid_q,
            },
        )
