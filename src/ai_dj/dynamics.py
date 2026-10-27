from pydantic import BaseModel, Field
from ai_dj.effects import EffectConfig


class CompressorConfig(BaseModel):
    threshold_db: float = Field(default=-16.0, ge=-60.0, le=0.0)
    ratio: float = Field(default=4.0, ge=1.0, le=20.0)
    attack_ms: float = Field(default=10.0, ge=0.1, le=500.0)
    release_ms: float = Field(default=100.0, ge=5.0, le=2000.0)

    def to_effect_config(self, effect_id: str = "comp") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="compressor",
            parameters={
                "threshold_db": self.threshold_db,
                "ratio": self.ratio,
                "attack_ms": self.attack_ms,
                "release_ms": self.release_ms,
            },
        )


class LimiterConfig(BaseModel):
    threshold_db: float = Field(default=-0.5, ge=-12.0, le=0.0)
    release_ms: float = Field(default=50.0, ge=1.0, le=500.0)

    def to_effect_config(self, effect_id: str = "lim") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="limiter",
            parameters={
                "threshold_db": self.threshold_db,
                "release_ms": self.release_ms,
            },
        )
