from pydantic import BaseModel, Field
from ai_dj.effects import EffectConfig


class DelayConfig(BaseModel):
    delay_seconds: float = Field(default=0.3, ge=0.01, le=2.0)
    feedback: float = Field(default=0.35, ge=0.0, le=0.95)
    mix: float = Field(default=0.3, ge=0.0, le=1.0)

    def to_effect_config(self, effect_id: str = "delay") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="delay",
            parameters={
                "delay_seconds": self.delay_seconds,
                "feedback": self.feedback,
                "mix": self.mix,
            },
        )
