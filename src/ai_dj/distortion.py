from pydantic import BaseModel, Field
from ai_dj.effects import EffectConfig


class DistortionConfig(BaseModel):
    drive_db: float = Field(default=12.0, ge=0.0, le=40.0)

    def to_effect_config(self, effect_id: str = "distortion") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="distortion",
            parameters={"drive_db": self.drive_db},
        )
