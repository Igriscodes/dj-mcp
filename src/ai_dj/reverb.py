from pydantic import BaseModel, Field
from ai_dj.effects import EffectConfig


class ReverbConfig(BaseModel):
    room_size: float = Field(default=0.5, ge=0.0, le=1.0)
    damping: float = Field(default=0.5, ge=0.0, le=1.0)
    wet_level: float = Field(default=0.33, ge=0.0, le=1.0)
    dry_level: float = Field(default=0.67, ge=0.0, le=1.0)
    width: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_effect_config(self, effect_id: str = "verb") -> EffectConfig:
        return EffectConfig(
            effect_id=effect_id,
            effect_type="reverb",
            parameters={
                "room_size": self.room_size,
                "damping": self.damping,
                "wet_level": self.wet_level,
                "dry_level": self.dry_level,
                "width": self.width,
            },
        )
