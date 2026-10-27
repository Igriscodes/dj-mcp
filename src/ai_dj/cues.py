from typing import Optional
from pydantic import BaseModel, Field


class CuePoint(BaseModel):
    cue_index: int = Field(ge=0, le=16)
    position_sec: float = Field(ge=0.0)
    label: Optional[str] = None
    color: Optional[str] = None
