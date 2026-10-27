from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AudioRegion(BaseModel):
    region_id: str
    name: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    label: Optional[str] = None
    color: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)
