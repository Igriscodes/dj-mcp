from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BusRoute(BaseModel):
    source_id: str
    target_bus_id: str
    send_gain_db: float = Field(default=0.0)
    is_pre_fader: bool = Field(default=False)


class AudioBus(BaseModel):
    bus_id: str
    name: str
    gain_db: float = Field(default=0.0)
    pan: float = Field(default=0.0)
    mute: bool = Field(default=False)
    solo: bool = Field(default=False)
    output_bus_id: Optional[str] = "master"


class RoutingMatrix(BaseModel):
    buses: Dict[str, AudioBus] = Field(default_factory=dict)
    sends: List[BusRoute] = Field(default_factory=list)

    def add_bus(self, bus: AudioBus) -> None:
        self.buses[bus.bus_id] = bus

    def add_send(self, send: BusRoute) -> None:
        self.sends.append(send)
