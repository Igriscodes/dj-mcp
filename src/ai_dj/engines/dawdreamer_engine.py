import importlib.util
from typing import Any, Dict


class DawDreamerEngine:
    def __init__(self):
        self._is_avail = importlib.util.find_spec("dawdreamer") is not None

    @property
    def is_available(self) -> bool:
        return self._is_avail

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available,
            "engine": "dawdreamer",
            "purpose": "VST/AU host and DSP graph rendering engine",
        }
