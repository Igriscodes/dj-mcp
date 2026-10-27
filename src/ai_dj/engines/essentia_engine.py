import importlib.util
from typing import Any, Dict


class EssentiaEngine:
    def __init__(self):
        self._is_avail = importlib.util.find_spec("essentia") is not None

    @property
    def is_available(self) -> bool:
        return self._is_avail

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available,
            "engine": "essentia",
            "purpose": "Audio analysis and audio feature extraction",
        }
