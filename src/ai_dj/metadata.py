import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import soundfile as sf
from ai_dj.engines.ffmpeg_engine import FFmpegEngine


def compute_file_hash(path: Path, max_bytes: int = 10 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    size = path.stat().st_size
    hasher.update(str(size).encode())
    with open(path, "rb") as f:
        chunk = f.read(max_bytes)
        hasher.update(chunk)
    return hasher.hexdigest()[:16]


class AudioMetadata(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    duration_seconds: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    format: str = "wav"
    file_size_bytes: int = 0
    file_hash: str = ""

    @classmethod
    def from_file(cls, path: Path) -> "AudioMetadata":
        file_hash = compute_file_hash(path)
        size = path.stat().st_size
        meta = cls(file_size_bytes=size, file_hash=file_hash)

        # Try soundfile first
        try:
            info = sf.info(str(path))
            meta.duration_seconds = float(info.duration)
            meta.sample_rate = int(info.samplerate)
            meta.channels = int(info.channels)
            meta.format = str(info.format).lower()
            subtype = str(info.subtype)
            if "16" in subtype:
                meta.bit_depth = 16
            elif "24" in subtype:
                meta.bit_depth = 24
            elif "32" in subtype or "FLOAT" in subtype:
                meta.bit_depth = 32
            return meta
        except Exception:
            pass

        # Fallback to FFmpeg
        ffmpeg = FFmpegEngine()
        if ffmpeg.is_available():
            info = ffmpeg.probe(path)
            meta.duration_seconds = float(info.get("duration_seconds", 0.0))
            meta.sample_rate = int(info.get("sample_rate", 44100))
            meta.channels = int(info.get("channels", 2))
            meta.format = str(info.get("format", path.suffix.strip(".")))
            tags = info.get("tags", {})
            meta.title = tags.get("title") or tags.get("TITLE")
            meta.artist = tags.get("artist") or tags.get("ARTIST")
            meta.album = tags.get("album") or tags.get("ALBUM")
            meta.genre = tags.get("genre") or tags.get("GENRE")
            date_str = tags.get("date") or tags.get("DATE")
            if date_str and date_str[:4].isdigit():
                meta.year = int(date_str[:4])

        return meta
