from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ai_dj.analysis import AudioAnalyzer
from ai_dj.config import DJConfig, get_default_config
from ai_dj.errors import DJError, ResourceNotFoundError
from ai_dj.metadata import AudioMetadata


class LibraryTrack(BaseModel):
    track_id: str
    file_path: Path
    metadata: AudioMetadata
    analysis: Optional[Dict[str, Any]] = None


class AudioLibrary:
    def __init__(self, config: Optional[DJConfig] = None):
        self.config = config or get_default_config()
        self.tracks: Dict[str, LibraryTrack] = {}
        self.analyzer = AudioAnalyzer(self.config)

    def scan_directory(self, directory_path: Path, recursive: bool = True) -> List[Dict[str, Any]]:
        if not directory_path.exists():
            raise DJError(
                code="DIRECTORY_NOT_FOUND",
                message=f"Directory '{directory_path}' does not exist.",
            )

        supported_exts = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}
        pattern = "**/*" if recursive else "*"
        found = []

        for p in directory_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in supported_exts:
                track = self.import_track(p)
                found.append(track.model_dump())

        return found

    def import_track(self, file_path: Path, track_id: Optional[str] = None) -> LibraryTrack:
        if not file_path.exists():
            raise DJError(
                code="FILE_NOT_FOUND",
                message=f"Audio file '{file_path}' does not exist.",
            )

        tid = track_id or file_path.stem
        meta = AudioMetadata.from_file(file_path)
        lib_track = LibraryTrack(track_id=tid, file_path=file_path.resolve(), metadata=meta)
        self.tracks[tid] = lib_track
        return lib_track

    def get_track(self, track_id: str) -> LibraryTrack:
        if track_id not in self.tracks:
            raise ResourceNotFoundError(
                resource_type="track",
                resource_id=track_id,
                suggested_tool="library_import",
                suggested_arguments={"file_path": f"{track_id}.wav"},
            )
        return self.tracks[track_id]

    def analyze_track(self, track_id: str, force: bool = False) -> Dict[str, Any]:
        track = self.get_track(track_id)
        analysis_data = self.analyzer.analyze_track(track.file_path, force_recompute=force)
        track.analysis = analysis_data
        return analysis_data

    def search(
        self,
        query: Optional[str] = None,
        min_bpm: Optional[float] = None,
        max_bpm: Optional[float] = None,
        key: Optional[str] = None,
    ) -> List[LibraryTrack]:
        results = list(self.tracks.values())
        if query:
            q = query.lower()
            results = [
                t for t in results
                if q in t.track_id.lower()
                or (t.metadata.title and q in t.metadata.title.lower())
                or (t.metadata.artist and q in t.metadata.artist.lower())
            ]

        if min_bpm is not None or max_bpm is not None:
            filtered = []
            for t in results:
                bpm = t.analysis.get("bpm") if t.analysis else None
                if bpm is not None:
                    if min_bpm is not None and bpm < min_bpm:
                        continue
                    if max_bpm is not None and bpm > max_bpm:
                        continue
                    filtered.append(t)
            results = filtered

        if key:
            results = [
                t for t in results
                if t.analysis and (t.analysis.get("key") == key or t.analysis.get("camelot") == key.upper())
            ]

        return results
