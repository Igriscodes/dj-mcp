import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ai_dj.decks import Deck
from ai_dj.errors import DJError, ResourceNotFoundError
from ai_dj.mixer import Mixer, MixerChannel
from ai_dj.recovery import StateSnapshotManager, atomic_write_text
from ai_dj.stems import StemSet
from ai_dj.timeline import Timeline


class Project(BaseModel):
    project_id: str
    name: str
    version: str = "1.0.0"
    sample_rate: int = Field(default=44100)
    channels: int = Field(default=2)
    timeline: Timeline = Field(default_factory=Timeline)
    mixer: Mixer = Field(default_factory=Mixer)
    decks: Dict[str, Deck] = Field(default_factory=dict)
    stems: Dict[str, StemSet] = Field(default_factory=dict)

    @classmethod
    def create_new(
        cls,
        project_id: str,
        name: str,
        sample_rate: int = 44100,
        num_decks: int = 2,
    ) -> "Project":
        proj = cls(project_id=project_id, name=name, sample_rate=sample_rate)
        # Create default mixer channels and decks
        for i in range(num_decks):
            deck_letter = chr(ord("A") + i)
            deck_id = f"deck_{deck_letter.lower()}"
            ch_id = f"ch_{deck_letter.lower()}"

            proj.decks[deck_id] = Deck(deck_id=deck_id)
            proj.mixer.add_channel(MixerChannel(
                channel_id=ch_id,
                name=f"Deck {deck_letter}",
                crossfader_assign="a" if i == 0 else ("b" if i == 1 else "thru"),
            ))
        return proj

    def save_to_file(self, file_path: Path) -> Path:
        data = self.model_dump(mode="json")
        atomic_write_text(file_path, json.dumps(data, indent=2))
        return file_path

    @classmethod
    def load_from_file(cls, file_path: Path) -> "Project":
        if not file_path.exists():
            raise DJError(
                code="PROJECT_FILE_NOT_FOUND",
                message=f"Project file '{file_path}' does not exist.",
            )
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except Exception as e:
            raise DJError(
                code="PROJECT_LOAD_FAILED",
                message=f"Failed to load project from '{file_path}': {str(e)}",
            ) from e

    def validate_project(self) -> Dict[str, Any]:
        issues = []
        # Check source audio clips exist
        for track_id, track in self.timeline.tracks.items():
            for clip in track.clips:
                if not Path(clip.source_file).exists():
                    issues.append({
                        "type": "missing_audio_file",
                        "track_id": track_id,
                        "clip_id": clip.clip_id,
                        "file": str(clip.source_file),
                    })
        # Check loaded deck files exist
        for deck_id, deck in self.decks.items():
            if deck.loaded_audio_path and not Path(deck.loaded_audio_path).exists():
                issues.append({
                    "type": "missing_deck_audio",
                    "deck_id": deck_id,
                    "file": str(deck.loaded_audio_path),
                })

        return {
            "valid": len(issues) == 0,
            "project_id": self.project_id,
            "name": self.name,
            "issues": issues,
            "track_count": len(self.timeline.tracks),
            "deck_count": len(self.decks),
            "duration_seconds": self.timeline.total_duration_sec,
        }
