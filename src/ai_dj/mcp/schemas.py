from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Project
class ProjectCreateInput(BaseModel):
    project_id: str = Field(description="Unique project identifier (e.g. 'summer_mix_2026')")
    name: str = Field(description="Human readable project name")
    sample_rate: int = Field(default=44100, description="Audio sample rate (44100 or 48000 Hz)")
    num_decks: int = Field(default=2, ge=2, le=4, description="Number of DJ decks (2 or 4)")


class ProjectOpenInput(BaseModel):
    file_path: str = Field(description="Path to .djproj project file")


class ProjectSaveInput(BaseModel):
    file_path: Optional[str] = Field(default=None, description="Optional destination path for project file")


class ProjectSnapshotInput(BaseModel):
    label: Optional[str] = Field(default=None, description="Optional label for snapshot")


class ProjectRestoreInput(BaseModel):
    snapshot_id: str = Field(description="Snapshot ID to restore")


# Library
class LibraryScanInput(BaseModel):
    directory: str = Field(description="Directory to scan for audio files")
    recursive: bool = Field(default=True, description="Whether to scan subdirectories")


class LibrarySearchInput(BaseModel):
    query: Optional[str] = Field(default=None, description="Text query to search title/artist/filename")
    min_bpm: Optional[float] = Field(default=None, ge=40.0, le=250.0, description="Minimum BPM")
    max_bpm: Optional[float] = Field(default=None, ge=40.0, le=250.0, description="Maximum BPM")
    key: Optional[str] = Field(default=None, description="Key or Camelot notation (e.g. '8A' or 'A minor')")


class LibraryImportInput(BaseModel):
    file_path: str = Field(description="Path to audio file to import")
    track_id: Optional[str] = Field(default=None, description="Optional custom track ID")


# Tracks
class TrackAnalyzeInput(BaseModel):
    track_id: str = Field(description="ID of track to analyze")
    force: bool = Field(default=False, description="Force recomputing analysis even if cached")


class TrackWaveformInput(BaseModel):
    track_id: str = Field(description="ID of track")
    num_points: int = Field(default=256, ge=32, le=2048, description="Number of waveform peak bins")


# Decks
class DeckCreateInput(BaseModel):
    deck_id: str = Field(description="Unique deck identifier (e.g. 'deck_c')")


class DeckLoadInput(BaseModel):
    deck_id: str = Field(description="Deck identifier (e.g. 'deck_a')")
    track_id: str = Field(description="Track ID to load onto deck")


class DeckSeekInput(BaseModel):
    deck_id: str = Field(description="Deck identifier")
    position_sec: float = Field(ge=0.0, description="Target playback position in seconds")
    quantize: bool = Field(default=True, description="Snap target position to nearest beat")


class DeckSyncInput(BaseModel):
    deck_id: str = Field(description="Deck to synchronize")
    master_deck_id: str = Field(description="Master deck to sync tempo and phase with")


class DeckSetBpmInput(BaseModel):
    deck_id: str = Field(description="Deck identifier")
    bpm: float = Field(gt=20.0, lt=300.0, description="Target BPM")


class DeckSetPitchInput(BaseModel):
    deck_id: str = Field(description="Deck identifier")
    pitch_percent: float = Field(ge=-100.0, le=100.0, description="Pitch fader percentage shift")


class DeckSetKeyInput(BaseModel):
    deck_id: str = Field(description="Deck identifier")
    semitones: float = Field(ge=-24.0, le=24.0, description="Key pitch shift in semitones")


# Beatgrid
class DetectBpmInput(BaseModel):
    file_path: str = Field(description="Path to audio file")


class CreateBeatgridInput(BaseModel):
    track_id: str = Field(description="Track ID")
    bpm: float = Field(gt=20.0, lt=300.0, description="Track BPM")
    first_beat_sec: float = Field(default=0.0, ge=0.0, description="Timestamp of first downbeat")


class CorrectBeatgridInput(BaseModel):
    track_id: str = Field(description="Track ID")
    offset_sec: float = Field(description="Time shift in seconds to nudge beatgrid (+/-)")
    bpm_override: Optional[float] = Field(default=None, description="Optional corrected BPM")


class QuantizeToGridInput(BaseModel):
    track_id: str = Field(description="Track ID")
    position_sec: float = Field(ge=0.0, description="Timestamp in seconds")
    quantum: str = Field(default="beat", description="Quantum unit: 'bar', 'beat', 'half', 'quarter', 'eighth'")


# Editing
class ClipCreateInput(BaseModel):
    track_id: str = Field(description="Target timeline track ID")
    clip_id: str = Field(description="Unique clip ID")
    source_file: str = Field(description="Path to source audio file")
    timeline_position_sec: float = Field(default=0.0, ge=0.0, description="Timeline start position")
    source_offset_sec: float = Field(default=0.0, ge=0.0, description="Offset within source file")
    duration_sec: float = Field(gt=0.0, description="Clip duration in seconds")


class ClipTrimInput(BaseModel):
    track_id: str = Field(description="Timeline track ID")
    clip_id: str = Field(description="Clip ID to trim")
    start_delta_sec: float = Field(default=0.0, ge=0.0, description="Seconds to trim from start")
    end_delta_sec: float = Field(default=0.0, ge=0.0, description="Seconds to trim from end")


class ClipSplitInput(BaseModel):
    track_id: str = Field(description="Timeline track ID")
    clip_id: str = Field(description="Clip ID to split")
    split_offset_sec: float = Field(gt=0.0, description="Offset inside clip where split occurs")
    new_clip_id: str = Field(description="New clip ID for second half")


class ClipFadeInput(BaseModel):
    track_id: str = Field(description="Timeline track ID")
    clip_id: str = Field(description="Clip ID")
    fade_in_sec: float = Field(default=0.0, ge=0.0, description="Fade in duration in seconds")
    fade_out_sec: float = Field(default=0.0, ge=0.0, description="Fade out duration in seconds")


class ClipGainInput(BaseModel):
    track_id: str = Field(description="Timeline track ID")
    clip_id: str = Field(description="Clip ID")
    gain_db: float = Field(ge=-60.0, le=24.0, description="Gain in dB")


class ClipTimeStretchInput(BaseModel):
    track_id: str = Field(description="Timeline track ID")
    clip_id: str = Field(description="Clip ID")
    rate: float = Field(gt=0.25, le=4.0, description="Speed playback rate (e.g. 1.25x speed)")


class ClipPitchInput(BaseModel):
    track_id: str = Field(description="Timeline track ID")
    clip_id: str = Field(description="Clip ID")
    semitones: float = Field(ge=-24.0, le=24.0, description="Semitones shift")


# Loops and Cues
class CueCreateInput(BaseModel):
    deck_id: str = Field(description="Deck identifier")
    cue_index: int = Field(ge=0, le=16, description="Cue number (0 to 16)")
    position_sec: Optional[float] = Field(default=None, description="Optional timestamp; defaults to current deck position")


class LoopCreateInput(BaseModel):
    deck_id: str = Field(description="Deck identifier")
    length_beats: float = Field(default=4.0, gt=0.0, description="Loop length in beats (e.g. 1, 2, 4, 8, 16, 32)")
    start_sec: Optional[float] = Field(default=None, description="Optional loop start time")


# Mixer
class MixerGainInput(BaseModel):
    channel_id: str = Field(description="Channel ID (e.g. 'ch_a')")
    gain_db: float = Field(ge=-60.0, le=24.0, description="Channel volume gain in dB")


class MixerPanInput(BaseModel):
    channel_id: str = Field(description="Channel ID")
    pan: float = Field(ge=-1.0, le=1.0, description="Pan position (-1.0=Left, 0.0=Center, 1.0=Right)")


class MixerCrossfadeInput(BaseModel):
    position: float = Field(ge=-1.0, le=1.0, description="Crossfader position (-1.0=Deck A, 0.0=Center, 1.0=Deck B)")
    curve: str = Field(default="equal_power", description="Curve type: 'equal_power', 'linear', 'cut', 'smooth'")


class MixerRouteInput(BaseModel):
    channel_id: str = Field(description="Channel ID")
    crossfader_assign: str = Field(description="Crossfader assignment: 'a', 'b', or 'thru'")


# Effects
class EffectAddInput(BaseModel):
    channel_id: str = Field(description="Channel ID")
    effect_id: str = Field(description="Unique effect ID")
    effect_type: str = Field(description="Type: 'eq', 'compressor', 'limiter', 'reverb', 'delay', 'distortion', 'highpass', 'lowpass'")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Effect parameters")


class EffectSetParameterInput(BaseModel):
    channel_id: str = Field(description="Channel ID")
    effect_id: str = Field(description="Effect ID")
    parameter_name: str = Field(description="Parameter name")
    value: Any = Field(description="Parameter value")


# Stems
class SeparateStemsInput(BaseModel):
    audio_path: str = Field(description="Path to audio file to separate")
    output_dir: Optional[str] = Field(default=None, description="Destination directory for stems")
    model_name: str = Field(default="htdemucs", description="Demucs model name (htdemucs, htdemucs_ft)")


class StemControlInput(BaseModel):
    set_id: str = Field(description="Stem set identifier")
    stem_name: str = Field(description="Stem name: 'vocals', 'drums', 'bass', 'other'")
    mute: Optional[bool] = Field(default=None, description="Mute state")
    solo: Optional[bool] = Field(default=None, description="Solo state")
    gain_db: Optional[float] = Field(default=None, description="Gain in dB")


# Mastering
class MasterProcessInput(BaseModel):
    input_file: str = Field(description="Path to unmastered audio file")
    output_file: str = Field(description="Destination path for mastered audio file")
    profile: str = Field(default="dj_master", description="Mastering target: 'streaming', 'dj_master', 'cd', 'club', 'dynamic'")
    target_lufs: Optional[float] = Field(default=None, description="Explicit target LUFS override")
    true_peak_ceiling_db: Optional[float] = Field(default=None, description="True peak ceiling in dBTP")


# Rendering
class RenderProjectInput(BaseModel):
    output_path: str = Field(description="Target output audio file path (WAV, FLAC, MP3, etc.)")
    start_sec: float = Field(default=0.0, ge=0.0, description="Start render timestamp")
    end_sec: Optional[float] = Field(default=None, description="End render timestamp (None = full project)")
    format: Optional[str] = Field(default=None, description="Audio format: 'wav', 'flac', 'mp3', 'ogg'")
    bit_depth: int = Field(default=24, description="Bit depth: 16, 24, 32")


class RenderPreviewInput(BaseModel):
    output_path: str = Field(description="Target output audio file path for preview snippet")
    start_sec: float = Field(default=0.0, ge=0.0, description="Preview start timestamp")
    duration_sec: float = Field(default=15.0, ge=1.0, le=60.0, description="Preview duration in seconds")


# Workflows
class WorkflowCreateInput(BaseModel):
    workflow_id: str = Field(description="Unique workflow ID")
    name: str = Field(description="Workflow name")


class WorkflowAddStepInput(BaseModel):
    workflow_id: str = Field(description="Workflow ID")
    tool_name: str = Field(description="Name of tool to execute")
    arguments: Dict[str, Any] = Field(description="Arguments for tool")
    condition_field: Optional[str] = Field(default=None, description="Optional context field to check")
    condition_operator: Optional[str] = Field(default="equals", description="Operator: equals, not_equals, is_none, is_not_none")
    condition_value: Optional[Any] = Field(default=None, description="Expected value for condition")
    on_failure: str = Field(default="stop", description="Action on failure: 'stop', 'retry', 'rollback', 'continue'")


class WorkflowExecuteInput(BaseModel):
    workflow_id: str = Field(description="Workflow ID to execute")
