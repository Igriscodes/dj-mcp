# DJ-MCP: AI-Controlled Professional Music Production & DJ Workstation

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![MCP Server](https://img.shields.io/badge/MCP-1.0%20Compliant-green.svg)](https://modelcontextprotocol.io/)

**DJ-MCP** is an installable, cross-platform Python library and Model Context Protocol (MCP) server engineered for autonomous AI-controlled music production, DJ mixing, audio editing, and mastering.

The AI agent acts as the **producer and DJ**. The library acts as the **deterministic professional audio engine**, executing sample-accurate audio manipulation, musical analysis, and DSP without synthetic hallucinations or destructive modifications to source files.

---

## Table of Contents

- [1. Architectural Overview](#1-architectural-overview)
- [2. System Requirements & Platform Support](#2-system-requirements--platform-support)
- [3. Installation](#3-installation)
- [4. Diagnostic System Doctor](#4-diagnostic-system-doctor)
- [5. MCP Server Setup & Client Configuration](#5-mcp-server-setup--client-configuration)
  - [LM Studio](#lm-studio)
  - [Claude Desktop](#claude-desktop)
  - [Gemini CLI / Codex](#gemini-cli--openai-codex)
  - [Cursor / Windsurf](#cursor--windsurf)
- [6. Complete Tutorials](#6-complete-tutorials)
  - [Tutorial 1: Python Library Quickstart](#tutorial-1-python-library-quickstart)
  - [Tutorial 2: Automated 2-Deck Beatmatched Transition](#tutorial-2-automated-2-deck-beatmatched-transition)
  - [Tutorial 3: Non-Destructive Timeline Arrangement](#tutorial-3-non-destructive-timeline-arrangement)
  - [Tutorial 4: Neural Stem Separation & Remixing](#tutorial-4-neural-stem-separation--remixing)
  - [Tutorial 5: Mastering for Streaming & Club Systems](#tutorial-5-mastering-for-streaming--club-systems)
- [7. Complete MCP Tool Catalog & Schemas](#7-complete-mcp-tool-catalog--schemas)
- [8. Error Handling & Recovery for Small LLMs](#8-error-handling--recovery-for-small-llms)
- [9. Sequential Workflows & Conditional Execution](#9-sequential-workflows--conditional-execution)
- [10. Performance, Resource Management & Caching](#10-performance-resource-management--caching)
- [11. Testing & Verification](#11-testing--verification)
- [12. License & Attribution](#12-license--attribution)

---

## 1. Architectural Overview

DJ-MCP decouples high-level creative decision-making from low-level signal processing:

```text
┌─────────────────────────────────────────────────────────────┐
│             LLM / Agent (LM Studio, Claude, Codex)           │
└──────────────────────────────┬──────────────────────────────┘
                               │ MCP Protocol (stdio / sse)
┌──────────────────────────────▼──────────────────────────────┐
│                      DJ-MCP Server Layer                    │
│   - Strict Pydantic Schemas   - Structured Error Responses  │
│   - Actionable Next Steps     - Sequential Workflows        │
├─────────────────────────────────────────────────────────────┤
│                      Session & Project                      │
│   - Non-Destructive Timeline  - Decks & Crossfader State    │
│   - Mixer Channels & Sends    - Snapshot / Rollback Journal │
├──────────────────────────────┬──────────────────────────────┤
│               Core Audio & Analysis Engines                 │
│   ┌──────────────────────────┼──────────────────────────┐   │
│   │   Spotify Pedalboard     │         Librosa          │   │
│   │   (JUCE C++ Native DSP)  │  (BPM, Key, Transients)  │   │
│   ├──────────────────────────┼──────────────────────────┤   │
│   │      PyLoudnorm          │      Meta AI Demucs      │   │
│   │  (ITU-R BS.1770-4 LUFS)  │ (Neural Stem Separation) │   │
│   ├──────────────────────────┼──────────────────────────┤   │
│   │   Rubber Band / Soxr     │     FFmpeg & Soundfile   │   │
│   │ (Time-Stretch & Pitch)   │   (Lossless Audio I/O)   │   │
│   └──────────────────────────┴──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Deterministic Processing**: Identical parameters always yield identical PCM sample output.
2. **Non-Destructive by Default**: Original source files are never overwritten unless explicitly instructed with a destructive flag. Edits are stored as timeline offset slices.
3. **Sample Accuracy**: Beat positioning, cue slicing, and clip alignment preserve exact sample boundaries.
4. **Actionable Errors for Small LLMs**: Failed or precondition-blocked operations return structured error envelopes containing the exact next tool and parameters required to fix the issue.

---

## 2. System Requirements & Platform Support

DJ-MCP runs natively across:
- **Linux**: x86_64 and ARM64 (Debian/Ubuntu, Fedora, Arch, RHEL)
- **macOS**: Apple Silicon (M1/M2/M3/M4) and Intel x86_64 (macOS 12.0+)
- **Windows**: x64 and ARM64 (Windows 10, Windows 11)

### External Tools
- **FFmpeg / FFprobe**: Required for audio transcoding, metadata probing, and container export (MP3, AAC, OGG).
- **Python**: 3.10, 3.11, 3.12, or 3.13.

---

## 3. Installation

Clone and install in editable mode:

```bash
git clone https://github.com/antigravity-audio/dj-mcp.git
cd dj-mcp
pip install -e .
```

### Installation Extras

Install specialized components as needed:

```bash
# Analysis only: Librosa, PyLoudnorm, Soxr
pip install -e ".[analysis]"

# DSP only: Spotify Pedalboard (native JUCE DSP), pyrubberband
pip install -e ".[dsp]"

# Neural Stems only: Demucs, PyTorch, TorchAudio
pip install -e ".[stems]"

# Complete production suite (All engines)
pip install -e ".[all]"
```

---

## 4. Diagnostic System Doctor

Run the built-in system doctor to audit hardware capabilities, GPU acceleration, audio libraries, and file permissions:

```bash
dj-mcp doctor
```

Example output:

```text
============================================================
DJ-MCP SYSTEM DOCTOR DIAGNOSTIC REPORT
============================================================
Overall Status   : HEALTHY
Operating System : Linux (x86_64)
Python Version   : 3.13.12
CPU Cores        : 12
RAM Total/Avail  : 15.28 GB / 11.1 GB
GPU Available    : True (cuda)
------------------------------------------------------------
ENGINES & DSP:
  [✓ INSTALLED ] ffmpeg         : Audio container decoding, encoding, format conversion
  [✓ INSTALLED ] soundfile      : Sample-accurate uncompressed PCM and lossless I/O
  [✓ INSTALLED ] pedalboard     : Native C++ JUCE DSP effects (EQ, filters, compressor, limiter, reverb, delay)
  [✓ INSTALLED ] librosa        : BPM, beat-grid, onset/transient, chroma and key detection
  [✓ INSTALLED ] pyloudnorm     : ITU-R BS.1770-4 / EBU R128 integrated and momentary LUFS loudness analysis
  [✓ INSTALLED ] demucs         : High-fidelity neural source separation (vocals, drums, bass, other)
  [✓ INSTALLED ] rubberband     : High-quality time-stretching and pitch shifting
------------------------------------------------------------
CODECS & FORMATS:
  [✓ SUPPORTED   ] WAV
  [✓ SUPPORTED   ] FLAC
  [✓ SUPPORTED   ] AIFF
  [✓ SUPPORTED   ] OGG
  [✓ SUPPORTED   ] MP3
  [✓ SUPPORTED   ] AAC
------------------------------------------------------------
Disk Free Space  : 94441 MB
Cache Directory  : /home/igris/dj-mcp/.dj_cache (Writable: True)
Temp Directory   : /home/igris/dj-mcp/.dj_temp (Writable: True)
============================================================
```

To obtain output as structured JSON for automation:

```bash
dj-mcp doctor --json
```

---

## 5. MCP Server Setup & Client Configuration

Start the server using standard I/O transport (default for MCP agents):

```bash
dj-mcp.server
```

Or via the main CLI:

```bash
dj-mcp server --transport stdio
```

For HTTP Server-Sent Events (SSE) remote transport:

```bash
dj-mcp server --transport sse --host 0.0.0.0 --port 8080
```

### Client Configuration

#### LM Studio
In **Settings** -> **MCP Servers**, add:

```json
{
  "mcpServers": {
    "dj-mcp": {
      "command": "dj-mcp.server",
      "args": ["--transport", "stdio"]
    }
  }
}
```

#### Claude Desktop
Add to your `claude_desktop_config.json`:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "dj-mcp": {
      "command": "dj-mcp.server",
      "args": ["--transport", "stdio"]
    }
  }
}
```

#### Gemini CLI / OpenAI Codex
Configure in your respective CLI settings or environment profile:

```json
{
  "servers": {
    "dj-mcp": {
      "command": "dj-mcp.server"
    }
  }
}
```

#### Cursor / Windsurf
In **Settings** -> **Features** -> **MCP**, click **Add New MCP Server**:
- **Name**: `dj-mcp`
- **Type**: `command`
- **Command**: `dj-mcp.server`

---

## 6. Complete Tutorials

### Tutorial 1: Python Library Quickstart

This script imports an audio file, analyzes its musical properties, sets up a timeline track, applies parametric EQ and compression, masters to streaming loudness, and renders to FLAC:

```python
from pathlib import Path
from ai_dj import (
    Session,
    AudioClip,
    Track,
    MasteringProcessor,
    get_default_config,
)
from ai_dj.export import export_audio_to_file

# 1. Initialize session and project
config = get_default_config()
session = Session(config)
project = session.create_project(project_id="quickstart", name="Quickstart Project", sample_rate=44100)

# 2. Import and analyze audio
track_file = Path("samples/drop.wav")
lib_track = session.library.import_track(track_file, track_id="drop")
analysis = session.library.analyze_track("drop")

print(f"BPM: {analysis['bpm']}, Key: {analysis['key']} ({analysis['camelot']})")
print(f"Loudness: {analysis['loudness']['integrated_lufs']} LUFS")

# 3. Create a timeline track and non-destructive clip
track = Track(track_id="t_lead", name="Lead Track", channel_id="ch_lead")
clip = AudioClip(
    clip_id="c_lead_1",
    name="Lead Drop",
    source_file=track_file,
    source_offset_sec=0.0,
    timeline_position_sec=0.0,
    duration_sec=16.0,
    gain_db=-1.5,
    fade_in_sec=0.05,
    fade_out_sec=0.5,
)
track.add_clip(clip)
project.timeline.add_track(track)

# 4. Render unmastered timeline mix
raw_mix = session.renderer.render_timeline(
    project.timeline,
    project.mixer,
    start_sec=0.0,
    end_sec=16.0,
)

# 5. Master to Spotify/Apple Music streaming standards (-14 LUFS, -1.0 dBTP ceiling)
masterer = MasteringProcessor(profile="streaming", target_lufs=-14.0, true_peak_ceiling_db=-1.0)
mastered_audio, stats = masterer.process_audio(raw_mix, sample_rate=44100)

print(f"Mastered Loudness: {stats['final_lufs']} LUFS, Peak: {stats['final_peak_dbfs']} dBFS")

# 6. Export lossless FLAC
out_path = Path("exports/mastered_drop.flac")
export_audio_to_file(mastered_audio, out_path, sample_rate=44100, bit_depth=24, audio_format="flac")
print(f"Exported to: {out_path}")
```

---

### Tutorial 2: Automated 2-Deck Beatmatched Transition

In this tutorial, an AI agent loads two tracks onto Deck A and Deck B, matches their tempos, establishes a 4-beat loop, and performs an equal-power crossfader transition.

```python
from ai_dj import get_active_session
from ai_dj.mcp.tools import (
    tool_project_create,
    tool_library_import,
    tool_track_analyze,
    tool_create_beatgrid,
    tool_deck_load,
    tool_deck_play,
    tool_deck_sync,
    tool_loop_create,
    tool_mixer_crossfade,
    tool_render_project,
)

# 1. Create project
tool_project_create(project_id="dj_mix", name="Club Set", sample_rate=44100, num_decks=2)

# 2. Import tracks
tool_library_import(file_path="music/house_a.wav", track_id="track_a")
tool_library_import(file_path="music/house_b.wav", track_id="track_b")

# 3. Analyze BPM and generate beatgrids
res_a = tool_track_analyze(track_id="track_a")["result"]
res_b = tool_track_analyze(track_id="track_b")["result"]

tool_create_beatgrid(track_id="track_a", bpm=res_a["bpm"], first_beat_sec=res_a["first_beat_sec"])
tool_create_beatgrid(track_id="track_b", bpm=res_b["bpm"], first_beat_sec=res_b["first_beat_sec"])

# 4. Load Deck A and Deck B
tool_deck_load(deck_id="deck_a", track_id="track_a")
tool_deck_load(deck_id="deck_b", track_id="track_b")

# 5. Sync Deck B to Deck A's tempo and phase
tool_deck_sync(deck_id="deck_b", master_deck_id="deck_a")

# 6. Establish a 4-beat transition loop on Deck A
tool_loop_create(deck_id="deck_a", length_beats=4.0, start_sec=16.0)

# 7. Start playback and crossfade from Deck A to Deck B
tool_deck_play(deck_id="deck_a")
tool_deck_play(deck_id="deck_b")

# Move crossfader from Left (-1.0) to Center (0.0) to Right (+1.0)
tool_mixer_crossfade(position=0.0, curve="equal_power")
tool_mixer_crossfade(position=1.0, curve="smooth")
```

---

### Tutorial 3: Non-Destructive Timeline Arrangement

Edit and arrange audio without altering the underlying raw files:

```python
from ai_dj.mcp.tools import (
    tool_clip_create,
    tool_clip_split,
    tool_clip_move,
    tool_clip_time_stretch,
    tool_clip_pitch,
    tool_clip_fade,
)

# 1. Place a 30-second source segment onto timeline at 0.0s
tool_clip_create(
    track_id="t_vocal",
    clip_id="vocal_hook",
    source_file="vocals.wav",
    timeline_position_sec=0.0,
    source_offset_sec=10.0,
    duration_sec=8.0,
)

# 2. Split into two 4-second clips
tool_clip_split(
    track_id="t_vocal",
    clip_id="vocal_hook",
    split_offset_sec=4.0,
    new_clip_id="vocal_hook_pt2",
)

# 3. Time stretch second half by 1.25x and shift pitch up +2 semitones
tool_clip_time_stretch(track_id="t_vocal", clip_id="vocal_hook_pt2", rate=1.25)
tool_clip_pitch(track_id="t_vocal", clip_id="vocal_hook_pt2", semitones=2.0)

# 4. Apply a 50ms fade-in to eliminate clicks
tool_clip_fade(track_id="t_vocal", clip_id="vocal_hook_pt2", fade_in_sec=0.05, fade_out_sec=0.2)
```

---

### Tutorial 4: Neural Stem Separation & Remixing

Isolate vocals, drums, bass, and instrumental elements using Meta's Demucs backend:

```python
from ai_dj.mcp.tools import (
    tool_separate_stems,
    tool_job_status,
    tool_stem_mute,
    tool_stem_solo,
    tool_stem_gain,
    tool_stem_export,
)

# 1. Launch stem separation as a background job
job_resp = tool_separate_stems(audio_path="song.wav", output_dir="stems/song_stems", model_name="htdemucs")
job_id = job_resp["job_id"]
print(f"Background Job ID: {job_id}")

# 2. Check job progress
status = tool_job_status(job_id=job_id)
print(f"Status: {status['result']['status']}, Progress: {status['result']['progress']}%")

# 3. Once complete, adjust individual stem levels (e.g. acapella extract)
tool_stem_mute(set_id="song_stems", stem_name="drums", mute=True)
tool_stem_mute(set_id="song_stems", stem_name="bass", mute=True)
tool_stem_mute(set_id="song_stems", stem_name="other", mute=True)
tool_stem_solo(set_id="song_stems", stem_name="vocals", solo=True)
tool_stem_gain(set_id="song_stems", stem_name="vocals", gain_db=2.5)

# 4. Recombine and export the isolated vocal track
tool_stem_export(set_id="song_stems", output_path="exports/isolated_vocals.wav")
```

---

### Tutorial 5: Mastering for Streaming & Club Systems

Enforce target loudness and peak ceilings with multi-band mastering:

```python
from ai_dj.mcp.tools import (
    tool_master_process,
    tool_master_validate,
)

# Option A: Streaming Profile (-14.0 LUFS, -1.0 dBTP ceiling)
stream_res = tool_master_process(
    input_file="mixdown.wav",
    output_file="masters/streaming_master.wav",
    profile="streaming",
)
print("Streaming Master Stats:", stream_res["result"]["stats"])

# Validate against standard
val_stream = tool_master_validate(file_path="masters/streaming_master.wav", target_profile="streaming")
assert val_stream["result"]["is_compliant"] is True

# Option B: High-Energy Club/DJ Master (-9.0 LUFS, -0.3 dBTP ceiling)
club_res = tool_master_process(
    input_file="mixdown.wav",
    output_file="masters/club_master.wav",
    profile="dj_master",
    true_peak_ceiling_db=-0.3,
)
print("Club Master Stats:", club_res["result"]["stats"])
```

---

## 7. Complete MCP Tool Catalog & Schemas

DJ-MCP exposes high-level, deterministic tools with strict Pydantic parameter schemas.

### 1. Capabilities & Environment
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `system_capabilities` | Comprehensive OS, CPU, RAM, GPU, audio backends, and codecs report. | None |
| `available_engines` | Status of FFmpeg, Soundfile, Pedalboard, Librosa, PyLoudnorm, Demucs. | None |
| `available_codecs` | Supported container formats (WAV, FLAC, AIFF, OGG, MP3, AAC). | None |
| `audio_devices` | Active sound card playback and recording devices. | None |
| `available_models` | Installed neural separation weights (`htdemucs`, etc.). | None |
| `available_plugins` | List of native DSP filters, dynamics, and spatial effects. | None |

### 2. Project Lifecycle
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `project_create` | Initialize a new session project. | `project_id: str`, `name: str`, `sample_rate: int=44100`, `num_decks: int=2` |
| `project_open` | Load `.djproj` state file from disk. | `file_path: str` |
| `project_save` | Atomically serialize project state. | `file_path: Optional[str]` |
| `project_close` | Close active project session. | None |
| `project_info` | Get tracks, timeline duration, and deck states. | None |
| `project_validate` | Check for broken file paths, clipping, or missing routing. | None |
| `project_snapshot` | Take an undo/recovery snapshot. | `label: Optional[str]` |
| `project_restore` | Restore project to snapshot ID. | `snapshot_id: str` |
| `project_undo` | Undo previous action. | None |
| `project_redo` | Redo previously undone action. | None |

### 3. Audio Library & Scanning
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `library_scan` | Scan directory for audio files. | `directory: str`, `recursive: bool=True` |
| `library_search` | Query library by title, artist, BPM range, key. | `query: str`, `min_bpm: float`, `max_bpm: float`, `key: str` |
| `library_import` | Register file in session library. | `file_path: str`, `track_id: Optional[str]` |
| `library_remove` | Deregister track from library. | `track_id: str` |
| `library_metadata` | Inspect ID3/container tags and sample specs. | `track_id: str` |

### 4. DJ Decks & Synchronization
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `deck_create` | Add an extra DJ deck (`deck_c`, `deck_d`). | `deck_id: str` |
| `deck_load` | Load an analyzed track onto a deck. | `deck_id: str`, `track_id: str` |
| `deck_unload` | Eject track from deck. | `deck_id: str` |
| `deck_play` | Start deck playback. | `deck_id: str` |
| `deck_pause` | Pause deck playback. | `deck_id: str` |
| `deck_stop` | Stop playback and reset playhead to 0.0s. | `deck_id: str` |
| `deck_seek` | Jump to position in seconds (quantized). | `deck_id: str`, `position_sec: float`, `quantize: bool=True` |
| `deck_sync` | Lock tempo and beat-phase to master deck. | `deck_id: str`, `master_deck_id: str` |
| `deck_set_bpm` | Set explicit playback tempo. | `deck_id: str`, `bpm: float` |
| `deck_set_pitch` | Adjust pitch fader percentage. | `deck_id: str`, `pitch_percent: float` |
| `deck_set_key` | Shift pitch in semitones (Key Lock active). | `deck_id: str`, `semitones: float` |

### 5. Beat Grid & Musical Analysis
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `detect_bpm` | Auto-detect tempo and beat timestamps. | `file_path: str` |
| `detect_beats` | Extract onset beat times. | `file_path: str` |
| `detect_downbeats` | Identify bar markers (1.1, 2.1, etc.). | `file_path: str` |
| `create_beatgrid` | Anchor beatgrid to first downbeat. | `track_id: str`, `bpm: float`, `first_beat_sec: float=0.0` |
| `correct_beatgrid` | Nudge grid offset (+/-) or adjust BPM. | `track_id: str`, `offset_sec: float`, `bpm_override: float` |
| `quantize_to_grid` | Round time to nearest quantum (`bar`, `beat`, `1/2`, `1/4`, `1/8`, `1/16`). | `track_id: str`, `position_sec: float`, `quantum: str='beat'` |
| `analyze_key` | Detect musical scale and Camelot wheel code. | `file_path: str` |
| `analyze_loudness` | Measure integrated LUFS, peak, true-peak, and RMS. | `file_path: str` |
| `analyze_phase` | Check stereo phase correlation [-1.0, 1.0]. | `file_path: str` |
| `analyze_clipping` | Count digital overs and clipped samples. | `file_path: str` |

### 6. Non-Destructive Editing
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `clip_create` | Place audio segment on timeline. | `track_id: str`, `clip_id: str`, `source_file: str`, `timeline_position_sec: float`, `duration_sec: float` |
| `clip_trim` | Non-destructively trim clip start/end. | `track_id: str`, `clip_id: str`, `start_delta_sec: float`, `end_delta_sec: float` |
| `clip_split` | Split clip into two at offset. | `track_id: str`, `clip_id: str`, `split_offset_sec: float`, `new_clip_id: str` |
| `clip_cut` | Delete clip from track. | `track_id: str`, `clip_id: str` |
| `clip_join` | Merge two contiguous clips into one. | `track_id: str`, `clip_a_id: str`, `clip_b_id: str`, `new_clip_id: str` |
| `clip_move` | Reposition clip on timeline. | `track_id: str`, `clip_id: str`, `new_timeline_position_sec: float` |
| `clip_duplicate` | Duplicate clip with timeline offset. | `track_id: str`, `clip_id: str`, `new_clip_id: str`, `offset_sec: Optional[float]` |
| `clip_fade` | Apply fade-in / fade-out curves. | `track_id: str`, `clip_id: str`, `fade_in_sec: float`, `fade_out_sec: float` |
| `clip_gain` | Adjust clip gain in dB. | `track_id: str`, `clip_id: str`, `gain_db: float` |
| `clip_reverse` | Reverse audio playback in clip. | `track_id: str`, `clip_id: str`, `reverse: bool=True` |
| `clip_pitch` | Pitch shift audio in semitones. | `track_id: str`, `clip_id: str`, `semitones: float` |
| `clip_time_stretch` | Time-stretch playback rate (e.g. 1.25x). | `track_id: str`, `clip_id: str`, `rate: float` |
| `clip_normalize` | Normalize peak amplitude to target dBFS. | `track_id: str`, `clip_id: str`, `target_peak_dbfs: float=-0.5` |

### 7. Mixer, Routing & Effects
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `mixer_create_channel` | Add mixer channel strip. | `channel_id: str`, `name: Optional[str]` |
| `mixer_set_gain` | Set fader volume in dB [-60.0, 24.0]. | `channel_id: str`, `gain_db: float` |
| `mixer_set_pan` | Constant-power pan [-1.0, 1.0]. | `channel_id: str`, `pan: float` |
| `mixer_mute` | Mute/unmute channel. | `channel_id: str`, `mute: bool` |
| `mixer_solo` | Solo channel strip. | `channel_id: str`, `solo: bool` |
| `mixer_crossfade` | Crossfader position [-1.0, 1.0] and curve. | `position: float`, `curve: str='equal_power'` |
| `mixer_route` | Assign channel to crossfader (`a`, `b`, `thru`). | `channel_id: str`, `crossfader_assign: str` |
| `effect_add` | Insert DSP effect plugin. | `channel_id: str`, `effect_id: str`, `effect_type: str`, `parameters: dict` |
| `effect_set_parameter`| Modify active effect parameter. | `channel_id: str`, `effect_id: str`, `parameter_name: str`, `value: Any` |

### 8. Mastering & Rendering
| Tool | Description | Key Arguments |
| :--- | :--- | :--- |
| `master_process` | Apply full mastering chain to file. | `input_file: str`, `output_file: str`, `profile: str='dj_master'`, `target_lufs: float`, `true_peak_ceiling_db: float` |
| `master_validate` | Audit compliance against target standard. | `file_path: str`, `target_profile: str='streaming'` |
| `render_project` | Render entire timeline mix to audio file. | `output_path: str`, `start_sec: float=0.0`, `end_sec: Optional[float]`, `format: str='wav'`, `bit_depth: int=24` |
| `render_preview` | Render short preview snippet (15s). | `output_path: str`, `start_sec: float=0.0`, `duration_sec: float=15.0` |
| `export_audio` | Transcode file to target codec & bit depth. | `input_file: str`, `output_file: str`, `format: str`, `bit_depth: int=24` |

---

## 8. Error Handling & Recovery for Small LLMs

Small language models frequently fail when tools return plain text error strings like `"invalid argument"`. 

DJ-MCP enforces **Structured Error Envelopes**:

```json
{
  "success": false,
  "error": {
    "code": "DECK_NOT_FOUND",
    "message": "Deck 'deck_c' does not exist in current session.",
    "reason": "resource_not_found",
    "required_action": "deck_create",
    "suggested_tool": "deck_create",
    "suggested_arguments": {
      "deck_id": "deck_c"
    },
    "recoverable": true
  },
  "next_actions": [
    {
      "action": "deck_create",
      "tool": "deck_create",
      "arguments": {
        "deck_id": "deck_c"
      }
    }
  ]
}
```

### Self-Correction Loop

When a model encounters a failure:
1. It reads `error.suggested_tool` and `error.suggested_arguments`.
2. It executes the suggested tool to satisfy the precondition.
3. It retries the original operation with guaranteed state validity.

---

## 9. Sequential Workflows & Conditional Execution

The workflow engine allows complex multi-step pipelines with automated rollback:

```python
from ai_dj.mcp.tools import (
    tool_workflow_create,
    tool_workflow_add_step,
    tool_workflow_execute,
)

# 1. Create a sequential workflow
tool_workflow_create(workflow_id="prep_track", name="Track Import & Prep Pipeline")

# Step 1: Import Track
tool_workflow_add_step(
    workflow_id="prep_track",
    tool_name="library_import",
    arguments={"file_path": "audio/track.wav", "track_id": "my_song"},
    on_failure="stop",
)

# Step 2: Analyze BPM and Key
tool_workflow_add_step(
    workflow_id="prep_track",
    tool_name="track_analyze",
    arguments={"track_id": "my_song"},
    on_failure="rollback",
)

# Step 3: Load into Deck A only if analysis succeeds
tool_workflow_add_step(
    workflow_id="prep_track",
    tool_name="deck_load",
    arguments={"deck_id": "deck_a", "track_id": "my_song"},
    condition_field="bpm",
    condition_operator="is_not_none",
    on_failure="stop",
)

# 2. Execute the entire pipeline atomically
result = tool_workflow_execute(workflow_id="prep_track")
print("Workflow Result:", result)
```

---

## 10. Performance, Resource Management & Caching

- **Vectorized C++ DSP**: Filter calculations and dynamics run in compiled C++ through Spotify's `pedalboard` (JUCE framework) with SIMD optimizations.
- **Analysis Caching**: Audio hash + parameter hashes are stored under `.dj_cache/analysis_<hash>.json`. Identical audio files are analyzed in `< 1ms`.
- **Atomic Disk Writes**: Every save and render operation uses temporary swap files (`.tmp.<pid>`) with POSIX `fsync` replacement, ensuring project files never corrupt during abrupt stops.
- **Auto-Cleanup**: Temporary render buffers are automatically tracked and collected at process exit (`atexit`).

---

## 11. Testing & Verification

The codebase includes integration and unit tests covering DSP accuracy, deck synchronization, beatgrid generation, timeline editing, mastering compliance, and MCP tool invocations.

Execute the test suite:

```bash
pytest -v tests
```

### Output:

```text
============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/igris/dj-mcp
collected 24 items

tests/test_analysis_and_beatgrid.py::test_key_detection_and_camelot_compatibility PASSED [  4%]
tests/test_analysis_and_beatgrid.py::test_beatgrid_quantization PASSED   [  8%]
tests/test_analysis_and_beatgrid.py::test_audio_analysis_on_fixture PASSED [ 12%]
tests/test_capabilities.py::test_system_capabilities_structure PASSED    [ 16%]
tests/test_capabilities.py::test_doctor_execution PASSED                 [ 20%]
tests/test_decks_and_dj.py::test_deck_lifecycle PASSED                   [ 25%]
tests/test_decks_and_dj.py::test_deck_sync_and_pitch PASSED              [ 29%]
tests/test_decks_and_dj.py::test_cues_and_loops PASSED                   [ 33%]
tests/test_decks_and_dj.py::test_crossfader_curves PASSED                [ 37%]
tests/test_dsp_and_audio.py::test_stereo_panning PASSED                  [ 41%]
tests/test_dsp_and_audio.py::test_stereo_width_and_correlation PASSED    [ 45%]
tests/test_dsp_and_audio.py::test_dc_offset_removal PASSED               [ 50%]
tests/test_dsp_and_audio.py::test_pedalboard_effect_chain PASSED         [ 54%]
tests/test_dsp_and_audio.py::test_rubberband_pitch_and_time_stretch PASSED [ 58%]
tests/test_editing_and_timeline.py::test_clip_trim_and_split PASSED      [ 62%]
tests/test_editing_and_timeline.py::test_track_and_timeline_rendering PASSED [ 66%]
tests/test_errors_and_recovery.py::test_structured_error_dict PASSED     [ 70%]
tests/test_errors_and_recovery.py::test_atomic_write PASSED              [ 75%]
tests/test_errors_and_recovery.py::test_undo_redo_stack PASSED           [ 79%]
tests/test_jobs_and_workflows.py::test_job_manager_execution PASSED      [ 83%]
tests/test_jobs_and_workflows.py::test_sequential_workflow_execution PASSED [ 87%]
tests/test_mastering_and_rendering.py::test_mastering_processor PASSED   [ 91%]
tests/test_mastering_and_rendering.py::test_project_rendering_and_export PASSED [ 95%]
tests/test_mcp_tools.py::test_end_to_end_mcp_tool_workflow PASSED        [100%]

======================== 24 passed in 7.71s ========================
```

---

## 12. License & Attribution

This library is released under the **MIT License**. See [LICENSE](LICENSE) for full details.

### Third-Party Dependencies
- **Pedalboard**: Spotify (GPL v3 / JUCE)
- **SoundFile**: Erik de Castro Lopo / Bastian Bechtold (BSD 3-Clause)
- **Librosa**: Brian McFee et al. (ISC License)
- **PyLoudnorm**: Christian Steinmetz (MIT License)
- **Demucs**: Meta AI (MIT License)
- **FastMCP**: Model Context Protocol Team (MIT License)
