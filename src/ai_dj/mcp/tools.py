import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from ai_dj.analysis import AudioAnalyzer
from ai_dj.beatgrid import BeatGrid
from ai_dj.capabilities import get_system_capabilities
from ai_dj.clips import AudioClip
from ai_dj.config import get_default_config
from ai_dj.decks import Deck
from ai_dj.effects import EffectChain, EffectConfig
from ai_dj.engines.ffmpeg_engine import FFmpegEngine
from ai_dj.engines.rubberband_engine import RubberBandEngine
from ai_dj.engines.separation_engine import SeparationEngine
from ai_dj.errors import DJError, PrerequisiteError, ResourceNotFoundError, ValidationError
from ai_dj.export import export_audio_to_file
from ai_dj.key import calculate_key_semitone_shift, check_key_compatibility, key_to_camelot
from ai_dj.mastering import MasteringProcessor
from ai_dj.mcp.errors import handle_tool_errors
from ai_dj.mcp.responses import error_response, job_response, success_response
from ai_dj.mixer import MixerChannel
from ai_dj.session import Workflow, get_active_session
from ai_dj.stereo import calculate_phase_correlation
from ai_dj.tracks import Track
from ai_dj.validation import resolve_safe_path, validate_gain, validate_pan, validate_sample_rate
from ai_dj.waveform import generate_waveform_overview


# ==========================================
# Capabilities
# ==========================================

@handle_tool_errors
def tool_system_capabilities() -> Dict[str, Any]:
    caps = get_system_capabilities()
    return success_response(caps)


@handle_tool_errors
def tool_available_engines() -> Dict[str, Any]:
    caps = get_system_capabilities()
    return success_response(caps["engines"])


@handle_tool_errors
def tool_available_codecs() -> Dict[str, Any]:
    caps = get_system_capabilities()
    return success_response(caps["codecs"])


@handle_tool_errors
def tool_audio_devices() -> Dict[str, Any]:
    return success_response({
        "devices": [{"id": "default", "name": "Default Audio Device", "is_default": True}],
        "driver": "ALSA/Pulse/CoreAudio/WASAPI",
    })


@handle_tool_errors
def tool_available_models() -> Dict[str, Any]:
    sep = SeparationEngine()
    return success_response({
        "separation_models": ["htdemucs", "htdemucs_ft", "mdx_extra"],
        "active_model": sep.model_name,
        "is_available": sep.is_available,
    })


@handle_tool_errors
def tool_available_plugins() -> Dict[str, Any]:
    return success_response({
        "built_in_effects": ["eq", "compressor", "limiter", "reverb", "delay", "highpass", "lowpass", "distortion", "chorus"],
        "vst3_support": False,
        "clap_support": False,
    })


# ==========================================
# Project Tools
# ==========================================

@handle_tool_errors
def tool_project_create(project_id: str, name: str, sample_rate: int = 44100, num_decks: int = 2) -> Dict[str, Any]:
    sess = get_active_session()
    validate_sample_rate(sample_rate)
    proj = sess.create_project(project_id=project_id, name=name, sample_rate=sample_rate, num_decks=num_decks)
    return success_response(
        {
            "project_id": proj.project_id,
            "name": proj.name,
            "sample_rate": proj.sample_rate,
            "decks": list(proj.decks.keys()),
            "channels": list(proj.mixer.channels.keys()),
        },
        next_actions=[{"action": "import_audio", "tool": "library_import", "arguments": {"file_path": "audio.wav"}}],
    )


@handle_tool_errors
def tool_project_open(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    proj = sess.current_project = sess.current_project.load_from_file(safe_path) if sess.current_project else None
    from ai_dj.project import Project
    proj = Project.load_from_file(safe_path)
    sess.current_project = proj
    return success_response({"project_id": proj.project_id, "name": proj.name, "tracks": len(proj.timeline.tracks)})


@handle_tool_errors
def tool_project_save(file_path: Optional[str] = None) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    target_path = resolve_safe_path(file_path or f"{proj.project_id}.djproj", sess.config)
    saved = proj.save_to_file(target_path)
    return success_response({"file_path": str(saved), "project_id": proj.project_id})


@handle_tool_errors
def tool_project_close() -> Dict[str, Any]:
    sess = get_active_session()
    proj_id = sess.current_project.project_id if sess.current_project else None
    sess.current_project = None
    return success_response({"closed_project_id": proj_id})


@handle_tool_errors
def tool_project_info() -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    return success_response({
        "project_id": proj.project_id,
        "name": proj.name,
        "sample_rate": proj.sample_rate,
        "total_duration_sec": proj.timeline.total_duration_sec,
        "decks": {k: {"state": d.state, "loaded": d.loaded_track_id, "bpm": d.current_bpm} for k, d in proj.decks.items()},
        "tracks": list(proj.timeline.tracks.keys()),
    })


@handle_tool_errors
def tool_project_validate() -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    report = proj.validate_project()
    return success_response(report)


@handle_tool_errors
def tool_project_snapshot(label: Optional[str] = None) -> Dict[str, Any]:
    sess = get_active_session()
    snap_id = sess.take_snapshot(label)
    return success_response({"snapshot_id": snap_id})


@handle_tool_errors
def tool_project_restore(snapshot_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.restore_snapshot(snapshot_id)
    return success_response({"restored_snapshot_id": snapshot_id, "project_id": proj.project_id})


@handle_tool_errors
def tool_project_undo() -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.undo()
    if not proj:
        return error_response(DJError("CANNOT_UNDO", "No earlier snapshot in undo stack.", recoverable=True))
    return success_response({"action": "undo", "project_id": proj.project_id})


@handle_tool_errors
def tool_project_redo() -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.redo()
    if not proj:
        return error_response(DJError("CANNOT_REDO", "No forward snapshot in redo stack.", recoverable=True))
    return success_response({"action": "redo", "project_id": proj.project_id})


# ==========================================
# Library Tools
# ==========================================

@handle_tool_errors
def tool_library_scan(directory: str, recursive: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    safe_dir = resolve_safe_path(directory, sess.config, must_exist=True)
    tracks = sess.library.scan_directory(safe_dir, recursive=recursive)
    return success_response({"scanned_count": len(tracks), "tracks": tracks})


@handle_tool_errors
def tool_library_search(
    query: Optional[str] = None,
    min_bpm: Optional[float] = None,
    max_bpm: Optional[float] = None,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    sess = get_active_session()
    matches = sess.library.search(query=query, min_bpm=min_bpm, max_bpm=max_bpm, key=key)
    return success_response({"count": len(matches), "results": [t.model_dump() for t in matches]})


@handle_tool_errors
def tool_library_import(file_path: str, track_id: Optional[str] = None) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    track = sess.library.import_track(safe_path, track_id=track_id)
    return success_response(
        track.model_dump(),
        next_actions=[{"action": "analyze_track", "tool": "track_analyze", "arguments": {"track_id": track.track_id}}],
    )


@handle_tool_errors
def tool_library_remove(track_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    if track_id in sess.library.tracks:
        del sess.library.tracks[track_id]
        return success_response({"removed_track_id": track_id})
    raise ResourceNotFoundError("track", track_id)


@handle_tool_errors
def tool_library_metadata(track_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    track = sess.library.get_track(track_id)
    return success_response(track.metadata.model_dump())


# ==========================================
# Tracks Tools
# ==========================================

@handle_tool_errors
def tool_track_import(file_path: str, track_id: Optional[str] = None) -> Dict[str, Any]:
    return tool_library_import(file_path, track_id)


@handle_tool_errors
def tool_track_analyze(track_id: str, force: bool = False) -> Dict[str, Any]:
    sess = get_active_session()
    analysis = sess.library.analyze_track(track_id, force=force)
    return success_response(
        analysis,
        next_actions=[
            {"action": "load_deck", "tool": "deck_load", "arguments": {"deck_id": "deck_a", "track_id": track_id}},
            {"action": "create_beatgrid", "tool": "create_beatgrid", "arguments": {"track_id": track_id, "bpm": analysis["bpm"], "first_beat_sec": analysis["first_beat_sec"]}},
        ],
    )


@handle_tool_errors
def tool_track_info(track_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    track = sess.library.get_track(track_id)
    return success_response(track.model_dump())


@handle_tool_errors
def tool_track_waveform(track_id: str, num_points: int = 256) -> Dict[str, Any]:
    sess = get_active_session()
    track = sess.library.get_track(track_id)
    waveform = generate_waveform_overview(track.file_path, num_points=num_points)
    return success_response(waveform)


@handle_tool_errors
def tool_track_regions(track_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if track_id in proj.timeline.tracks:
        t = proj.timeline.tracks[track_id]
        return success_response({"track_id": track_id, "regions": [r.model_dump() for r in t.regions]})
    return success_response({"track_id": track_id, "regions": []})


@handle_tool_errors
def tool_track_export(track_id: str, output_path: str, format: Optional[str] = None) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    track = proj.timeline.get_track(track_id)
    audio = track.render_track_audio(target_sample_rate=proj.sample_rate)
    target = resolve_safe_path(output_path, sess.config)
    exported = export_audio_to_file(audio, target, sample_rate=proj.sample_rate, audio_format=format)
    return success_response({"exported_file": str(exported)})


# ==========================================
# Decks Tools
# ==========================================

@handle_tool_errors
def tool_deck_create(deck_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id in proj.decks:
        return success_response({"deck_id": deck_id, "status": "already_exists"})
    deck = Deck(deck_id=deck_id)
    proj.decks[deck_id] = deck
    ch_id = f"ch_{deck_id}"
    proj.mixer.add_channel(MixerChannel(channel_id=ch_id, name=f"Channel {deck_id}"))
    return success_response({"created_deck_id": deck_id})


@handle_tool_errors
def tool_deck_load(
    deck_id: str,
    track_id: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id, suggested_tool="deck_create", suggested_arguments={"deck_id": deck_id})

    identifier = track_id or audio_path
    if not identifier:
        raise ValidationError("track_id", "Either track_id or audio_path must be specified.")

    p = Path(identifier).expanduser()
    if p.exists() and p.is_file():
        track = sess.library.import_track(p)
    else:
        track = sess.library.get_track(identifier)

    if not track.analysis:
        track.analysis = sess.library.analyze_track(track.track_id)

    bpm = track.analysis.get("bpm", 120.0)
    first_beat = track.analysis.get("first_beat_sec", 0.0)
    bg = BeatGrid.generate_constant_grid(bpm=bpm, first_beat_sec=first_beat, duration_sec=track.metadata.duration_seconds)

    deck = proj.decks[deck_id]
    deck.load_track(track_id=track.track_id, audio_path=track.file_path, bpm=bpm, beatgrid=bg)

    return success_response(
        {
            "deck_id": deck_id,
            "track_id": track.track_id,
            "bpm": bpm,
            "key": track.analysis.get("key"),
            "camelot": track.analysis.get("camelot"),
        },
        next_actions=[
            {"action": "play_deck", "tool": "deck_play", "arguments": {"deck_id": deck_id}},
            {"action": "set_cue", "tool": "cue_create", "arguments": {"deck_id": deck_id, "cue_index": 1}},
        ],
    )


@handle_tool_errors
def tool_deck_unload(deck_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].unload()
    return success_response({"deck_id": deck_id, "status": "unloaded"})


@handle_tool_errors
def tool_deck_play(deck_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].play()
    return success_response({"deck_id": deck_id, "state": "playing", "position_sec": proj.decks[deck_id].position_sec})


@handle_tool_errors
def tool_deck_pause(deck_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].pause()
    return success_response({"deck_id": deck_id, "state": "paused", "position_sec": proj.decks[deck_id].position_sec})


@handle_tool_errors
def tool_deck_stop(deck_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].stop()
    return success_response({"deck_id": deck_id, "state": "stopped", "position_sec": 0.0})


@handle_tool_errors
def tool_deck_seek(deck_id: str, position_sec: float, quantize: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    new_pos = proj.decks[deck_id].seek(position_sec, quantize=quantize)
    return success_response({"deck_id": deck_id, "position_sec": new_pos})


@handle_tool_errors
def tool_deck_sync(deck_id: str, master_deck_id: str, allow_half_double_time: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    if master_deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", master_deck_id)

    master = proj.decks[master_deck_id]
    master.require_loaded()
    slave = proj.decks[deck_id]
    slave.require_loaded()

    master_bpm = master.current_bpm
    slave_orig = slave.original_bpm

    target_bpm = master_bpm
    sync_type = "exact"
    if allow_half_double_time and slave_orig > 0:
        candidates = [
            (master_bpm, "exact"),
            (master_bpm * 0.5, "half_time"),
            (master_bpm * 2.0, "double_time"),
        ]
        best_bpm, best_type = min(candidates, key=lambda c: abs(math.log2(c[0] / slave_orig)))
        target_bpm = best_bpm
        sync_type = best_type

    slave.sync_to(target_bpm=target_bpm, target_beat_phase_sec=master.position_sec)
    return success_response({
        "deck_id": deck_id,
        "master_deck_id": master_deck_id,
        "master_bpm": master_bpm,
        "synced_bpm": slave.current_bpm,
        "sync_type": sync_type,
        "pitch_percent": round(slave.pitch_percent, 2),
        "synced_position_sec": slave.position_sec,
    })


@handle_tool_errors
def tool_deck_set_bpm(deck_id: str, bpm: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].set_bpm(bpm)
    return success_response({"deck_id": deck_id, "current_bpm": bpm, "pitch_percent": proj.decks[deck_id].pitch_percent})


@handle_tool_errors
def tool_deck_set_pitch(deck_id: str, pitch_percent: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].set_pitch_percent(pitch_percent)
    return success_response({
        "deck_id": deck_id,
        "pitch_percent": pitch_percent,
        "current_bpm": proj.decks[deck_id].current_bpm,
    })


@handle_tool_errors
def tool_deck_set_key(deck_id: str, semitones: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].set_key_shift(semitones)
    return success_response({"deck_id": deck_id, "semitones": semitones})


# ==========================================
# Beat Grid Tools
# ==========================================

@handle_tool_errors
def tool_detect_bpm(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    analyzer = AudioAnalyzer(sess.config)
    y, sr = analyzer.load_audio_mono(safe_path)
    res = analyzer.analyze_bpm_and_beats(y, sr)
    return success_response(res)


@handle_tool_errors
def tool_detect_beats(file_path: str) -> Dict[str, Any]:
    return tool_detect_bpm(file_path)


@handle_tool_errors
def tool_detect_downbeats(file_path: str) -> Dict[str, Any]:
    bpm_res = tool_detect_bpm(file_path)
    if not bpm_res.get("success"):
        return bpm_res
    return success_response({"downbeats": bpm_res["result"].get("downbeats", [])})


@handle_tool_errors
def tool_create_beatgrid(track_id: str, bpm: float, first_beat_sec: float = 0.0) -> Dict[str, Any]:
    sess = get_active_session()
    track = sess.library.get_track(track_id)
    dur = track.metadata.duration_seconds
    bg = BeatGrid.generate_constant_grid(bpm=bpm, first_beat_sec=first_beat_sec, duration_sec=dur)
    if not track.analysis:
        track.analysis = {}
    track.analysis["bpm"] = bpm
    track.analysis["first_beat_sec"] = first_beat_sec
    track.analysis["beatgrid"] = bg.model_dump()
    return success_response({"track_id": track_id, "bpm": bpm, "beat_count": len(bg.beat_positions)})


@handle_tool_errors
def tool_correct_beatgrid(track_id: str, offset_sec: float, bpm_override: Optional[float] = None) -> Dict[str, Any]:
    sess = get_active_session()
    track = sess.library.get_track(track_id)
    if not track.analysis or "beatgrid" not in track.analysis:
        raise PrerequisiteError(
            code="BEATGRID_MISSING",
            message=f"Track '{track_id}' has no beatgrid.",
            required_action="create_beatgrid",
            suggested_tool="create_beatgrid",
            suggested_arguments={"track_id": track_id, "bpm": 120.0, "first_beat_sec": 0.0},
        )
    bg = BeatGrid.model_validate(track.analysis["beatgrid"])
    bg.shift_grid(offset_sec)
    if bpm_override:
        bg.correct_bpm(bpm_override, duration_sec=track.metadata.duration_seconds)
    track.analysis["beatgrid"] = bg.model_dump()
    track.analysis["bpm"] = bg.bpm
    return success_response({"track_id": track_id, "bpm": bg.bpm, "first_beat_sec": bg.first_beat_sec})


@handle_tool_errors
def tool_quantize_to_grid(track_id: str, position_sec: float, quantum: str = "beat") -> Dict[str, Any]:
    sess = get_active_session()
    track = sess.library.get_track(track_id)
    if not track.analysis or "beatgrid" not in track.analysis:
        raise PrerequisiteError(
            code="BEATGRID_MISSING",
            message=f"Track '{track_id}' has no beatgrid to quantize against.",
            required_action="create_beatgrid",
            suggested_tool="create_beatgrid",
            suggested_arguments={"track_id": track_id, "bpm": 120.0, "first_beat_sec": 0.0},
        )
    bg = BeatGrid.model_validate(track.analysis["beatgrid"])
    snapped = bg.quantize_position(position_sec, quantum=quantum)
    return success_response({"input_position": position_sec, "quantized_position": snapped, "quantum": quantum})


# ==========================================
# Editing Tools
# ==========================================

@handle_tool_errors
def tool_clip_create(
    track_id: str,
    clip_id: str,
    source_file: Optional[str] = None,
    file_path: Optional[str] = None,
    timeline_position_sec: float = 0.0,
    source_offset_sec: float = 0.0,
    duration_sec: float = 10.0,
) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    src = source_file or file_path
    if not src:
        raise ValidationError("source_file", "Either source_file or file_path must be specified.")
    safe_file = resolve_safe_path(src, sess.config, must_exist=True)

    if track_id not in proj.timeline.tracks:
        # Create track if not exists
        t = Track(track_id=track_id, name=f"Track {track_id}", channel_id=f"ch_{track_id}")
        proj.timeline.add_track(t)
        proj.mixer.add_channel(MixerChannel(channel_id=f"ch_{track_id}", name=f"Track {track_id}"))

    clip = AudioClip(
        clip_id=clip_id,
        name=clip_id,
        source_file=safe_file,
        timeline_position_sec=timeline_position_sec,
        source_offset_sec=source_offset_sec,
        duration_sec=duration_sec,
    )
    proj.timeline.get_track(track_id).add_clip(clip)
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_trim(
    track_id: str,
    clip_id: str,
    start_delta_sec: float = 0.0,
    end_delta_sec: float = 0.0,
) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    clip.trim(start_delta_sec=start_delta_sec, end_delta_sec=end_delta_sec)
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_split(
    track_id: str,
    clip_id: str,
    split_offset_sec: float,
    new_clip_id: str,
) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    track = proj.timeline.get_track(track_id)
    clip = track.get_clip(clip_id)
    first, second = clip.split(split_offset_sec=split_offset_sec, new_clip_id=new_clip_id)
    track.remove_clip(clip_id)
    track.add_clip(first)
    track.add_clip(second)
    return success_response({"first_clip": first.model_dump(), "second_clip": second.model_dump()})


@handle_tool_errors
def tool_clip_cut(track_id: str, clip_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    track = proj.timeline.get_track(track_id)
    removed = track.remove_clip(clip_id)
    if not removed:
        raise ResourceNotFoundError("clip", clip_id)
    return success_response({"removed_clip_id": clip_id})


@handle_tool_errors
def tool_clip_join(track_id: str, clip_a_id: str, clip_b_id: str, new_clip_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    track = proj.timeline.get_track(track_id)
    clip_a = track.get_clip(clip_a_id)
    clip_b = track.get_clip(clip_b_id)

    # Render combined audio
    audio_a = clip_a.render_audio(proj.sample_rate)
    audio_b = clip_b.render_audio(proj.sample_rate)
    combined = np.concatenate([audio_a, audio_b], axis=1)

    joined_path = sess.config.temp_dir / f"{new_clip_id}.wav"
    export_audio_to_file(combined, joined_path, sample_rate=proj.sample_rate)

    new_clip = AudioClip(
        clip_id=new_clip_id,
        name=new_clip_id,
        source_file=joined_path,
        timeline_position_sec=clip_a.timeline_position_sec,
        duration_sec=round(combined.shape[1] / proj.sample_rate, 4),
    )
    track.remove_clip(clip_a_id)
    track.remove_clip(clip_b_id)
    track.add_clip(new_clip)
    return success_response(new_clip.model_dump())


@handle_tool_errors
def tool_clip_move(track_id: str, clip_id: str, new_timeline_position_sec: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    clip.timeline_position_sec = max(0.0, float(new_timeline_position_sec))
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_duplicate(track_id: str, clip_id: str, new_clip_id: str, offset_sec: Optional[float] = None) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    track = proj.timeline.get_track(track_id)
    clip = track.get_clip(clip_id)
    dup = clip.model_copy(deep=True)
    dup.clip_id = new_clip_id
    dup.name = f"{clip.name}_copy"
    if offset_sec is not None:
        dup.timeline_position_sec = max(0.0, clip.timeline_position_sec + offset_sec)
    else:
        dup.timeline_position_sec = clip.end_timeline_position_sec
    track.add_clip(dup)
    return success_response(dup.model_dump())


@handle_tool_errors
def tool_clip_fade(
    track_id: str,
    clip_id: str,
    fade_in_sec: float = 0.0,
    fade_out_sec: float = 0.0,
) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    clip.fade_in_sec = max(0.0, float(fade_in_sec))
    clip.fade_out_sec = max(0.0, float(fade_out_sec))
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_gain(track_id: str, clip_id: str, gain_db: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    clip.gain_db = validate_gain(gain_db)
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_reverse(track_id: str, clip_id: str, reverse: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    clip.reverse = reverse
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_pitch(track_id: str, clip_id: str, semitones: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    clip.pitch_semitones = float(semitones)
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_time_stretch(track_id: str, clip_id: str, rate: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    if rate <= 0.0:
        raise ValidationError("rate", "Rate must be positive")
    clip.time_stretch_rate = float(rate)
    return success_response(clip.model_dump())


@handle_tool_errors
def tool_clip_normalize(track_id: str, clip_id: str, target_peak_dbfs: float = -0.5) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    clip = proj.timeline.get_track(track_id).get_clip(clip_id)
    audio = clip.render_audio(proj.sample_rate)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        current_peak_db = 20.0 * math.log10(peak)
        gain_needed = target_peak_dbfs - current_peak_db
        clip.gain_db += gain_needed
    return success_response({"clip_id": clip_id, "gain_db": round(clip.gain_db, 2)})


@handle_tool_errors
def tool_clip_align(track_id: str, clip_id: str, target_time_sec: float) -> Dict[str, Any]:
    return tool_clip_move(track_id, clip_id, target_time_sec)


# ==========================================
# Cues and Loops Tools
# ==========================================

@handle_tool_errors
def tool_cue_create(deck_id: str, cue_index: int, position_sec: Optional[float] = None) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    cue = proj.decks[deck_id].set_cue(cue_idx=cue_index, position_sec=position_sec)
    return success_response(cue.model_dump())


@handle_tool_errors
def tool_cue_delete(deck_id: str, cue_index: int) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    deck = proj.decks[deck_id]
    if cue_index in deck.cues:
        del deck.cues[cue_index]
        return success_response({"deck_id": deck_id, "deleted_cue": cue_index})
    raise DJError("CUE_NOT_FOUND", f"Cue {cue_index} not found on deck {deck_id}.")


@handle_tool_errors
def tool_loop_create(deck_id: str, length_beats: float = 4.0, start_sec: Optional[float] = None) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    loop = proj.decks[deck_id].set_loop(length_beats=length_beats, start_sec=start_sec)
    return success_response(loop.model_dump())


@handle_tool_errors
def tool_loop_edit(deck_id: str, action: str) -> Dict[str, Any]:
    # action: "double", "halve", "shift_forward", "shift_backward"
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    deck = proj.decks[deck_id]
    if not deck.active_loop:
        raise DJError("NO_ACTIVE_LOOP", f"Deck '{deck_id}' has no active loop.", suggested_tool="loop_create")

    if action == "double":
        deck.active_loop.double_length()
    elif action == "halve":
        deck.active_loop.halve_length()
    elif action == "shift_forward":
        dur = deck.active_loop.duration_sec
        deck.active_loop.shift(dur)
    elif action == "shift_backward":
        dur = deck.active_loop.duration_sec
        deck.active_loop.shift(-dur)
    else:
        raise ValidationError("action", f"Unknown loop edit action '{action}'")

    return success_response(deck.active_loop.model_dump())


@handle_tool_errors
def tool_loop_delete(deck_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if deck_id not in proj.decks:
        raise ResourceNotFoundError("deck", deck_id)
    proj.decks[deck_id].clear_loop()
    return success_response({"deck_id": deck_id, "status": "loop_cleared"})


# ==========================================
# Mixer Tools
# ==========================================

@handle_tool_errors
def tool_mixer_create_channel(channel_id: str, name: Optional[str] = None) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    ch = MixerChannel(channel_id=channel_id, name=name or f"Channel {channel_id}")
    proj.mixer.add_channel(ch)
    return success_response(ch.model_dump())


@handle_tool_errors
def tool_mixer_set_gain(channel_id: str, gain_db: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    proj.mixer.channels[channel_id].gain_db = validate_gain(gain_db)
    return success_response({"channel_id": channel_id, "gain_db": gain_db})


@handle_tool_errors
def tool_mixer_set_pan(channel_id: str, pan: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    proj.mixer.channels[channel_id].pan = validate_pan(pan)
    return success_response({"channel_id": channel_id, "pan": pan})


@handle_tool_errors
def tool_mixer_mute(channel_id: str, mute: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    proj.mixer.channels[channel_id].mute = mute
    return success_response({"channel_id": channel_id, "mute": mute})


@handle_tool_errors
def tool_mixer_solo(channel_id: str, solo: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    proj.mixer.channels[channel_id].solo = solo
    return success_response({"channel_id": channel_id, "solo": solo})


@handle_tool_errors
def tool_mixer_crossfade(position: float, curve: str = "equal_power") -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    proj.mixer.crossfader_position = max(-1.0, min(1.0, float(position)))
    proj.mixer.crossfader_curve = curve
    from ai_dj.mixer import calculate_crossfader_gains
    ga, gb = calculate_crossfader_gains(proj.mixer.crossfader_position, curve)
    return success_response({
        "crossfader_position": proj.mixer.crossfader_position,
        "curve": curve,
        "gain_deck_a": ga,
        "gain_deck_b": gb,
    })


@handle_tool_errors
def tool_mixer_route(channel_id: str, crossfader_assign: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    if crossfader_assign not in ("a", "b", "thru"):
        raise ValidationError("crossfader_assign", "Must be 'a', 'b', or 'thru'")
    proj.mixer.channels[channel_id].crossfader_assign = crossfader_assign
    return success_response({"channel_id": channel_id, "crossfader_assign": crossfader_assign})


@handle_tool_errors
def tool_mixer_create_bus(bus_id: str, name: str) -> Dict[str, Any]:
    sess = get_active_session()
    return success_response({"bus_id": bus_id, "name": name, "status": "created"})


@handle_tool_errors
def tool_mixer_create_send(source_id: str, target_bus_id: str, send_gain_db: float = 0.0) -> Dict[str, Any]:
    sess = get_active_session()
    return success_response({"source_id": source_id, "target_bus_id": target_bus_id, "send_gain_db": send_gain_db})


# ==========================================
# Effects Tools
# ==========================================

@handle_tool_errors
def tool_effect_add(channel_id: str, effect_id: str, effect_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    # Directly set channel parameters if EQ or Filter
    ch = proj.mixer.channels[channel_id]
    p = parameters or {}
    if effect_type == "eq":
        ch.eq_low_gain_db = float(p.get("low_gain_db", 0.0))
        ch.eq_mid_gain_db = float(p.get("mid_gain_db", 0.0))
        ch.eq_high_gain_db = float(p.get("high_gain_db", 0.0))
    elif effect_type == "highpass":
        ch.filter_hpf_hz = float(p.get("cutoff_hz", 80.0))
    elif effect_type == "lowpass":
        ch.filter_lpf_hz = float(p.get("cutoff_hz", 12000.0))

    return success_response({
        "channel_id": channel_id,
        "effect_id": effect_id,
        "effect_type": effect_type,
        "parameters": p,
    })


@handle_tool_errors
def tool_effect_remove(channel_id: str, effect_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    return success_response({"channel_id": channel_id, "removed_effect_id": effect_id})


@handle_tool_errors
def tool_effect_bypass(channel_id: str, effect_id: str, bypass: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    return success_response({"channel_id": channel_id, "effect_id": effect_id, "bypassed": bypass})


@handle_tool_errors
def tool_effect_set_parameter(channel_id: str, effect_id: str, parameter_name: str, value: Any) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    ch = proj.mixer.channels[channel_id]
    if hasattr(ch, parameter_name):
        setattr(ch, parameter_name, value)
    return success_response({"channel_id": channel_id, "effect_id": effect_id, "parameter": parameter_name, "value": value})


@handle_tool_errors
def tool_effect_list(channel_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if channel_id not in proj.mixer.channels:
        raise ResourceNotFoundError("channel", channel_id)
    ch = proj.mixer.channels[channel_id]
    effects = [
        {"effect_id": "eq", "type": "eq", "parameters": {"low": ch.eq_low_gain_db, "mid": ch.eq_mid_gain_db, "high": ch.eq_high_gain_db}},
        {"effect_id": "filter", "type": "filter", "parameters": {"hpf": ch.filter_hpf_hz, "lpf": ch.filter_lpf_hz}},
    ]
    return success_response({"channel_id": channel_id, "effects": effects})


@handle_tool_errors
def tool_effect_chain(channel_id: str) -> Dict[str, Any]:
    return tool_effect_list(channel_id)


# ==========================================
# Audio Analysis Tools
# ==========================================

@handle_tool_errors
def tool_analyze_bpm(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    analyzer = AudioAnalyzer(sess.config)
    y, sr = analyzer.load_audio_mono(safe_path)
    return success_response(analyzer.analyze_bpm_and_beats(y, sr))


@handle_tool_errors
def tool_analyze_key(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    analyzer = AudioAnalyzer(sess.config)
    y, sr = analyzer.load_audio_mono(safe_path)
    return success_response(analyzer.analyze_key(y, sr))


@handle_tool_errors
def tool_analyze_loudness(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    analyzer = AudioAnalyzer(sess.config)
    return success_response(analyzer.analyze_loudness_and_dynamics(safe_path))


@handle_tool_errors
def tool_analyze_dynamics(file_path: str) -> Dict[str, Any]:
    return tool_analyze_loudness(file_path)


@handle_tool_errors
def tool_analyze_spectrum(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    y, sr = AudioAnalyzer(sess.config).load_audio_mono(safe_path, max_duration=30.0)
    fft_mag = np.abs(np.fft.rfft(y[:4096]))
    return success_response({
        "sample_rate": sr,
        "fft_size": 4096,
        "spectral_centroid_hz": round(float(np.sum(np.arange(len(fft_mag)) * fft_mag) / (np.sum(fft_mag) + 1e-9) * (sr / 4096)), 1),
    })


@handle_tool_errors
def tool_analyze_phase(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    data, sr = sf.read(str(safe_path), always_2d=True)
    corr = calculate_phase_correlation(data.T)
    return success_response({"phase_correlation": corr, "in_phase": corr > 0.0})


@handle_tool_errors
def tool_analyze_clipping(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    data, sr = sf.read(str(safe_path), always_2d=True)
    clipped_samples = int(np.sum(np.abs(data) >= 0.9999))
    return success_response({
        "clipped_samples_count": clipped_samples,
        "clipping_detected": clipped_samples > 0,
        "peak_amplitude": round(float(np.max(np.abs(data))), 4),
    })


@handle_tool_errors
def tool_analyze_silence(file_path: str, threshold_db: float = -60.0) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    data, sr = sf.read(str(safe_path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    lin_thresh = math.pow(10.0, threshold_db / 20.0)
    silent_samples = int(np.sum(np.abs(data) < lin_thresh))
    pct = round(silent_samples / len(data) * 100.0, 2)
    return success_response({"silent_percentage": pct, "threshold_db": threshold_db})


@handle_tool_errors
def tool_analyze_transients(file_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    safe_path = resolve_safe_path(file_path, sess.config, must_exist=True)
    import librosa
    y, sr = AudioAnalyzer(sess.config).load_audio_mono(safe_path, max_duration=60.0)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    return success_response({"transient_count": len(onsets), "transient_times_sec": [round(float(t), 4) for t in onsets[:64]]})


# ==========================================
# Stem Operations Tools
# ==========================================

@handle_tool_errors
def tool_separate_stems(
    audio_path: str,
    output_dir: Optional[str] = None,
    model_name: str = "htdemucs",
    wait: bool = True,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_audio = resolve_safe_path(audio_path, sess.config, must_exist=True)
    out_dir = resolve_safe_path(output_dir or f"stems_{safe_audio.stem}", sess.config)

    sep = SeparationEngine(model_name=model_name)
    if not sep.is_available:
        raise PrerequisiteError(
            code="DEMUCS_NOT_INSTALLED",
            message="Demucs stem separation engine is not available.",
            required_action="install_demucs",
            suggested_tool="system_capabilities",
        )

    if wait:
        res_paths = sep.separate(safe_audio, out_dir)
        stems_dict = {k: str(v) for k, v in res_paths.items()}
        if sess.current_project:
            from ai_dj.stems import StemSet
            s_set = StemSet(
                set_id=f"stems_{safe_audio.stem}",
                source_file=safe_audio,
                model_used=model_name,
                stems=res_paths,
            )
            sess.current_project.stems[s_set.set_id] = s_set

        return success_response({
            "status": "completed",
            "output_directory": str(out_dir),
            "stems": stems_dict,
            "model_used": model_name,
            "stems_count": len(stems_dict),
        })

    # Run separation as background job
    def job_fn():
        return sep.separate(safe_audio, out_dir)

    job_id = sess.job_manager.submit_job(f"Stem Separation: {safe_audio.name}", job_fn)
    return job_response(job_id)


@handle_tool_errors
def tool_list_stems(set_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if set_id in proj.stems:
        return success_response(proj.stems[set_id].model_dump())
    return success_response({"set_id": set_id, "stems": []})


@handle_tool_errors
def tool_stem_mute(set_id: str, stem_name: str, mute: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if set_id not in proj.stems:
        raise ResourceNotFoundError("stem_set", set_id)
    proj.stems[set_id].mute_stem(stem_name, mute)
    return success_response({"set_id": set_id, "stem_name": stem_name, "mute": mute})


@handle_tool_errors
def tool_stem_solo(set_id: str, stem_name: str, solo: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if set_id not in proj.stems:
        raise ResourceNotFoundError("stem_set", set_id)
    proj.stems[set_id].solo_stem(stem_name, solo)
    return success_response({"set_id": set_id, "stem_name": stem_name, "solo": solo})


@handle_tool_errors
def tool_stem_gain(set_id: str, stem_name: str, gain_db: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if set_id not in proj.stems:
        raise ResourceNotFoundError("stem_set", set_id)
    proj.stems[set_id].set_stem_gain(stem_name, gain_db)
    return success_response({"set_id": set_id, "stem_name": stem_name, "gain_db": gain_db})


@handle_tool_errors
def tool_stem_align(set_id: str, stem_name: str, offset_sec: float) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if set_id not in proj.stems:
        raise ResourceNotFoundError("stem_set", set_id)
    stems = proj.stems[set_id]
    if stem_name not in stems.stems:
        raise ResourceNotFoundError("stem", stem_name)
    stems.stems[stem_name].time_offset_sec = float(offset_sec)
    return success_response({"set_id": set_id, "stem_name": stem_name, "offset_sec": offset_sec})


@handle_tool_errors
def tool_stem_export(set_id: str, output_path: str) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    if set_id not in proj.stems:
        raise ResourceNotFoundError("stem_set", set_id)
    stems = proj.stems[set_id]
    recombined = stems.recombine(proj.sample_rate)
    target = resolve_safe_path(output_path, sess.config)
    exported = export_audio_to_file(recombined, target, sample_rate=proj.sample_rate)
    return success_response({"exported_file": str(exported)})


@handle_tool_errors
def tool_audio_merge(
    input_files: Any,
    output_file: str,
    gains_db: Optional[Dict[str, float]] = None,
    normalize: bool = False,
    brickwall_limit: bool = True,
    target_peak_dbfs: float = -0.5,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    target_out = resolve_safe_path(output_file, sess.config)

    resolved_files: List[Path] = []
    # 1. Resolve inputs: directory, list of files, or stem set ID
    if isinstance(input_files, str):
        clean_str = input_files.strip()
        p = Path(clean_str).expanduser()
        if p.is_dir():
            audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}
            all_files = [f for f in sorted(p.iterdir()) if f.suffix.lower() in audio_exts]
            # Avoid duplicate tracks if both vocals.wav and Yesha Nagula_vocals.wav exist
            standard_stems = {"vocals", "drums", "bass", "other"}
            std_matches = [f for f in all_files if f.stem.lower() in standard_stems]
            resolved_files = std_matches if len(std_matches) >= 2 else all_files
        elif sess.current_project and clean_str in sess.current_project.stems:
            resolved_files = list(sess.current_project.stems[clean_str].stems.values())
        elif "," in clean_str:
            resolved_files = [resolve_safe_path(x.strip(), sess.config, must_exist=True) for x in clean_str.split(",") if x.strip()]
        else:
            resolved_files = [resolve_safe_path(clean_str, sess.config, must_exist=True)]
    elif isinstance(input_files, (list, tuple)):
        resolved_files = [resolve_safe_path(str(x), sess.config, must_exist=True) for x in input_files]
    else:
        raise ValidationError("input_files", "input_files must be a list of paths or directory string.")

    if not resolved_files:
        raise ValidationError("input_files", "No valid audio files found to merge.")

    # 2. Read and align audio signals
    signals: List[np.ndarray] = []
    target_sr = 44100
    for idx, fpath in enumerate(resolved_files):
        data, sr = sf.read(str(fpath), always_2d=True, dtype="float32")
        sig = data.T
        if idx == 0:
            target_sr = sr
        elif sr != target_sr:
            import scipy.signal as signal
            gcd = math.gcd(sr, target_sr)
            sig = signal.resample_poly(sig, target_sr // gcd, sr // gcd, axis=-1).astype(np.float32)

        if sig.shape[0] == 1:
            sig = np.repeat(sig, 2, axis=0)
        elif sig.shape[0] > 2:
            sig = sig[:2]

        if gains_db:
            gain = gains_db.get(fpath.name, gains_db.get(fpath.stem, 0.0))
            if abs(gain) > 1e-4:
                sig = sig * (10.0 ** (gain / 20.0))

        signals.append(sig)

    max_samples = max(s.shape[1] for s in signals)
    summed = np.zeros((2, max_samples), dtype=np.float32)
    for s in signals:
        summed[:, :s.shape[1]] += s

    # 3. Peak limiting and normalization to prevent clipping
    peak = float(np.max(np.abs(summed))) if summed.size > 0 else 0.0
    if normalize:
        target_lin = 10.0 ** (target_peak_dbfs / 20.0)
        if peak > 0:
            summed = summed * (target_lin / peak)
    elif brickwall_limit and peak > 1.0:
        target_lin = 10.0 ** (target_peak_dbfs / 20.0)
        summed = summed * (target_lin / peak)

    exported = export_audio_to_file(
        summed,
        target_out,
        sample_rate=target_sr,
        bit_depth=bit_depth,
        audio_format=format,
    )

    final_peak = float(np.max(np.abs(summed))) if summed.size > 0 else 0.0
    peak_dbfs = round(20.0 * math.log10(max(1e-6, final_peak)), 2)

    return success_response({
        "output_file": str(exported),
        "files_merged": [str(f) for f in resolved_files],
        "file_count": len(resolved_files),
        "duration_seconds": round(summed.shape[1] / target_sr, 3),
        "channels": 2,
        "sample_rate": target_sr,
        "peak_dbfs": peak_dbfs,
        "limited_or_normalized": normalize or (brickwall_limit and peak > 1.0),
    })


@handle_tool_errors
def tool_stems_merge(
    stems_dir: str,
    output_file: str,
    normalize: bool = True,
    target_peak_dbfs: float = -0.5,
) -> Dict[str, Any]:
    return tool_audio_merge(
        input_files=stems_dir,
        output_file=output_file,
        normalize=normalize,
        target_peak_dbfs=target_peak_dbfs,
    )


# ==========================================
# Mastering Tools
# ==========================================

@handle_tool_errors
def tool_master_analyze(file_path: str) -> Dict[str, Any]:
    return tool_analyze_loudness(file_path)


@handle_tool_errors
def tool_master_process(
    input_file: str,
    output_file: str,
    profile: str = "dj_master",
    target_lufs: Optional[float] = None,
    true_peak_ceiling_db: Optional[float] = None,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = resolve_safe_path(input_file, sess.config, must_exist=True)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    masterer = MasteringProcessor(
        profile=profile,
        target_lufs=target_lufs,
        true_peak_ceiling_db=true_peak_ceiling_db,
    )
    mastered, stats = masterer.process_audio(data.T, sr)
    export_audio_to_file(mastered, safe_out, sample_rate=sr, bit_depth=24)
    return success_response({"output_file": str(safe_out), "stats": stats})


@handle_tool_errors
def tool_master_normalize(file_path: str, output_path: str, target_lufs: float = -14.0) -> Dict[str, Any]:
    return tool_master_process(file_path, output_path, profile="custom", target_lufs=target_lufs)


@handle_tool_errors
def tool_master_limit(file_path: str, output_path: str, ceiling_db: float = -0.3) -> Dict[str, Any]:
    return tool_master_process(file_path, output_path, profile="custom", true_peak_ceiling_db=ceiling_db)


@handle_tool_errors
def tool_master_validate(file_path: str, target_profile: str = "dj_master") -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = resolve_safe_path(file_path, sess.config, must_exist=True)
    analyzer = AudioAnalyzer(sess.config)
    stats = analyzer.analyze_loudness_and_dynamics(safe_in)
    from ai_dj.mastering import MASTERING_PROFILES
    target = MASTERING_PROFILES.get(target_profile, MASTERING_PROFILES["dj_master"])

    lufs_ok = abs(stats["integrated_lufs"] - target["target_lufs"]) <= 1.0
    tp_ok = stats["true_peak_dbfs"] <= target["true_peak_ceiling_db"] + 0.1

    return success_response({
        "file": str(safe_in),
        "profile": target_profile,
        "is_compliant": lufs_ok and tp_ok,
        "measured_lufs": stats["integrated_lufs"],
        "target_lufs": target["target_lufs"],
        "measured_true_peak": stats["true_peak_dbfs"],
        "ceiling_true_peak": target["true_peak_ceiling_db"],
    })


# ==========================================
# Rendering Tools
# ==========================================

@handle_tool_errors
def tool_render_project(
    output_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    target = resolve_safe_path(output_path, sess.config)
    audio = sess.renderer.render_timeline(
        timeline=proj.timeline,
        mixer=proj.mixer,
        decks=proj.decks,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_rate=proj.sample_rate,
    )
    exported = export_audio_to_file(audio, target, sample_rate=proj.sample_rate, bit_depth=bit_depth, audio_format=format)
    return success_response({
        "rendered_file": str(exported),
        "duration_seconds": round(audio.shape[1] / proj.sample_rate, 3),
        "channels": audio.shape[0],
    })


@handle_tool_errors
def tool_render_preview(
    output_path: str,
    start_sec: float = 0.0,
    duration_sec: float = 15.0,
) -> Dict[str, Any]:
    return tool_render_project(output_path, start_sec=start_sec, end_sec=start_sec + duration_sec)


@handle_tool_errors
def tool_render_region(output_path: str, start_sec: float, end_sec: float) -> Dict[str, Any]:
    return tool_render_project(output_path, start_sec=start_sec, end_sec=end_sec)


@handle_tool_errors
def tool_render_track(track_id: str, output_path: str) -> Dict[str, Any]:
    return tool_track_export(track_id, output_path)


@handle_tool_errors
def tool_render_stems(set_id: str, output_path: str) -> Dict[str, Any]:
    return tool_stem_export(set_id, output_path)


@handle_tool_errors
def tool_export_audio(
    input_file: str,
    output_file: str,
    format: Optional[str] = None,
    bit_depth: int = 24,
    bit_rate_kbps: int = 320,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = resolve_safe_path(input_file, sess.config, must_exist=True)
    safe_out = resolve_safe_path(output_file, sess.config)
    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    exported = export_audio_to_file(
        data.T,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
        bit_rate_kbps=bit_rate_kbps,
    )
    return success_response({"exported_file": str(exported)})


# ==========================================
# Direct Audio & Pitch/Tempo Editing Tools (Audacity-Style & Standalone DSP)
# ==========================================

def _resolve_audio_input(identifier: str, sess) -> Path:
    p = Path(identifier).expanduser()
    if p.exists() and p.is_file():
        return resolve_safe_path(str(p), sess.config, must_exist=True)
    if identifier in sess.library.tracks:
        return sess.library.tracks[identifier].file_path
    return resolve_safe_path(identifier, sess.config, must_exist=True)


@handle_tool_errors
def tool_audio_change_tempo(
    input_file: str,
    output_file: str,
    target_bpm: Optional[float] = None,
    tempo_change_percent: Optional[float] = None,
    source_bpm: Optional[float] = None,
    target_duration_sec: Optional[float] = None,
    preserve_pitch: bool = True,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T
    orig_dur = audio.shape[1] / sr

    applied_source_bpm = source_bpm
    if target_bpm is not None:
        if not applied_source_bpm:
            analyzer = AudioAnalyzer(sess.config)
            track_info = analyzer.analyze_track(safe_in)
            applied_source_bpm = track_info.get("bpm", 120.0)
        rate = float(target_bpm / applied_source_bpm) if applied_source_bpm > 0 else 1.0
    elif tempo_change_percent is not None:
        rate = float(1.0 + (tempo_change_percent / 100.0))
    elif target_duration_sec is not None and target_duration_sec > 0:
        rate = float(orig_dur / target_duration_sec)
    else:
        raise ValidationError("target_bpm", "Provide target_bpm, tempo_change_percent, or target_duration_sec.")

    rb = RubberBandEngine()
    if preserve_pitch:
        processed = rb.time_stretch(audio, sr, rate=rate)
    else:
        processed = rb.change_speed(audio, sr, speed_ratio=rate)

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    final_dur = processed.shape[1] / sr
    return success_response({
        "output_file": str(exported),
        "source_bpm": round(applied_source_bpm, 2) if applied_source_bpm else None,
        "target_bpm": round(target_bpm, 2) if target_bpm else (round(applied_source_bpm * rate, 2) if applied_source_bpm else None),
        "tempo_change_percent": round((rate - 1.0) * 100.0, 2),
        "stretch_rate": round(rate, 4),
        "original_duration_seconds": round(orig_dur, 3),
        "final_duration_seconds": round(final_dur, 3),
        "pitch_preserved": preserve_pitch,
    })


@handle_tool_errors
def tool_audio_change_pitch(
    input_file: str,
    output_file: str,
    semitones: Optional[float] = None,
    cents: Optional[float] = None,
    from_key: Optional[str] = None,
    to_key: Optional[str] = None,
    preserve_tempo: bool = True,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T

    shift = 0.0
    detected_key = None
    if semitones is not None:
        shift = float(semitones)
    elif cents is not None:
        shift = float(cents) / 100.0
    elif to_key is not None:
        source_k = from_key
        if not source_k:
            analyzer = AudioAnalyzer(sess.config)
            track_info = analyzer.analyze_track(safe_in)
            source_k = track_info.get("key")
            detected_key = source_k
        if not source_k:
            raise ValidationError("from_key", "Source key could not be detected. Please provide from_key explicitly.")
        shift = calculate_key_semitone_shift(source_k, to_key)
    else:
        raise ValidationError("semitones", "Provide semitones, cents, or to_key.")

    rb = RubberBandEngine()
    if preserve_tempo:
        processed = rb.pitch_shift(audio, sr, n_semitones=shift)
    else:
        speed_ratio = math.pow(2.0, shift / 12.0)
        processed = rb.change_speed(audio, sr, speed_ratio=speed_ratio)

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({
        "output_file": str(exported),
        "semitones_applied": round(shift, 2),
        "source_key": from_key or detected_key,
        "target_key": to_key,
        "tempo_preserved": preserve_tempo,
        "duration_seconds": round(processed.shape[1] / sr, 3),
    })


@handle_tool_errors
def tool_audio_change_speed(
    input_file: str,
    output_file: str,
    speed_ratio: Optional[float] = None,
    speed_change_percent: Optional[float] = None,
    from_rpm: Optional[float] = None,
    to_rpm: Optional[float] = None,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T

    if speed_ratio is not None:
        ratio = float(speed_ratio)
    elif speed_change_percent is not None:
        ratio = float(1.0 + (speed_change_percent / 100.0))
    elif from_rpm is not None and to_rpm is not None and from_rpm > 0:
        ratio = float(to_rpm / from_rpm)
    else:
        raise ValidationError("speed_ratio", "Provide speed_ratio, speed_change_percent, or from_rpm and to_rpm.")

    rb = RubberBandEngine()
    processed = rb.change_speed(audio, sr, speed_ratio=ratio)

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({
        "output_file": str(exported),
        "speed_ratio": round(ratio, 4),
        "speed_change_percent": round((ratio - 1.0) * 100.0, 2),
        "pitch_shift_semitones": round(12.0 * math.log2(ratio), 2),
        "duration_seconds": round(processed.shape[1] / sr, 3),
    })


@handle_tool_errors
def tool_audio_paulstretch(
    input_file: str,
    output_file: str,
    stretch_factor: float = 8.0,
    window_size_sec: float = 0.25,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T

    rb = RubberBandEngine()
    processed = rb.paulstretch(audio, sr, stretch_factor=stretch_factor, window_size_sec=window_size_sec)

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({
        "output_file": str(exported),
        "stretch_factor": stretch_factor,
        "window_size_sec": window_size_sec,
        "duration_seconds": round(processed.shape[1] / sr, 3),
    })


@handle_tool_errors
def tool_audio_reverse(
    input_file: str,
    output_file: str,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = np.flip(data.T, axis=-1)

    exported = export_audio_to_file(
        audio,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({"output_file": str(exported), "duration_seconds": round(audio.shape[1] / sr, 3)})


@handle_tool_errors
def tool_audio_invert_polarity(
    input_file: str,
    output_file: str,
    channels: str = "both",
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T.copy()
    if channels == "both" or audio.ndim == 1 or audio.shape[0] == 1:
        audio = -audio
    elif channels == "left":
        audio[0] = -audio[0]
    elif channels == "right":
        audio[1] = -audio[1]

    exported = export_audio_to_file(
        audio,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({"output_file": str(exported), "inverted_channels": channels})


@handle_tool_errors
def tool_audio_truncate_silence(
    input_file: str,
    output_file: str,
    threshold_db: float = -50.0,
    min_silence_sec: float = 0.5,
    truncate_to_sec: float = 0.0,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T

    rb = RubberBandEngine()
    processed = rb.truncate_silence(
        audio,
        sr,
        threshold_db=threshold_db,
        min_silence_sec=min_silence_sec,
        truncate_to_sec=truncate_to_sec,
    )

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({
        "output_file": str(exported),
        "original_duration_seconds": round(audio.shape[1] / sr, 3),
        "final_duration_seconds": round(processed.shape[1] / sr, 3),
        "removed_seconds": round((audio.shape[1] - processed.shape[1]) / sr, 3),
    })


@handle_tool_errors
def tool_audio_denoise(
    input_file: str,
    output_file: str,
    reduction_db: float = 12.0,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T

    rb = RubberBandEngine()
    processed = rb.denoise(audio, sr, reduction_db=reduction_db)

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({"output_file": str(exported), "reduction_db": reduction_db})


@handle_tool_errors
def tool_audio_declick(
    input_file: str,
    output_file: str,
    sensitivity: float = 0.5,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T

    rb = RubberBandEngine()
    processed = rb.declick(audio, sr, sensitivity=sensitivity)

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({"output_file": str(exported), "sensitivity": sensitivity})


@handle_tool_errors
def tool_audio_vocal_remover(
    input_file: str,
    output_file: str,
    mode: str = "remove_vocals",
    low_cutoff_hz: float = 120.0,
    high_cutoff_hz: float = 8000.0,
    format: Optional[str] = None,
    bit_depth: int = 24,
) -> Dict[str, Any]:
    sess = get_active_session()
    safe_in = _resolve_audio_input(input_file, sess)
    safe_out = resolve_safe_path(output_file, sess.config)

    data, sr = sf.read(str(safe_in), always_2d=True, dtype="float32")
    audio = data.T

    rb = RubberBandEngine()
    processed = rb.vocal_remover(audio, sr, mode=mode, low_cutoff_hz=low_cutoff_hz, high_cutoff_hz=high_cutoff_hz)

    exported = export_audio_to_file(
        processed,
        safe_out,
        sample_rate=sr,
        bit_depth=bit_depth,
        audio_format=format,
    )
    return success_response({"output_file": str(exported), "mode": mode})


@handle_tool_errors
def tool_timeline_set_tempo(bpm: float, stretch_clips: bool = True) -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    old_bpm = proj.timeline.bpm
    new_bpm = proj.timeline.set_tempo(bpm, stretch_clips=stretch_clips)
    return success_response({
        "old_bpm": old_bpm,
        "new_bpm": new_bpm,
        "clips_stretched": stretch_clips,
        "total_tracks": len(proj.timeline.tracks),
    })


@handle_tool_errors
def tool_timeline_get_tempo() -> Dict[str, Any]:
    sess = get_active_session()
    proj = sess.require_project()
    return success_response({
        "bpm": proj.timeline.bpm,
        "time_signature": proj.timeline.time_signature,
    })


# ==========================================
# Jobs Tools
# ==========================================

@handle_tool_errors
def tool_job_status(job_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    job = sess.job_manager.get_job(job_id)
    return success_response(job.model_dump())


@handle_tool_errors
def tool_job_cancel(job_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    cancelled = sess.job_manager.cancel_job(job_id)
    return success_response({"job_id": job_id, "cancelled": cancelled})


@handle_tool_errors
def tool_job_result(job_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    job = sess.job_manager.get_job(job_id)
    if job.status == "completed":
        return success_response(job.result)
    elif job.status == "failed":
        return error_response(DJError("JOB_FAILED", f"Job failed: {job.error}"))
    return success_response({"status": job.status, "progress": job.progress})


@handle_tool_errors
def tool_job_logs(job_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    job = sess.job_manager.get_job(job_id)
    return success_response({"job_id": job_id, "logs": [l.model_dump() for l in job.logs]})


# ==========================================
# Sequential Workflows Tools
# ==========================================

@handle_tool_errors
def tool_workflow_create(workflow_id: str, name: str) -> Dict[str, Any]:
    sess = get_active_session()
    wf = Workflow(workflow_id=workflow_id, name=name)
    sess.workflows[workflow_id] = wf
    return success_response({"workflow_id": workflow_id, "name": name, "status": "created"})


@handle_tool_errors
def tool_workflow_add_step(
    workflow_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    condition_field: Optional[str] = None,
    condition_operator: Optional[str] = "equals",
    condition_value: Optional[Any] = None,
    on_failure: str = "stop",
) -> Dict[str, Any]:
    sess = get_active_session()
    if workflow_id not in sess.workflows:
        raise ResourceNotFoundError("workflow", workflow_id)
    wf = sess.workflows[workflow_id]
    step = wf.add_step(
        tool_name=tool_name,
        arguments=arguments,
        condition_field=condition_field,
        condition_operator=condition_operator,
        condition_value=condition_value,
        on_failure=on_failure,
    )
    return success_response({"workflow_id": workflow_id, "step_id": step.step_id, "total_steps": len(wf.steps)})


@handle_tool_errors
def tool_workflow_execute(workflow_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    res = sess.execute_workflow(workflow_id)
    return success_response(res)


@handle_tool_errors
def tool_workflow_status(workflow_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    if workflow_id not in sess.workflows:
        raise ResourceNotFoundError("workflow", workflow_id)
    wf = sess.workflows[workflow_id]
    return success_response({
        "workflow_id": wf.workflow_id,
        "name": wf.name,
        "status": wf.status,
        "current_step_idx": wf.current_step_idx,
        "total_steps": len(wf.steps),
    })


@handle_tool_errors
def tool_workflow_rollback(workflow_id: str) -> Dict[str, Any]:
    sess = get_active_session()
    sess.undo()
    return success_response({"workflow_id": workflow_id, "status": "rolled_back"})


# Central Tool Dispatcher for Workflows
TOOL_REGISTRY = {
    "system_capabilities": tool_system_capabilities,
    "available_engines": tool_available_engines,
    "available_codecs": tool_available_codecs,
    "audio_devices": tool_audio_devices,
    "available_models": tool_available_models,
    "available_plugins": tool_available_plugins,
    "project_create": tool_project_create,
    "project_open": tool_project_open,
    "project_save": tool_project_save,
    "project_close": tool_project_close,
    "project_info": tool_project_info,
    "project_validate": tool_project_validate,
    "project_snapshot": tool_project_snapshot,
    "project_restore": tool_project_restore,
    "project_undo": tool_project_undo,
    "project_redo": tool_project_redo,
    "library_scan": tool_library_scan,
    "library_search": tool_library_search,
    "library_import": tool_library_import,
    "library_remove": tool_library_remove,
    "library_metadata": tool_library_metadata,
    "track_import": tool_track_import,
    "track_analyze": tool_track_analyze,
    "track_info": tool_track_info,
    "track_waveform": tool_track_waveform,
    "track_regions": tool_track_regions,
    "track_export": tool_track_export,
    "deck_create": tool_deck_create,
    "deck_load": tool_deck_load,
    "deck_unload": tool_deck_unload,
    "deck_play": tool_deck_play,
    "deck_pause": tool_deck_pause,
    "deck_stop": tool_deck_stop,
    "deck_seek": tool_deck_seek,
    "deck_sync": tool_deck_sync,
    "deck_set_bpm": tool_deck_set_bpm,
    "deck_set_pitch": tool_deck_set_pitch,
    "deck_set_key": tool_deck_set_key,
    "detect_bpm": tool_detect_bpm,
    "detect_beats": tool_detect_beats,
    "detect_downbeats": tool_detect_downbeats,
    "create_beatgrid": tool_create_beatgrid,
    "correct_beatgrid": tool_correct_beatgrid,
    "quantize_to_grid": tool_quantize_to_grid,
    "clip_create": tool_clip_create,
    "clip_trim": tool_clip_trim,
    "clip_cut": tool_clip_cut,
    "clip_split": tool_clip_split,
    "clip_join": tool_clip_join,
    "clip_move": tool_clip_move,
    "clip_duplicate": tool_clip_duplicate,
    "clip_fade": tool_clip_fade,
    "clip_gain": tool_clip_gain,
    "clip_reverse": tool_clip_reverse,
    "clip_pitch": tool_clip_pitch,
    "clip_time_stretch": tool_clip_time_stretch,
    "clip_normalize": tool_clip_normalize,
    "clip_align": tool_clip_align,
    "cue_create": tool_cue_create,
    "cue_delete": tool_cue_delete,
    "loop_create": tool_loop_create,
    "loop_edit": tool_loop_edit,
    "loop_delete": tool_loop_delete,
    "mixer_create_channel": tool_mixer_create_channel,
    "mixer_set_gain": tool_mixer_set_gain,
    "mixer_set_pan": tool_mixer_set_pan,
    "mixer_mute": tool_mixer_mute,
    "mixer_solo": tool_mixer_solo,
    "mixer_crossfade": tool_mixer_crossfade,
    "mixer_route": tool_mixer_route,
    "mixer_create_bus": tool_mixer_create_bus,
    "mixer_create_send": tool_mixer_create_send,
    "effect_add": tool_effect_add,
    "effect_remove": tool_effect_remove,
    "effect_bypass": tool_effect_bypass,
    "effect_set_parameter": tool_effect_set_parameter,
    "effect_list": tool_effect_list,
    "effect_chain": tool_effect_chain,
    "analyze_bpm": tool_analyze_bpm,
    "analyze_key": tool_analyze_key,
    "analyze_loudness": tool_analyze_loudness,
    "analyze_dynamics": tool_analyze_dynamics,
    "analyze_spectrum": tool_analyze_spectrum,
    "analyze_phase": tool_analyze_phase,
    "analyze_clipping": tool_analyze_clipping,
    "analyze_silence": tool_analyze_silence,
    "analyze_transients": tool_analyze_transients,
    "separate_stems": tool_separate_stems,
    "list_stems": tool_list_stems,
    "stem_mute": tool_stem_mute,
    "stem_solo": tool_stem_solo,
    "stem_gain": tool_stem_gain,
    "stem_align": tool_stem_align,
    "stem_export": tool_stem_export,
    "stems_merge": tool_stems_merge,
    "audio_merge": tool_audio_merge,
    "master_analyze": tool_master_analyze,
    "master_process": tool_master_process,
    "master_normalize": tool_master_normalize,
    "master_limit": tool_master_limit,
    "master_validate": tool_master_validate,
    "render_preview": tool_render_preview,
    "render_region": tool_render_region,
    "render_track": tool_render_track,
    "render_stems": tool_render_stems,
    "render_project": tool_render_project,
    "export_audio": tool_export_audio,
    "audio_change_tempo": tool_audio_change_tempo,
    "audio_change_pitch": tool_audio_change_pitch,
    "audio_change_speed": tool_audio_change_speed,
    "audio_paulstretch": tool_audio_paulstretch,
    "audio_reverse": tool_audio_reverse,
    "audio_invert_polarity": tool_audio_invert_polarity,
    "audio_truncate_silence": tool_audio_truncate_silence,
    "audio_denoise": tool_audio_denoise,
    "audio_declick": tool_audio_declick,
    "audio_vocal_remover": tool_audio_vocal_remover,
    "timeline_set_tempo": tool_timeline_set_tempo,
    "timeline_get_tempo": tool_timeline_get_tempo,
    "job_status": tool_job_status,
    "job_cancel": tool_job_cancel,
    "job_result": tool_job_result,
    "job_logs": tool_job_logs,
    "workflow_create": tool_workflow_create,
    "workflow_add_step": tool_workflow_add_step,
    "workflow_execute": tool_workflow_execute,
    "workflow_status": tool_workflow_status,
    "workflow_rollback": tool_workflow_rollback,
}


def dispatch_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOL_REGISTRY.get(tool_name)
    if not fn:
        raise ResourceNotFoundError("tool", tool_name)
    return fn(**args)


# Register dispatcher on session initialization
get_active_session().register_tool_dispatcher(dispatch_tool)
