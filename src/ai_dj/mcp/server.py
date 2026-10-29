import argparse
import sys
from fastmcp import FastMCP
from ai_dj.mcp.tools import (
    tool_system_capabilities,
    tool_available_engines,
    tool_available_codecs,
    tool_audio_devices,
    tool_available_models,
    tool_available_plugins,
    tool_workspace_get,
    tool_workspace_set,
    tool_project_create,
    tool_project_open,
    tool_project_save,
    tool_project_close,
    tool_project_info,
    tool_project_validate,
    tool_project_snapshot,
    tool_project_restore,
    tool_project_undo,
    tool_project_redo,
    tool_library_scan,
    tool_library_search,
    tool_library_import,
    tool_library_remove,
    tool_library_metadata,
    tool_track_import,
    tool_track_analyze,
    tool_track_info,
    tool_track_waveform,
    tool_track_regions,
    tool_track_export,
    tool_deck_create,
    tool_deck_load,
    tool_deck_unload,
    tool_deck_play,
    tool_deck_pause,
    tool_deck_stop,
    tool_deck_seek,
    tool_deck_sync,
    tool_deck_set_bpm,
    tool_deck_set_pitch,
    tool_deck_set_key,
    tool_detect_bpm,
    tool_detect_beats,
    tool_detect_downbeats,
    tool_create_beatgrid,
    tool_correct_beatgrid,
    tool_quantize_to_grid,
    tool_clip_create,
    tool_clip_trim,
    tool_clip_cut,
    tool_clip_split,
    tool_clip_join,
    tool_clip_move,
    tool_clip_duplicate,
    tool_clip_fade,
    tool_clip_gain,
    tool_clip_reverse,
    tool_clip_pitch,
    tool_clip_time_stretch,
    tool_clip_normalize,
    tool_clip_align,
    tool_cue_create,
    tool_cue_delete,
    tool_loop_create,
    tool_loop_edit,
    tool_loop_delete,
    tool_mixer_create_channel,
    tool_mixer_set_gain,
    tool_mixer_set_pan,
    tool_mixer_mute,
    tool_mixer_solo,
    tool_mixer_crossfade,
    tool_mixer_route,
    tool_mixer_create_bus,
    tool_mixer_create_send,
    tool_effect_add,
    tool_effect_remove,
    tool_effect_bypass,
    tool_effect_set_parameter,
    tool_effect_list,
    tool_effect_chain,
    tool_analyze_bpm,
    tool_analyze_key,
    tool_analyze_loudness,
    tool_analyze_dynamics,
    tool_analyze_spectrum,
    tool_analyze_phase,
    tool_analyze_clipping,
    tool_analyze_silence,
    tool_analyze_transients,
    tool_separate_stems,
    tool_list_stems,
    tool_stem_mute,
    tool_stem_solo,
    tool_stem_gain,
    tool_stem_align,
    tool_stem_export,
    tool_stems_merge,
    tool_audio_merge,
    tool_master_analyze,
    tool_master_process,
    tool_master_normalize,
    tool_master_limit,
    tool_master_validate,
    tool_render_preview,
    tool_render_region,
    tool_render_track,
    tool_render_stems,
    tool_render_project,
    tool_export_audio,
    tool_audio_convert_format,
    tool_audio_apply_effect,
    tool_audio_reverb,
    tool_audio_change_tempo,
    tool_audio_change_pitch,
    tool_audio_change_speed,
    tool_audio_paulstretch,
    tool_audio_reverse,
    tool_audio_invert_polarity,
    tool_audio_truncate_silence,
    tool_audio_denoise,
    tool_audio_declick,
    tool_audio_vocal_remover,
    tool_timeline_set_tempo,
    tool_timeline_get_tempo,
    tool_job_status,
    tool_job_cancel,
    tool_job_result,
    tool_job_logs,
    tool_workflow_create,
    tool_workflow_add_step,
    tool_workflow_execute,
    tool_workflow_status,
    tool_workflow_rollback,
)


def create_mcp_server() -> FastMCP:
    server = FastMCP(name="dj-mcp")

    # Capabilities
    server.tool(name="system_capabilities", description="Inspect system capabilities, CPU, RAM, GPU, audio backends, and codecs.")(tool_system_capabilities)
    server.tool(name="available_engines", description="List available DSP, separation, and analysis engines.")(tool_available_engines)
    server.tool(name="available_codecs", description="List supported audio containers and codecs.")(tool_available_codecs)
    server.tool(name="audio_devices", description="List active audio playback/record devices.")(tool_audio_devices)
    server.tool(name="available_models", description="List installed neural separation models.")(tool_available_models)
    server.tool(name="available_plugins", description="List available DSP effects and host plugins.")(tool_available_plugins)
    server.tool(name="workspace_get", description="Get current active workspace directory, free disk space, and export defaults.")(tool_workspace_get)
    server.tool(name="workspace_set", description="Set or change the active workspace directory where audio, projects, and stems are saved.")(tool_workspace_set)

    # Project
    server.tool(name="project_create", description="Create a new music production or DJ project.")(tool_project_create)
    server.tool(name="project_open", description="Open an existing project from a .djproj file.")(tool_project_open)
    server.tool(name="project_save", description="Save the current project to disk atomically.")(tool_project_save)
    server.tool(name="project_close", description="Close the current project.")(tool_project_close)
    server.tool(name="project_info", description="Get full status of the active project, tracks, and decks.")(tool_project_info)
    server.tool(name="project_validate", description="Validate project integrity, missing files, and routing.")(tool_project_validate)
    server.tool(name="project_snapshot", description="Create an undo/recovery snapshot.")(tool_project_snapshot)
    server.tool(name="project_restore", description="Restore project to a specific snapshot ID.")(tool_project_restore)
    server.tool(name="project_undo", description="Undo the last operation.")(tool_project_undo)
    server.tool(name="project_redo", description="Redo the last undone operation.")(tool_project_redo)

    # Library
    server.tool(name="library_scan", description="Scan a directory for audio files and import metadata.")(tool_library_scan)
    server.tool(name="library_search", description="Search audio library by title, artist, BPM range, or key.")(tool_library_search)
    server.tool(name="library_import", description="Import an audio file into library.")(tool_library_import)
    server.tool(name="library_remove", description="Remove a track from library.")(tool_library_remove)
    server.tool(name="library_metadata", description="Inspect audio file metadata tags and format info.")(tool_library_metadata)

    # Tracks
    server.tool(name="track_import", description="Import audio file as a project track.")(tool_track_import)
    server.tool(name="track_analyze", description="Run comprehensive audio analysis (BPM, key, beats, loudness).")(tool_track_analyze)
    server.tool(name="track_info", description="Get detailed information for a track.")(tool_track_info)
    server.tool(name="track_waveform", description="Get 3-band frequency visual waveform peaks.")(tool_track_waveform)
    server.tool(name="track_regions", description="List regions inside a track.")(tool_track_regions)
    server.tool(name="track_export", description="Export a single track to an audio file.")(tool_track_export)

    # Decks
    server.tool(name="deck_create", description="Create an additional DJ deck.")(tool_deck_create)
    server.tool(name="deck_load", description="Load an analyzed track onto a DJ deck.")(tool_deck_load)
    server.tool(name="deck_unload", description="Unload track from deck.")(tool_deck_unload)
    server.tool(name="deck_play", description="Start playback on deck.")(tool_deck_play)
    server.tool(name="deck_pause", description="Pause playback on deck.")(tool_deck_pause)
    server.tool(name="deck_stop", description="Stop playback and rewind to 0.")(tool_deck_stop)
    server.tool(name="deck_seek", description="Seek to position in seconds (optionally beat-quantized).")(tool_deck_seek)
    server.tool(name="deck_sync", description="Synchronize deck BPM and beat-phase to master deck.")(tool_deck_sync)
    server.tool(name="deck_set_bpm", description="Set explicit BPM on deck.")(tool_deck_set_bpm)
    server.tool(name="deck_set_pitch", description="Shift pitch fader percent on deck.")(tool_deck_set_pitch)
    server.tool(name="deck_set_key", description="Shift musical key in semitones without changing tempo.")(tool_deck_set_key)

    # Beat Grid
    server.tool(name="detect_bpm", description="Detect BPM and beat positions from audio file.")(tool_detect_bpm)
    server.tool(name="detect_beats", description="Detect beat timestamps.")(tool_detect_beats)
    server.tool(name="detect_downbeats", description="Detect bar downbeats (measure starts).")(tool_detect_downbeats)
    server.tool(name="create_beatgrid", description="Generate a beatgrid for a track.")(tool_create_beatgrid)
    server.tool(name="correct_beatgrid", description="Nudge or correct beatgrid offset and BPM.")(tool_correct_beatgrid)
    server.tool(name="quantize_to_grid", description="Quantize timestamp to nearest bar or beat division.")(tool_quantize_to_grid)

    # Editing
    server.tool(name="clip_create", description="Create an audio clip on timeline.")(tool_clip_create)
    server.tool(name="clip_trim", description="Trim clip boundaries non-destructively.")(tool_clip_trim)
    server.tool(name="clip_cut", description="Remove a clip from track.")(tool_clip_cut)
    server.tool(name="clip_split", description="Split clip into two adjacent clips at offset.")(tool_clip_split)
    server.tool(name="clip_join", description="Join two adjacent clips into one.")(tool_clip_join)
    server.tool(name="clip_move", description="Move clip to new timeline position.")(tool_clip_move)
    server.tool(name="clip_duplicate", description="Duplicate a clip.")(tool_clip_duplicate)
    server.tool(name="clip_fade", description="Set fade-in and fade-out on clip.")(tool_clip_fade)
    server.tool(name="clip_gain", description="Set gain in dB on clip.")(tool_clip_gain)
    server.tool(name="clip_reverse", description="Reverse audio in clip.")(tool_clip_reverse)
    server.tool(name="clip_pitch", description="Pitch shift clip in semitones.")(tool_clip_pitch)
    server.tool(name="clip_time_stretch", description="Time-stretch clip playback rate.")(tool_clip_time_stretch)
    server.tool(name="clip_normalize", description="Normalize clip peak gain.")(tool_clip_normalize)
    server.tool(name="clip_align", description="Align clip to specific timeline time.")(tool_clip_align)

    # Loops & Cues
    server.tool(name="cue_create", description="Set a hot cue on deck.")(tool_cue_create)
    server.tool(name="cue_delete", description="Delete hot cue from deck.")(tool_cue_delete)
    server.tool(name="loop_create", description="Create an active loop of N beats on deck.")(tool_loop_create)
    server.tool(name="loop_edit", description="Modify loop (double, halve, shift).")(tool_loop_edit)
    server.tool(name="loop_delete", description="Clear active loop on deck.")(tool_loop_delete)

    # Mixing
    server.tool(name="mixer_create_channel", description="Create a channel strip in mixer.")(tool_mixer_create_channel)
    server.tool(name="mixer_set_gain", description="Set channel volume gain in dB.")(tool_mixer_set_gain)
    server.tool(name="mixer_set_pan", description="Set channel pan [-1.0 to +1.0].")(tool_mixer_set_pan)
    server.tool(name="mixer_mute", description="Mute or unmute channel.")(tool_mixer_mute)
    server.tool(name="mixer_solo", description="Solo or unsolo channel.")(tool_mixer_solo)
    server.tool(name="mixer_crossfade", description="Set crossfader position and curve.")(tool_mixer_crossfade)
    server.tool(name="mixer_route", description="Route channel to crossfader side A, B, or thru.")(tool_mixer_route)
    server.tool(name="mixer_create_bus", description="Create sub-mix bus.")(tool_mixer_create_bus)
    server.tool(name="mixer_create_send", description="Create send routing from track to bus.")(tool_mixer_create_send)

    # Effects
    server.tool(name="effect_add", description="Add effect to channel strip.")(tool_effect_add)
    server.tool(name="effect_remove", description="Remove effect from channel strip.")(tool_effect_remove)
    server.tool(name="effect_bypass", description="Bypass or enable effect.")(tool_effect_bypass)
    server.tool(name="effect_set_parameter", description="Set parameter value on effect.")(tool_effect_set_parameter)
    server.tool(name="effect_list", description="List active effects on channel.")(tool_effect_list)
    server.tool(name="effect_chain", description="Get entire effect chain on channel.")(tool_effect_chain)

    # Analysis
    server.tool(name="analyze_bpm", description="Analyze BPM and beat grid of audio file.")(tool_analyze_bpm)
    server.tool(name="analyze_key", description="Analyze musical key and Camelot notation.")(tool_analyze_key)
    server.tool(name="analyze_loudness", description="Analyze integrated LUFS, peak, true peak, and RMS.")(tool_analyze_loudness)
    server.tool(name="analyze_dynamics", description="Analyze crest factor and dynamic range.")(tool_analyze_dynamics)
    server.tool(name="analyze_spectrum", description="Analyze spectral centroid and FFT magnitude.")(tool_analyze_spectrum)
    server.tool(name="analyze_phase", description="Analyze stereo phase correlation.")(tool_analyze_phase)
    server.tool(name="analyze_clipping", description="Analyze clipped samples and peaks.")(tool_analyze_clipping)
    server.tool(name="analyze_silence", description="Detect silence intervals in audio.")(tool_analyze_silence)
    server.tool(name="analyze_transients", description="Detect transient onset timestamps.")(tool_analyze_transients)

    # Stems
    server.tool(name="separate_stems", description="Run neural stem separation (vocals, drums, bass, other) on a single song, multiple songs, or folder. Output stems are saved directly next to the audio file (in stems_<track_name>) or in output_dir. Never saves to /tmp/.")(tool_separate_stems)
    server.tool(name="list_stems", description="List stems in a stem set.")(tool_list_stems)
    server.tool(name="stem_mute", description="Mute or unmute specific stem.")(tool_stem_mute)
    server.tool(name="stem_solo", description="Solo specific stem.")(tool_stem_solo)
    server.tool(name="stem_gain", description="Set gain in dB on specific stem.")(tool_stem_gain)
    server.tool(name="stem_align", description="Time-align stem with offset.")(tool_stem_align)
    server.tool(name="stem_export", description="Recombine and export stems to file.")(tool_stem_export)
    server.tool(name="stems_merge", description="Recombine and merge all stems from a directory into a single audio file.")(tool_stems_merge)
    server.tool(name="audio_merge", description="Mix and merge multiple audio files or stems together with anti-clipping protection.")(tool_audio_merge)

    # Mastering
    server.tool(name="master_analyze", description="Analyze loudness against mastering standards.")(tool_master_analyze)
    server.tool(name="master_process", description="Master audio file using target profile (streaming, dj_master, cd, club).")(tool_master_process)
    server.tool(name="master_normalize", description="Normalize audio to specific integrated LUFS target.")(tool_master_normalize)
    server.tool(name="master_limit", description="Apply true peak limiting to ceiling dBTP.")(tool_master_limit)
    server.tool(name="master_validate", description="Validate master compliance against target LUFS and true peak ceiling.")(tool_master_validate)

    # Rendering
    server.tool(name="render_preview", description="Render a fast short preview snippet.")(tool_render_preview)
    server.tool(name="render_region", description="Render specific region between start and end seconds.")(tool_render_region)
    server.tool(name="render_track", description="Render isolated track.")(tool_render_track)
    server.tool(name="render_stems", description="Render isolated stems mix.")(tool_render_stems)
    server.tool(name="render_project", description="Render entire project mix to audio file (MP3, WAV, FLAC, etc.).")(tool_render_project)
    server.tool(name="export_audio", description="Transcode and export audio file to target format and bit depth.")(tool_export_audio)
    server.tool(name="audio_convert_format", description="Convert single audio files or entire directories to MP3 (default), FLAC, WAV, OGG, AAC, OPUS, or AIFF with customizable bitrate and storage savings analysis.")(tool_audio_convert_format)
    server.tool(name="audio_apply_effect", description="Apply pedalboard effects (reverb, delay, eq, compressor, limiter, chorus, distortion, highpass, lowpass) to single file, list of files, or folder.")(tool_audio_apply_effect)
    server.tool(name="audio_reverb", description="Apply reverb effect with custom room size, damping, and wet/dry mix to single file, multiple mentioned songs, or an entire folder.")(tool_audio_reverb)

    # Standalone Audio & Pitch/Tempo Editing (Audacity-Style Direct DSP)
    server.tool(name="audio_change_tempo", description="Change tempo of an audio file or track (BPM, percent, duration) with pitch preservation.")(tool_audio_change_tempo)
    server.tool(name="audio_change_pitch", description="Shift pitch of an audio file or track (semitones, cents, musical key) with tempo preservation.")(tool_audio_change_pitch)
    server.tool(name="audio_change_speed", description="Change speed and pitch together (varispeed, tape speed, vinyl 33/45 RPM).")(tool_audio_change_speed)
    server.tool(name="audio_paulstretch", description="Apply extreme Paulstretch time-stretching for lush ambient/drone sound design.")(tool_audio_paulstretch)
    server.tool(name="audio_reverse", description="Reverse an audio file or track backward in time.")(tool_audio_reverse)
    server.tool(name="audio_invert_polarity", description="Invert phase/polarity of audio channels.")(tool_audio_invert_polarity)
    server.tool(name="audio_truncate_silence", description="Detect and truncate/strip silent intervals from an audio file.")(tool_audio_truncate_silence)
    server.tool(name="audio_denoise", description="Reduce background noise using adaptive noise gating.")(tool_audio_denoise)
    server.tool(name="audio_declick", description="Detect and remove clicks, vinyl crackle, and transient dropouts.")(tool_audio_declick)
    server.tool(name="audio_vocal_remover", description="Isolate or remove vocals using center-channel extraction with bass preservation.")(tool_audio_vocal_remover)
    server.tool(name="timeline_set_tempo", description="Set master project/timeline tempo in BPM with optional clip re-timing.")(tool_timeline_set_tempo)
    server.tool(name="timeline_get_tempo", description="Get master project/timeline tempo and time signature.")(tool_timeline_get_tempo)

    # Jobs
    server.tool(name="job_status", description="Check status and progress of a background job.")(tool_job_status)
    server.tool(name="job_cancel", description="Cancel a running background job.")(tool_job_cancel)
    server.tool(name="job_result", description="Get output result of a completed background job.")(tool_job_result)
    server.tool(name="job_logs", description="Get execution logs of a background job.")(tool_job_logs)

    # Workflows
    server.tool(name="workflow_create", description="Create a sequential multi-step workflow.")(tool_workflow_create)
    server.tool(name="workflow_add_step", description="Add a conditional step to workflow.")(tool_workflow_add_step)
    server.tool(name="workflow_execute", description="Execute workflow and return results.")(tool_workflow_execute)
    server.tool(name="workflow_status", description="Check status of a workflow.")(tool_workflow_status)
    server.tool(name="workflow_rollback", description="Rollback workflow changes.")(tool_workflow_rollback)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="DJ MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"], help="MCP transport protocol (default: stdio)")
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE server")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE server")
    args = parser.parse_args()

    server = create_mcp_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
    elif args.transport == "sse":
        server.run(transport="sse", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
