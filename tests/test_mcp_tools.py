from ai_dj.mcp.tools import (
    tool_analyze_bpm,
    tool_analyze_key,
    tool_analyze_loudness,
    tool_clip_create,
    tool_clip_pitch,
    tool_clip_time_stretch,
    tool_clip_trim,
    tool_create_beatgrid,
    tool_cue_create,
    tool_deck_create,
    tool_deck_load,
    tool_deck_play,
    tool_deck_seek,
    tool_deck_set_bpm,
    tool_deck_sync,
    tool_effect_add,
    tool_export_audio,
    tool_library_import,
    tool_loop_create,
    tool_master_process,
    tool_master_validate,
    tool_mixer_crossfade,
    tool_mixer_set_gain,
    tool_project_create,
    tool_project_info,
    tool_project_save,
    tool_project_validate,
    tool_render_preview,
    tool_render_project,
    tool_separate_stems,
    tool_stem_gain,
    tool_stem_mute,
    tool_stem_solo,
    tool_system_capabilities,
    tool_track_analyze,
    tool_track_waveform,
)


def test_end_to_end_mcp_tool_workflow(test_audio_file, tmp_path):
    # 1. Capabilities
    caps = tool_system_capabilities()
    assert caps["success"] is True

    # 2. Create project
    p_res = tool_project_create(project_id="dj_set_1", name="DJ Set 2026", sample_rate=44100, num_decks=2)
    assert p_res["success"] is True

    # 3. Import user audio
    imp = tool_library_import(file_path=str(test_audio_file), track_id="track_a")
    assert imp["success"] is True

    # 4. Analyze tracks
    ana = tool_track_analyze(track_id="track_a")
    assert ana["success"] is True
    assert ana["result"]["bpm"] > 0
    assert "key" in ana["result"]

    # 5. Detect BPM / Key / Beatgrid
    bg = tool_create_beatgrid(track_id="track_a", bpm=ana["result"]["bpm"], first_beat_sec=ana["result"]["first_beat_sec"])
    assert bg["success"] is True

    # 6. Decks: load and play
    l_res = tool_deck_load(deck_id="deck_a", track_id="track_a")
    assert l_res["success"] is True

    play_res = tool_deck_play(deck_id="deck_a")
    assert play_res["success"] is True
    assert play_res["result"]["state"] == "playing"

    # 7. Seeking and Sync
    seek_res = tool_deck_seek(deck_id="deck_a", position_sec=1.0, quantize=True)
    assert seek_res["success"] is True

    # Deck B setup and sync
    tool_deck_create(deck_id="deck_b")
    tool_deck_load(deck_id="deck_b", track_id="track_a")
    tool_deck_set_bpm(deck_id="deck_b", bpm=130.0)
    sync_res = tool_deck_sync(deck_id="deck_b", master_deck_id="deck_a")
    assert sync_res["success"] is True
    assert sync_res["result"]["synced_bpm"] == ana["result"]["bpm"]

    # 8. Cues and Loops
    cue_res = tool_cue_create(deck_id="deck_a", cue_index=1, position_sec=2.0)
    assert cue_res["success"] is True

    loop_res = tool_loop_create(deck_id="deck_a", length_beats=4.0, start_sec=2.0)
    assert loop_res["success"] is True

    # 9. Editing clips on timeline
    clip_res = tool_clip_create(
        track_id="track_main",
        clip_id="c_main",
        source_file=str(test_audio_file),
        timeline_position_sec=0.0,
        duration_sec=3.0,
    )
    assert clip_res["success"] is True

    trim_res = tool_clip_trim(track_id="track_main", clip_id="c_main", start_delta_sec=0.5)
    assert trim_res["success"] is True
    assert trim_res["result"]["source_offset_sec"] == 0.5

    # 10. Time stretch & 11. Pitch shift
    ts_res = tool_clip_time_stretch(track_id="track_main", clip_id="c_main", rate=1.2)
    assert ts_res["success"] is True

    ps_res = tool_clip_pitch(track_id="track_main", clip_id="c_main", semitones=2.0)
    assert ps_res["success"] is True

    # 12. Mix tracks
    xfade_res = tool_mixer_crossfade(position=0.2, curve="equal_power")
    assert xfade_res["success"] is True
    assert "gain_deck_a" in xfade_res["result"]

    gain_res = tool_mixer_set_gain(channel_id="ch_a", gain_db=-2.0)
    assert gain_res["success"] is True

    # 13. Effects
    eff_res = tool_effect_add(
        channel_id="ch_a",
        effect_id="dj_eq",
        effect_type="eq",
        parameters={"low_gain_db": -3.0, "high_gain_db": 1.5},
    )
    assert eff_res["success"] is True

    # 14. Render project
    render_out = str(tmp_path / "project_mix.wav")
    rend_res = tool_render_project(output_path=render_out, start_sec=0.0, end_sec=2.0)
    assert rend_res["success"] is True
    assert rend_res["result"]["duration_seconds"] == 2.0

    # 15. Analyze mix
    loud_res = tool_analyze_loudness(file_path=render_out)
    assert loud_res["success"] is True

    # 16. Mastering
    master_out = str(tmp_path / "project_master.wav")
    mast_res = tool_master_process(input_file=render_out, output_file=master_out, profile="streaming")
    assert mast_res["success"] is True
    assert mast_res["result"]["stats"]["final_peak_dbfs"] <= -0.9

    val_res = tool_master_validate(file_path=master_out, target_profile="streaming")
    assert val_res["success"] is True

    # 17. Export audio
    flac_out = str(tmp_path / "final_export.flac")
    exp_res = tool_export_audio(input_file=master_out, output_file=flac_out, format="flac")
    assert exp_res["success"] is True

    # 18. Structured Error Recovery for Small LLMs
    bad_res = tool_deck_play(deck_id="non_existent_deck")
    assert bad_res["success"] is False
    assert "error" in bad_res
    assert bad_res["error"]["code"] == "DECK_NOT_FOUND"
    assert bad_res["error"]["recoverable"] is True
    assert len(bad_res["next_actions"]) > 0
    assert bad_res["next_actions"][0]["tool"] == "deck_create"
