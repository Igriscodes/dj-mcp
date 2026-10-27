import pytest
import numpy as np
from ai_dj.beatgrid import BeatGrid
from ai_dj.decks import Deck
from ai_dj.errors import DJError, PrerequisiteError
from ai_dj.mixer import calculate_crossfader_gains


def test_deck_lifecycle(test_audio_file):
    deck = Deck(deck_id="deck_a")

    # Unloaded deck raises PrerequisiteError
    with pytest.raises(PrerequisiteError):
        deck.play()

    bg = BeatGrid.generate_constant_grid(bpm=120.0, first_beat_sec=0.0, duration_sec=4.0)
    deck.load_track(track_id="test_track", audio_path=test_audio_file, bpm=120.0, beatgrid=bg)

    assert deck.state == "stopped"
    deck.play()
    assert deck.state == "playing"
    deck.pause()
    assert deck.state == "paused"

    # Seeking with quantization
    new_pos = deck.seek(0.48, quantize=True)
    assert new_pos == 0.5


def test_deck_sync_and_pitch():
    deck_a = Deck(deck_id="deck_a")
    deck_a.load_track("t1", "a.wav", bpm=124.0)

    deck_b = Deck(deck_id="deck_b")
    deck_b.load_track("t2", "b.wav", bpm=128.0)

    # Sync deck B to Deck A
    deck_b.sync_to(target_bpm=deck_a.current_bpm)
    assert deck_b.current_bpm == 124.0
    assert abs(deck_b.pitch_percent - (((124.0 - 128.0) / 128.0) * 100.0)) < 1e-3


def test_cues_and_loops():
    deck = Deck(deck_id="deck_a")
    deck.load_track("t1", "a.wav", bpm=120.0)

    # Set hot cue 1
    cue = deck.set_cue(1, position_sec=2.0)
    assert cue.cue_index == 1
    assert cue.position_sec == 2.0

    deck.jump_to_cue(1)
    assert deck.position_sec == 2.0

    # Create 4-beat loop (2.0s at 120BPM)
    loop = deck.set_loop(length_beats=4.0, start_sec=2.0)
    assert loop.start_sec == 2.0
    assert loop.end_sec == 4.0
    assert loop.duration_sec == 2.0


def test_crossfader_curves():
    # Center position: (0.0)
    ga_eq, gb_eq = calculate_crossfader_gains(0.0, curve="equal_power")
    assert abs(ga_eq - 0.7071) < 1e-2
    assert abs(gb_eq - 0.7071) < 1e-2

    # Left position: (-1.0)
    ga_l, gb_l = calculate_crossfader_gains(-1.0, curve="equal_power")
    assert abs(ga_l - 1.0) < 1e-3
    assert abs(gb_l - 0.0) < 1e-3

    # Linear curve
    ga_lin, gb_lin = calculate_crossfader_gains(0.0, curve="linear")
    assert ga_lin == 0.5
    assert gb_lin == 0.5


def test_deck_audio_rendering_and_project_mix(tmp_path, test_audio_file):
    import pytest
    from ai_dj.project import Project
    from ai_dj.rendering import ProjectRenderer
    from ai_dj.errors import PrerequisiteError

    # 1. Deck audio rendering
    deck = Deck(deck_id="deck_a")
    deck.load_track("track_1", test_audio_file, bpm=120.0)
    deck.play()
    deck.set_bpm(130.0)  # Time stretch
    deck.set_loop(length_beats=4.0, start_sec=0.5)

    audio = deck.render_deck_audio(duration_sec=2.0, target_sample_rate=44100)
    assert audio.shape == (2, 44100 * 2)
    assert np.max(np.abs(audio)) > 0.0

    # 2. Project rendering with decks and empty timeline
    proj = Project.create_new("p_dj", "DJ Set", sample_rate=44100, num_decks=2)
    proj.decks["deck_a"].load_track("track_1", test_audio_file, bpm=120.0)
    proj.decks["deck_a"].play()

    renderer = ProjectRenderer()
    mix = renderer.render_timeline(
        timeline=proj.timeline,
        mixer=proj.mixer,
        decks=proj.decks,
        start_sec=0.0,
        end_sec=2.0,
        sample_rate=44100,
    )
    assert mix.shape == (2, 44100 * 2)
    assert np.max(np.abs(mix)) > 0.0

    # 3. PrerequisiteError when nothing to render
    empty_proj = Project.create_new("p_empty", "Empty Set")
    with pytest.raises(PrerequisiteError) as exc_info:
        renderer.render_timeline(
            timeline=empty_proj.timeline,
            mixer=empty_proj.mixer,
            decks=empty_proj.decks,
            start_sec=0.0,
            end_sec=2.0,
        )
    assert exc_info.value.code == "NOTHING_TO_RENDER"
