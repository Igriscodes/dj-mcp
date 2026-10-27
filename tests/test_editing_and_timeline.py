import numpy as np
from ai_dj.clips import AudioClip
from ai_dj.timeline import Timeline
from ai_dj.tracks import Track


def test_clip_trim_and_split(test_audio_file):
    clip = AudioClip(
        clip_id="c1",
        name="test_clip",
        source_file=test_audio_file,
        source_offset_sec=0.0,
        timeline_position_sec=0.0,
        duration_sec=3.0,
    )
    # Trim
    clip.trim(start_delta_sec=0.5, end_delta_sec=0.5)
    assert clip.source_offset_sec == 0.5
    assert clip.duration_sec == 2.0

    # Split
    c1, c2 = clip.split(split_offset_sec=1.0, new_clip_id="c2")
    assert c1.duration_sec == 1.0
    assert c2.duration_sec == 1.0
    assert c2.timeline_position_sec == 1.0
    assert c2.source_offset_sec == 1.5


def test_track_and_timeline_rendering(test_audio_file):
    track = Track(track_id="t1", name="Track 1", channel_id="ch_1")
    clip = AudioClip(
        clip_id="c1",
        name="test_clip",
        source_file=test_audio_file,
        source_offset_sec=0.0,
        timeline_position_sec=0.0,
        duration_sec=2.0,
    )
    track.add_clip(clip)

    audio = track.render_track_audio(target_sample_rate=44100)
    assert audio.shape[0] == 2
    assert audio.shape[1] == 44100 * 2
    assert np.max(np.abs(audio)) > 0.0
