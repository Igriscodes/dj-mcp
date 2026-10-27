import numpy as np
import soundfile as sf
from ai_dj.clips import AudioClip
from ai_dj.export import export_audio_to_file
from ai_dj.loudness import measure_lufs
from ai_dj.mastering import MasteringProcessor
from ai_dj.mixer import Mixer, MixerChannel
from ai_dj.rendering import ProjectRenderer
from ai_dj.timeline import Timeline
from ai_dj.tracks import Track


def test_mastering_processor():
    sr = 44100
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    # Generate test sine
    sig = np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 440 * t)]).astype(np.float32)

    masterer = MasteringProcessor(profile="streaming", target_lufs=-14.0, true_peak_ceiling_db=-1.0)
    mastered, stats = masterer.process_audio(sig, sr)

    assert stats["profile"] == "streaming"
    assert stats["final_peak_dbfs"] <= -0.9
    assert abs(stats["final_lufs"] - (-14.0)) <= 1.5


def test_project_rendering_and_export(test_audio_file, tmp_path):
    timeline = Timeline()
    track = Track(track_id="t1", name="Track 1", channel_id="ch_1")
    clip = AudioClip(
        clip_id="c1",
        name="Clip 1",
        source_file=test_audio_file,
        timeline_position_sec=0.0,
        duration_sec=2.0,
    )
    track.add_clip(clip)
    timeline.add_track(track)

    mixer = Mixer()
    mixer.add_channel(MixerChannel(channel_id="ch_1", name="Ch 1", gain_db=0.0))

    renderer = ProjectRenderer()
    mix = renderer.render_timeline(timeline, mixer, start_sec=0.0, end_sec=2.0)
    assert mix.shape[0] == 2
    assert mix.shape[1] == 44100 * 2

    # Export to WAV
    wav_out = tmp_path / "render.wav"
    export_audio_to_file(mix, wav_out, sample_rate=44100, bit_depth=24)
    assert wav_out.exists()
    info = sf.info(str(wav_out))
    assert info.channels == 2
    assert info.samplerate == 44100

    # Export to FLAC
    flac_out = tmp_path / "render.flac"
    export_audio_to_file(mix, flac_out, sample_rate=44100, bit_depth=16)
    assert flac_out.exists()
