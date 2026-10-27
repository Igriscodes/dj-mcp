from ai_dj.analysis import AudioAnalyzer
from ai_dj.beatgrid import BeatGrid
from ai_dj.key import check_key_compatibility, get_compatible_camelot_keys, key_to_camelot


def test_key_detection_and_camelot_compatibility():
    assert key_to_camelot("A minor") == "8A"
    assert key_to_camelot("C major") == "8B"
    assert key_to_camelot("E minor") == "9A"
    assert key_to_camelot("D minor") == "7A"

    comp = check_key_compatibility("8A", "8B")
    assert comp["compatible"] is True
    assert comp["relationship"] == "relative_major_minor"

    comp_adj = check_key_compatibility("8A", "9A")
    assert comp_adj["compatible"] is True
    assert comp_adj["relationship"] == "adjacent_harmonic"


def test_beatgrid_quantization():
    # 120 BPM = 0.5 sec per beat, 2.0 sec per 4/4 bar
    bg = BeatGrid.generate_constant_grid(bpm=120.0, first_beat_sec=0.0, duration_sec=10.0)
    assert len(bg.beat_positions) > 0
    assert bg.beat_interval_sec == 0.5

    # Quantize to nearest beat
    assert bg.quantize_position(0.48, quantum="beat") == 0.5
    assert bg.quantize_position(0.52, quantum="beat") == 0.5

    # Quantize to bar (every 2.0s)
    assert bg.quantize_position(1.9, quantum="bar") == 2.0

    # Nudge grid
    bg.shift_grid(0.05)
    assert bg.first_beat_sec == 0.05
    assert bg.beat_positions[0] == 0.05


def test_audio_analysis_on_fixture(test_audio_file):
    analyzer = AudioAnalyzer()
    res = analyzer.analyze_track(test_audio_file)
    assert res["bpm"] > 0
    assert "key" in res
    assert "camelot" in res
    assert "loudness" in res
    assert res["loudness"]["peak_dbfs"] <= 0.0
