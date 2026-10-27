import numpy as np
from ai_dj.effects import EffectChain, EffectConfig
from ai_dj.engines.rubberband_engine import RubberBandEngine
from ai_dj.stereo import apply_panning, apply_stereo_width, calculate_phase_correlation, remove_dc_offset


def test_stereo_panning():
    mono = np.ones((1, 100), dtype=np.float32)
    # Hard left
    left = apply_panning(mono, pan=-1.0)
    assert np.allclose(left[1, :], 0.0, atol=1e-3)
    assert left[0, 0] > 1.0

    # Hard right
    right = apply_panning(mono, pan=1.0)
    assert np.allclose(right[0, :], 0.0, atol=1e-3)
    assert right[1, 0] > 1.0


def test_stereo_width_and_correlation():
    # Identical channels = correlation 1.0
    stereo = np.ones((2, 100), dtype=np.float32)
    assert calculate_phase_correlation(stereo) == 1.0

    # Inverted right channel = correlation -1.0
    out_of_phase = np.stack([np.ones(100), -np.ones(100)]).astype(np.float32)
    assert calculate_phase_correlation(out_of_phase) == -1.0

    # Mid/side mono collapse
    collapsed = apply_stereo_width(out_of_phase, width=0.0)
    assert np.allclose(collapsed, 0.0, atol=1e-4)


def test_dc_offset_removal():
    sig = np.ones((2, 100), dtype=np.float32) * 2.5
    clean = remove_dc_offset(sig)
    assert np.allclose(np.mean(clean), 0.0, atol=1e-5)


def test_pedalboard_effect_chain():
    sr = 44100
    t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
    sig = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 440 * t)]).astype(np.float32)

    chain = EffectChain([
        EffectConfig(effect_id="eq1", effect_type="eq", parameters={"low_gain_db": 3.0, "high_gain_db": -2.0}),
        EffectConfig(effect_id="comp1", effect_type="compressor", parameters={"threshold_db": -12.0}),
        EffectConfig(effect_id="lim1", effect_type="limiter", parameters={"threshold_db": -0.5}),
    ])
    processed = chain.process(sig, sr)
    assert processed.shape == sig.shape
    assert np.max(np.abs(processed)) <= 1.0


def test_rubberband_pitch_and_time_stretch():
    sr = 44100
    sig = np.random.uniform(-0.5, 0.5, (2, 44100)).astype(np.float32)
    rb = RubberBandEngine()

    # Time stretch 2.0x faster -> duration should halve
    stretched = rb.time_stretch(sig, sr, rate=2.0)
    assert abs(stretched.shape[1] - 22050) < 1000

    # Pitch shift +2 semitones -> length preserved
    shifted = rb.pitch_shift(sig, sr, n_semitones=2.0)
    assert abs(shifted.shape[1] - sig.shape[1]) < 100
