import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ai_dj.errors import DJError, ValidationError


class RubberBandEngine:
    def __init__(self):
        self._has_cli = shutil.which("rubberband") is not None
        self._has_pyrubberband = False
        try:
            import pyrubberband
            self._has_pyrubberband = True
        except ImportError:
            pass

    @property
    def is_native_rubberband_available(self) -> bool:
        return self._has_cli and self._has_pyrubberband

    def time_stretch(
        self,
        audio: np.ndarray,
        sample_rate: int,
        rate: float,
    ) -> np.ndarray:
        if rate <= 0.0:
            raise ValidationError("rate", "Time stretch rate must be strictly positive (> 0.0)")
        if abs(rate - 1.0) < 1e-4:
            return audio.copy()

        if self.is_native_rubberband_available:
            try:
                import pyrubberband as pyrb
                # pyrubberband accepts shape (channels, samples) or (samples, channels)
                # Ensure float64 or float32
                if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
                    # (channels, samples) -> pyrb expects (samples, channels)
                    stretched = pyrb.time_stretch(audio.T, sample_rate, rate).T
                else:
                    stretched = pyrb.time_stretch(audio, sample_rate, rate)
                return np.ascontiguousarray(stretched, dtype=np.float32)
            except Exception:
                pass

        # Fallback using librosa high quality phase vocoder time stretching
        try:
            import librosa
            if audio.ndim == 1:
                return np.ascontiguousarray(librosa.effects.time_stretch(audio, rate=rate), dtype=np.float32)
            elif audio.ndim == 2:
                # Process channel by channel
                channels = []
                for ch in range(audio.shape[0]):
                    ch_stretched = librosa.effects.time_stretch(audio[ch], rate=rate)
                    channels.append(ch_stretched)
                # Pad to same length if needed
                max_len = max(len(c) for c in channels)
                res = np.zeros((len(channels), max_len), dtype=np.float32)
                for ch, c in enumerate(channels):
                    res[ch, :len(c)] = c
                return res
        except Exception as e:
            raise DJError(
                code="TIME_STRETCH_FAILED",
                message=f"Time stretch processing failed: {str(e)}",
            ) from e

    def pitch_shift(
        self,
        audio: np.ndarray,
        sample_rate: int,
        n_semitones: float,
    ) -> np.ndarray:
        if abs(n_semitones) < 1e-4:
            return audio.copy()

        if self.is_native_rubberband_available:
            try:
                import pyrubberband as pyrb
                if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
                    shifted = pyrb.pitch_shift(audio.T, sample_rate, n_semitones).T
                else:
                    shifted = pyrb.pitch_shift(audio, sample_rate, n_semitones)
                return np.ascontiguousarray(shifted, dtype=np.float32)
            except Exception:
                pass

        # Fallback 1: Pedalboard PitchShift
        try:
            import pedalboard
            board = pedalboard.Pedalboard([pedalboard.PitchShift(semitones=float(n_semitones))])
            # Pedalboard expects (channels, samples)
            data = audio
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            processed = board(data.astype(np.float32), sample_rate)
            if audio.ndim == 1:
                return processed[0]
            return processed
        except Exception:
            pass

        # Fallback 2: librosa pitch_shift
        try:
            import librosa
            if audio.ndim == 1:
                return np.ascontiguousarray(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_semitones), dtype=np.float32)
            else:
                channels = [
                    librosa.effects.pitch_shift(audio[ch], sr=sample_rate, n_steps=n_semitones)
                    for ch in range(audio.shape[0])
                ]
                return np.ascontiguousarray(np.stack(channels), dtype=np.float32)
        except Exception as e:
            raise DJError(
                code="PITCH_SHIFT_FAILED",
                message=f"Pitch shift processing failed: {str(e)}",
            ) from e

    def change_speed(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speed_ratio: float,
    ) -> np.ndarray:
        if speed_ratio <= 0.0:
            raise ValidationError("speed_ratio", "Speed ratio must be strictly positive (> 0.0)")
        if abs(speed_ratio - 1.0) < 1e-4:
            return audio.copy()

        import scipy.signal as signal
        from fractions import Fraction
        frac = Fraction(float(speed_ratio)).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        # Resample across time axis: output length = N / speed_ratio = N * down / up
        resampled = signal.resample_poly(audio, down, up, axis=-1)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    def paulstretch(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stretch_factor: float = 8.0,
        window_size_sec: float = 0.25,
    ) -> np.ndarray:
        if stretch_factor <= 1.0:
            return audio.copy()

        window_size = int(window_size_sec * sample_rate)
        if window_size % 2 == 1:
            window_size += 1
        half_win = window_size // 2
        h_in = max(1, int(half_win / stretch_factor))

        is_1d = audio.ndim == 1
        channels = [audio] if is_1d else [audio[c] for c in range(audio.shape[0])]
        out_channels = []

        win = np.hanning(window_size).astype(np.float32)
        for ch in channels:
            n_samples = len(ch)
            out_len = int(n_samples * stretch_factor)
            out = np.zeros(out_len + window_size * 2, dtype=np.float32)

            in_pos = 0
            out_pos = 0
            while in_pos + window_size <= n_samples:
                segment = ch[in_pos:in_pos + window_size] * win
                spec = np.fft.rfft(segment)
                phases = np.random.uniform(0, 2 * np.pi, len(spec))
                rand_spec = np.abs(spec) * np.exp(1j * phases)
                stretched_seg = np.fft.irfft(rand_spec, n=window_size) * win
                out[out_pos:out_pos + window_size] += stretched_seg
                in_pos += h_in
                out_pos += half_win
            out_channels.append(out[:out_len])

        res = np.vstack(out_channels) if not is_1d else out_channels[0]
        max_val = float(np.max(np.abs(res))) if res.size > 0 else 0.0
        if max_val > 1.0:
            res = res / max_val
        return np.ascontiguousarray(res, dtype=np.float32)

    def truncate_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold_db: float = -50.0,
        min_silence_sec: float = 0.5,
        truncate_to_sec: float = 0.0,
    ) -> np.ndarray:
        thresh_lin = 10.0 ** (threshold_db / 20.0)
        mono = np.max(np.abs(audio), axis=0) if audio.ndim == 2 else np.abs(audio)

        frame_len = max(1, int(0.01 * sample_rate))
        n_frames = len(mono) // frame_len
        if n_frames == 0:
            return audio.copy()

        frames_energy = np.array([np.max(mono[i * frame_len : (i + 1) * frame_len]) for i in range(n_frames)])
        is_silent = frames_energy < thresh_lin

        min_silence_frames = max(1, int(min_silence_sec / 0.01))
        target_silence_frames = max(0, int(truncate_to_sec / 0.01))

        keep_mask = np.ones(n_frames, dtype=bool)
        i = 0
        while i < n_frames:
            if is_silent[i]:
                j = i
                while j < n_frames and is_silent[j]:
                    j += 1
                silence_len = j - i
                if silence_len >= min_silence_frames:
                    drop_start = i + target_silence_frames
                    if drop_start < j:
                        keep_mask[drop_start:j] = False
                i = j
            else:
                i += 1

        kept_indices = []
        for idx, keep in enumerate(keep_mask):
            if keep:
                start_idx = idx * frame_len
                end_idx = min((idx + 1) * frame_len, audio.shape[-1])
                kept_indices.extend(range(start_idx, end_idx))

        if not kept_indices:
            return audio[:, :frame_len] if audio.ndim == 2 else audio[:frame_len]
        return np.ascontiguousarray(audio[:, kept_indices] if audio.ndim == 2 else audio[kept_indices], dtype=np.float32)

    def declick(
        self,
        audio: np.ndarray,
        sample_rate: int,
        sensitivity: float = 0.5,
    ) -> np.ndarray:
        # Detect sharp clicks via 2nd order discrete derivative
        out = audio.copy()
        thresh = max(0.05, (1.0 - sensitivity) * 0.5)
        if out.ndim == 1:
            diff2 = np.abs(np.diff(out, n=2))
            clicks = np.where(diff2 > thresh)[0] + 1
            for c in clicks:
                if 1 <= c < len(out) - 1:
                    out[c] = 0.5 * (out[c - 1] + out[c + 1])
        else:
            for ch in range(out.shape[0]):
                diff2 = np.abs(np.diff(out[ch], n=2))
                clicks = np.where(diff2 > thresh)[0] + 1
                for c in clicks:
                    if 1 <= c < out.shape[1] - 1:
                        out[ch, c] = 0.5 * (out[ch, c - 1] + out[ch, c + 1])
        return np.ascontiguousarray(out, dtype=np.float32)

    def denoise(
        self,
        audio: np.ndarray,
        sample_rate: int,
        reduction_db: float = 12.0,
    ) -> np.ndarray:
        # High quality downward noise gating via Pedalboard
        try:
            import pedalboard
            gate = pedalboard.NoiseGate(threshold_db=max(-80.0, -reduction_db - 40.0), ratio=3.0)
            data = audio if audio.ndim == 2 else np.expand_dims(audio, axis=0)
            processed = gate(data.astype(np.float32), sample_rate)
            return np.ascontiguousarray(processed if audio.ndim == 2 else processed[0], dtype=np.float32)
        except Exception:
            return audio.copy()

    def vocal_remover(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str = "remove_vocals",
        low_cutoff_hz: float = 120.0,
        high_cutoff_hz: float = 8000.0,
    ) -> np.ndarray:
        # Center-channel extraction / cancellation with bass preservation
        if audio.ndim == 1 or audio.shape[0] < 2:
            return audio.copy()

        l = audio[0]
        r = audio[1]

        # Mid (center) and Side
        mid = 0.5 * (l + r)
        side = 0.5 * (l - r)

        import scipy.signal as signal
        # Lowpass filter for preserving bass (kick, sub) in vocal removal
        sos_bass = signal.butter(4, low_cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
        bass_mono = signal.sosfilt(sos_bass, mid)

        if mode == "isolate_vocals":
            # Bandpass center channel for vocal isolation
            sos_vocal = signal.butter(4, [low_cutoff_hz, min(high_cutoff_hz, (sample_rate / 2) - 100)], btype="bandpass", fs=sample_rate, output="sos")
            vocal = signal.sosfilt(sos_vocal, mid)
            return np.ascontiguousarray(np.stack([vocal, vocal]), dtype=np.float32)
        else:
            # Vocal removal: Side signal + preserved bass
            out_l = side + bass_mono
            out_r = -side + bass_mono
            return np.ascontiguousarray(np.stack([out_l, out_r]), dtype=np.float32)
