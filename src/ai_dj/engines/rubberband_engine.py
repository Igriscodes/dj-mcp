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
