import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from ai_dj.config import DJConfig, get_default_config
from ai_dj.errors import DJError
from ai_dj.key import KEY_TO_CAMELOT, key_to_camelot
from ai_dj.metadata import AudioMetadata, compute_file_hash

# Krumhansl-Schmuckler Key Profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class AudioAnalyzer:
    def __init__(self, config: Optional[DJConfig] = None):
        self.config = config or get_default_config()

    def get_cached_analysis(self, file_hash: str) -> Optional[Dict[str, Any]]:
        cache_file = self.config.cache_dir / f"analysis_{file_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_cached_analysis(self, file_hash: str, data: Dict[str, Any]) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.config.cache_dir / f"analysis_{file_hash}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_audio_mono(self, audio_path: Path, max_duration: Optional[float] = 180.0) -> Tuple[np.ndarray, int]:
        try:
            import librosa
            y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=max_duration)
            return y, sr
        except Exception:
            data, sr = sf.read(str(audio_path), always_2d=False)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if max_duration:
                max_samples = int(max_duration * sr)
                data = data[:max_samples]
            return data.astype(np.float32), sr

    def analyze_bpm_and_beats(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        import librosa
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
        beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
        first_beat = beat_times[0] if beat_times else 0.0
        downbeats = [beat_times[i] for i in range(0, len(beat_times), 4)] if beat_times else []

        return {
            "bpm": round(bpm, 2),
            "first_beat_sec": round(first_beat, 4),
            "beat_count": len(beat_times),
            "beats": [round(b, 4) for b in beat_times[:128]],
            "downbeats": [round(db, 4) for db in downbeats[:32]],
        }

    def analyze_key(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        import librosa
        # Compute chroma feature
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        best_score = -1.0
        best_key = "C major"

        for i, note in enumerate(NOTE_NAMES):
            maj_prof = np.roll(MAJOR_PROFILE, i)
            min_prof = np.roll(MINOR_PROFILE, i)

            maj_corr = np.corrcoef(chroma_mean, maj_prof)[0, 1]
            min_corr = np.corrcoef(chroma_mean, min_prof)[0, 1]

            if maj_corr > best_score:
                best_score = maj_corr
                best_key = f"{note} major"

            if min_corr > best_score:
                best_score = min_corr
                best_key = f"{note} minor"

        camelot = key_to_camelot(best_key) or "Unknown"

        return {
            "key": best_key,
            "camelot": camelot,
            "confidence": round(float(best_score), 3),
        }

    def analyze_loudness_and_dynamics(self, audio_path: Path) -> Dict[str, Any]:
        data, sr = sf.read(str(audio_path), always_2d=True)
        # data shape: (samples, channels)
        peak_amp = float(np.max(np.abs(data)))
        peak_dbfs = round(20.0 * math.log10(peak_amp + 1e-9), 2)
        rms_amp = float(np.sqrt(np.mean(np.square(data))))
        rms_dbfs = round(20.0 * math.log10(rms_amp + 1e-9), 2)
        crest_factor = round(peak_dbfs - rms_dbfs, 2)

        integrated_lufs = -70.0
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            integrated_lufs = round(float(meter.integrated_loudness(data)), 2)
        except Exception:
            integrated_lufs = round(rms_dbfs - 3.0, 2)

        # True Peak 4x oversampling approximation
        true_peak_db = peak_dbfs
        try:
            import scipy.signal as signal
            # Resample short loudest section 4x
            max_idx = int(np.argmax(np.abs(data[:, 0])))
            start_idx = max(0, max_idx - 1000)
            end_idx = min(len(data), max_idx + 1000)
            chunk = data[start_idx:end_idx]
            if len(chunk) > 10:
                upsampled = signal.resample_poly(chunk, 4, 1)
                true_peak_amp = float(np.max(np.abs(upsampled)))
                true_peak_db = round(20.0 * math.log10(true_peak_amp + 1e-9), 2)
        except Exception:
            true_peak_db = peak_dbfs

        return {
            "integrated_lufs": integrated_lufs,
            "peak_dbfs": peak_dbfs,
            "true_peak_dbfs": true_peak_db,
            "rms_dbfs": rms_dbfs,
            "crest_factor_db": crest_factor,
            "is_clipping": peak_amp >= 0.9999 or true_peak_db >= 0.0,
        }

    def analyze_track(self, audio_path: Path, force_recompute: bool = False) -> Dict[str, Any]:
        if not audio_path.exists():
            raise DJError(
                code="FILE_NOT_FOUND",
                message=f"Audio file '{audio_path}' does not exist.",
            )

        meta = AudioMetadata.from_file(audio_path)
        if not force_recompute:
            cached = self.get_cached_analysis(meta.file_hash)
            if cached:
                cached["cached"] = True
                return cached

        y, sr = self.load_audio_mono(audio_path)
        bpm_info = self.analyze_bpm_and_beats(y, sr)
        key_info = self.analyze_key(y, sr)
        loud_info = self.analyze_loudness_and_dynamics(audio_path)

        result = {
            "file_hash": meta.file_hash,
            "duration_seconds": meta.duration_seconds,
            "sample_rate": meta.sample_rate,
            "channels": meta.channels,
            "bit_depth": meta.bit_depth,
            "format": meta.format,
            "bpm": bpm_info["bpm"],
            "first_beat_sec": bpm_info["first_beat_sec"],
            "beat_count": bpm_info["beat_count"],
            "beats": bpm_info["beats"],
            "downbeats": bpm_info["downbeats"],
            "key": key_info["key"],
            "camelot": key_info["camelot"],
            "key_confidence": key_info["confidence"],
            "loudness": loud_info,
            "cached": False,
        }
        self.save_cached_analysis(meta.file_hash, result)
        return result
