from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
from ai_dj.engines.ffmpeg_engine import FFmpegEngine
from ai_dj.errors import DJError, ValidationError
from ai_dj.resources import create_managed_temp_file


def export_audio_to_file(
    audio: np.ndarray,
    output_path: Path,
    sample_rate: int = 44100,
    bit_depth: int = 24,
    audio_format: Optional[str] = None,
    bit_rate_kbps: int = 320,
    overwrite: bool = True,
) -> Path:
    if not overwrite and output_path.exists():
        raise DJError(
            code="FILE_ALREADY_EXISTS",
            message=f"Output file '{output_path}' already exists and overwrite is set to False.",
            reason="file_exists",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (audio_format or output_path.suffix.strip(".")).lower()
    # soundfile expects (samples, channels)
    data = audio.T if audio.ndim == 2 else np.expand_dims(audio, axis=1)

    if data.shape[0] == 0:
        raise DJError(
            code="EMPTY_AUDIO_EXPORT",
            message=f"Cannot export '{output_path}': audio signal contains 0 samples.",
            reason="empty_audio",
        )

    subtype_map = {
        16: "PCM_16",
        24: "PCM_24",
        32: "FLOAT",
    }
    subtype = subtype_map.get(bit_depth, "PCM_24")

    # Fast native write for WAV / FLAC / AIFF / OGG via soundfile
    if fmt in ("wav", "wave"):
        sf.write(str(output_path), data, sample_rate, subtype=subtype)
        return output_path
    elif fmt == "flac":
        sf.write(str(output_path), data, sample_rate, format="FLAC", subtype=subtype if bit_depth in (16, 24) else "PCM_24")
        return output_path
    elif fmt in ("aiff", "aif"):
        sf.write(str(output_path), data, sample_rate, format="AIFF", subtype=subtype)
        return output_path
    elif fmt == "ogg":
        try:
            sf.write(str(output_path), data, sample_rate, format="OGG")
            return output_path
        except Exception:
            pass

    # For MP3, AAC, or formats needing FFmpeg
    ffmpeg = FFmpegEngine()
    if not ffmpeg.is_available():
        raise DJError(
            code="CODEC_UNAVAILABLE",
            message=f"Exporting to '{fmt}' format requires FFmpeg, which is not available.",
            reason="ffmpeg_missing",
        )

    temp_wav = create_managed_temp_file(suffix=".wav")
    sf.write(str(temp_wav), data, sample_rate, subtype="PCM_24")
    try:
        ffmpeg.convert(
            input_path=temp_wav,
            output_path=output_path,
            sample_rate=sample_rate,
            channels=data.shape[1],
            bit_rate_kbps=bit_rate_kbps,
            overwrite=overwrite,
        )
    finally:
        if temp_wav.exists():
            temp_wav.unlink()

    return output_path
