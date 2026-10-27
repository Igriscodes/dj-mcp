import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from ai_dj.errors import DJError, EngineUnavailableError


class FFmpegEngine:
    def __init__(self, ffmpeg_bin: Optional[str] = None, ffprobe_bin: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_bin = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"

    def is_available(self) -> bool:
        return shutil.which(self.ffmpeg_bin) is not None

    def require_available(self) -> None:
        if not self.is_available():
            raise EngineUnavailableError(
                engine_name="ffmpeg",
                required_feature="audio format conversion and metadata probing",
                install_command="sudo apt install ffmpeg (Linux) or brew install ffmpeg (macOS) or choco install ffmpeg (Windows)",
            )

    def probe(self, audio_path: Path) -> Dict[str, Any]:
        self.require_available()
        cmd = [
            self.ffprobe_bin,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(audio_path),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            info = json.loads(res.stdout)
            streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
            audio_stream = streams[0] if streams else {}
            fmt = info.get("format", {})

            return {
                "format": fmt.get("format_name", "unknown"),
                "duration_seconds": float(fmt.get("duration", audio_stream.get("duration", 0.0))),
                "sample_rate": int(audio_stream.get("sample_rate", 44100)),
                "channels": int(audio_stream.get("channels", 2)),
                "bit_rate": int(fmt.get("bit_rate", audio_stream.get("bit_rate", 0))),
                "codec": audio_stream.get("codec_name", "unknown"),
                "tags": fmt.get("tags", {}),
            }
        except subprocess.CalledProcessError as e:
            raise DJError(
                code="PROBE_FAILED",
                message=f"ffprobe failed to read audio metadata from '{audio_path}': {e.stderr.strip()}",
                details={"path": str(audio_path)},
            ) from e
        except Exception as e:
            raise DJError(
                code="PROBE_ERROR",
                message=f"Error inspecting audio metadata from '{audio_path}': {str(e)}",
            ) from e

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bit_depth: Optional[int] = None,
        codec: Optional[str] = None,
        bit_rate_kbps: Optional[int] = None,
        overwrite: bool = True,
    ) -> Path:
        self.require_available()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.ffmpeg_bin, "-y" if overwrite else "-n", "-i", str(input_path)]

        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        if channels:
            cmd.extend(["-ac", str(channels)])
        if bit_rate_kbps:
            cmd.extend(["-b:a", f"{bit_rate_kbps}k"])
        if codec:
            cmd.extend(["-c:a", codec])
        elif bit_depth and output_path.suffix.lower() == ".wav":
            if bit_depth == 16:
                cmd.extend(["-c:a", "pcm_s16le"])
            elif bit_depth == 24:
                cmd.extend(["-c:a", "pcm_s24le"])
            elif bit_depth == 32:
                cmd.extend(["-c:a", "pcm_f32le"])

        cmd.append(str(output_path))

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise DJError(
                code="FFMPEG_CONVERSION_FAILED",
                message=f"FFmpeg conversion failed: {e.stderr.strip()}",
                details={"input": str(input_path), "output": str(output_path)},
            ) from e
