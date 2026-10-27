import importlib.util
import os
import platform
import shutil
import subprocess
from typing import Any, Dict, List
import psutil


def is_tool_available(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def is_python_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_tool_version(cmd: List[str]) -> str:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        out = proc.stdout.strip() or proc.stderr.strip()
        return out.splitlines()[0] if out else "Unknown"
    except Exception:
        return "Unknown"


def get_system_capabilities() -> Dict[str, Any]:
    sys_name = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()

    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024 ** 3), 2)
    avail_ram_gb = round(vm.available / (1024 ** 3), 2)
    cpu_count = os.cpu_count() or 1

    ffmpeg_available = is_tool_available("ffmpeg")
    ffmpeg_ver = get_tool_version(["ffmpeg", "-version"]) if ffmpeg_available else None

    ffprobe_available = is_tool_available("ffprobe")
    rubberband_cli = is_tool_available("rubberband")
    sox_available = is_tool_available("sox")

    soundfile_avail = is_python_module_available("soundfile")
    pedalboard_avail = is_python_module_available("pedalboard")
    librosa_avail = is_python_module_available("librosa")
    pyloudnorm_avail = is_python_module_available("pyloudnorm")
    demucs_avail = is_python_module_available("demucs")
    torch_avail = is_python_module_available("torch")

    gpu_info = {"available": False, "type": "cpu", "device_count": 0, "device_names": []}
    if torch_avail:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["type"] = "cuda"
                gpu_info["device_count"] = torch.cuda.device_count()
                gpu_info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_info["available"] = True
                gpu_info["type"] = "metal_mps"
                gpu_info["device_count"] = 1
                gpu_info["device_names"] = ["Apple Silicon GPU"]
        except Exception:
            pass

    codecs = {
        "wav": True,
        "flac": soundfile_avail or ffmpeg_available,
        "aiff": soundfile_avail or ffmpeg_available,
        "ogg": soundfile_avail or ffmpeg_available,
        "mp3": ffmpeg_available or is_python_module_available("lameenc"),
        "aac": ffmpeg_available,
    }

    engines = {
        "ffmpeg": {
            "available": ffmpeg_available,
            "version": ffmpeg_ver,
            "purpose": "Audio container decoding, encoding, format conversion",
        },
        "soundfile": {
            "available": soundfile_avail,
            "purpose": "Sample-accurate uncompressed PCM and lossless I/O",
        },
        "pedalboard": {
            "available": pedalboard_avail,
            "purpose": "Native C++ JUCE DSP effects (EQ, filters, compressor, limiter, reverb, delay)",
        },
        "librosa": {
            "available": librosa_avail,
            "purpose": "BPM, beat-grid, onset/transient, chroma and key detection",
        },
        "pyloudnorm": {
            "available": pyloudnorm_avail,
            "purpose": "ITU-R BS.1770-4 / EBU R128 integrated and momentary LUFS loudness analysis",
        },
        "demucs": {
            "available": demucs_avail,
            "purpose": "High-fidelity neural source separation (vocals, drums, bass, other)",
        },
        "rubberband": {
            "available": rubberband_cli or is_python_module_available("pyrubberband"),
            "purpose": "High-quality time-stretching and pitch shifting",
        },
    }

    return {
        "os": {
            "system": sys_name,
            "architecture": machine,
            "python_version": python_version,
            "cpu_cores": cpu_count,
            "total_ram_gb": total_ram_gb,
            "available_ram_gb": avail_ram_gb,
        },
        "gpu": gpu_info,
        "engines": engines,
        "codecs": codecs,
        "features": {
            "non_destructive_editing": True,
            "beat_grid_detection": librosa_avail,
            "key_detection": librosa_avail,
            "stem_separation": demucs_avail,
            "mastering_lufs": pyloudnorm_avail,
            "hardware_acceleration": gpu_info["available"],
        },
    }
