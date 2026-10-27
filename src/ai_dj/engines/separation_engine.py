from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from ai_dj.errors import DJError, EngineUnavailableError


class SeparationEngine:
    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self._model = None
        self._device = None

    @property
    def is_available(self) -> bool:
        try:
            import demucs
            import torch
            return True
        except ImportError:
            return False

    def get_status(self) -> Dict[str, Any]:
        avail = self.is_available
        device = "cpu"
        if avail:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        return {
            "available": avail,
            "engine": "demucs",
            "model": self.model_name,
            "device": device,
            "stems_supported": ["vocals", "drums", "bass", "other"],
        }

    def separate(
        self,
        audio_path: Path,
        output_dir: Path,
        stems: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Path]:
        if not self.is_available:
            raise EngineUnavailableError(
                engine_name="demucs",
                required_feature="neural stem source separation",
                install_command="pip install demucs torch torchaudio",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            import torchaudio

            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            model = get_model(self.model_name)
            model.to(device)
            model.eval()

            # Load audio using torchaudio or soundfile
            wav, sr = torchaudio.load(str(audio_path))
            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                wav = resampler(wav)
                sr = model.samplerate

            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]

            # Demucs expects (batch, channels, time)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            wav = wav.unsqueeze(0).to(device)

            with torch.no_grad():
                sources = apply_model(model, wav, shifts=1, split=True, overlap=0.25, progress=False)[0]

            # Sources shape: (num_sources, channels, time)
            sources = sources * ref.std() + ref.mean()
            source_names = model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']

            results: Dict[str, Path] = {}
            for name, source in zip(source_names, sources):
                if stems and name not in stems:
                    continue
                out_file = output_dir / f"{audio_path.stem}_{name}.wav"
                source_cpu = source.cpu().numpy()
                # soundfile write expects (samples, channels)
                sf.write(str(out_file), source_cpu.T, sr, subtype="PCM_24")
                results[name] = out_file

            return results
        except Exception as e:
            raise DJError(
                code="STEM_SEPARATION_FAILED",
                message=f"Stem separation failed on '{audio_path}': {str(e)}",
                details={"audio_path": str(audio_path)},
            ) from e
