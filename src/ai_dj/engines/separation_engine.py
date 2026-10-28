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
        self.last_device_used: Optional[str] = None
        self.fallback_occurred: bool = False

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
            import gc
            import os
            import torch
            import torchaudio
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            # Maximize CPU multithreading performance across all cores
            total_cores = os.cpu_count() or 4
            os.environ["OMP_NUM_THREADS"] = str(total_cores)
            os.environ["MKL_NUM_THREADS"] = str(total_cores)
            torch.set_num_threads(total_cores)
            try:
                torch.set_num_interop_threads(total_cores)
            except Exception:
                pass

            target_device = device
            if target_device is None or target_device == "auto":
                if torch.cuda.is_available():
                    target_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"

            model = get_model(self.model_name)
            model.eval()

            # Load audio using torchaudio
            wav, sr = torchaudio.load(str(audio_path))
            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                wav = resampler(wav)
                sr = model.samplerate

            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]

            # Normalize audio for Demucs
            ref = wav.mean(0)
            norm_wav = (wav - ref.mean()) / ref.std()

            sources = None
            used_device = target_device
            self.fallback_occurred = False

            def run_inference(dev_name: str):
                if dev_name == "cpu":
                    # Guarantee all cores are utilized for PyTorch and OpenMP
                    torch.set_num_threads(total_cores)
                model.to(dev_name)
                input_tensor = norm_wav.unsqueeze(0).to(dev_name)
                with torch.no_grad():
                    res = apply_model(model, input_tensor, shifts=1, split=True, overlap=0.25, progress=False)[0]
                return res

            if target_device == "cuda":
                try:
                    sources = run_inference("cuda")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    err_msg = str(e).lower()
                    if "out of memory" in err_msg or "cuda" in err_msg or isinstance(e, torch.cuda.OutOfMemoryError):
                        # Force flush GPU memory cache
                        torch.cuda.empty_cache()
                        gc.collect()
                        self.fallback_occurred = True
                        used_device = "cpu"
                        sources = run_inference("cpu")
                    else:
                        raise
            else:
                sources = run_inference(target_device)

            self.last_device_used = used_device

            # Rescale sources
            sources = sources * ref.std() + ref.mean()
            source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

            results: Dict[str, Path] = {}
            for name, source in zip(source_names, sources):
                if stems and name not in stems:
                    continue
                out_file = output_dir / f"{audio_path.stem}_{name}.wav"
                short_file = output_dir / f"{name}.wav"
                source_cpu = source.cpu().numpy()
                # soundfile write expects (samples, channels)
                sf.write(str(out_file), source_cpu.T, sr, subtype="PCM_24")
                if out_file != short_file:
                    try:
                        import shutil
                        shutil.copyfile(str(out_file), str(short_file))
                    except Exception:
                        sf.write(str(short_file), source_cpu.T, sr, subtype="PCM_24")
                results[name] = out_file

            return results
        except Exception as e:
            raise DJError(
                code="STEM_SEPARATION_FAILED",
                message=f"Stem separation failed on '{audio_path}': {str(e)}",
                details={"audio_path": str(audio_path)},
            ) from e
