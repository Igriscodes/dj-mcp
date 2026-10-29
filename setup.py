import os
import sys
import site
import urllib.request
from pathlib import Path
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.build_py import build_py

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:
    editable_wheel = None


def pre_download_ai_models():
    # Allow skipping via environment variable if offline or in minimal CI
    if os.environ.get("SKIP_AI_MODEL_DOWNLOAD", "0").lower() in ("1", "true", "yes"):
        print("SKIP_AI_MODEL_DOWNLOAD is set. Skipping model download.")
        return

    print("\n" + "=" * 65)
    print("DJ-MCP: Checking and pre-downloading AI separation models...")
    print("=" * 65)

    # 1. Ensure host / virtualenv / user site-packages are visible even during pip build isolation
    candidates = []
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass

    for base in [sys.prefix, sys.base_prefix, os.environ.get("CONDA_PREFIX"), os.environ.get("VIRTUAL_ENV")]:
        if base:
            p = Path(base) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            candidates.append(str(p))

    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)

    # 2. Try loading / downloading via Demucs official API if available
    try:
        from demucs.pretrained import get_model
        print("Verifying / downloading Meta Demucs 'htdemucs' model via demucs...")
        model = get_model("htdemucs")
        print("✓ AI stem separation model 'htdemucs' is verified and cached locally.")
        print("=" * 65 + "\n")
        return
    except Exception as e:
        # Demucs not yet imported or network/hub issue
        pass

    # 3. Direct pre-download fallback to torch checkpoints cache (works with zero third-party deps)
    try:
        cache_dir = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser() / "hub" / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "955717e8-8726e21a.th"
        url = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"

        if target.exists() and target.stat().st_size > 80_000_000:
            print(f"✓ AI stem separation model already present in {target}")
        else:
            print(f"Pre-downloading 'htdemucs' checkpoint to {target}...")
            urllib.request.urlretrieve(url, target)
            print("✓ AI stem separation model checkpoint downloaded successfully.")
    except Exception as e:
        print(f"Notice: Could not pre-download model during setup ({e}).")
        print("The model will auto-download seamlessly on first use or via: dj-mcp download-models")

    print("=" * 65 + "\n")


class PostDevelopCommand(develop):
    def run(self):
        super().run()
        pre_download_ai_models()


class PostInstallCommand(install):
    def run(self):
        super().run()
        pre_download_ai_models()


class PostBuildPyCommand(build_py):
    def run(self):
        super().run()
        pre_download_ai_models()


cmdclass = {
    "develop": PostDevelopCommand,
    "install": PostInstallCommand,
    "build_py": PostBuildPyCommand,
}

if editable_wheel:
    class PostEditableWheelCommand(editable_wheel):
        def run(self):
            super().run()
            pre_download_ai_models()

    cmdclass["editable_wheel"] = PostEditableWheelCommand

setup(
    cmdclass=cmdclass,
)
