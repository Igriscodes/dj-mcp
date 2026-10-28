import os
import sys
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
    try:
        from demucs.pretrained import get_model
        print("Downloading / verifying Meta Demucs 'htdemucs' model...")
        model = get_model("htdemucs")
        print("✓ AI stem separation model 'htdemucs' is ready and cached locally.")
    except ImportError:
        print("Notice: 'demucs' is not installed in the active Python environment.")
        print("To enable neural stem separation, install with: pip install '.[stems]'")
        print("Then download models with: dj-mcp download-models")
    except Exception as e:
        print(f"Warning: Could not pre-download model during install: {e}")
        print("You can download it anytime by running: dj-mcp download-models")
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
