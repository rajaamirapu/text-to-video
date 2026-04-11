#!/usr/bin/env python3
"""
setup_opensora.py
=================
One-shot installer for Open-Sora.

What it does
------------
  1. Checks Python / CUDA / VRAM requirements
  2. Clones the Open-Sora repo (or pulls latest if already cloned)
  3. pip-installs Open-Sora and its dependencies inside the current env
  4. Downloads model weights from HuggingFace:
       • STDiT3-XL/2  (hpcai-tech/OpenSora-STDiT-v3)
       • VAE           (hpcai-tech/OpenSora-VAE-v1.2)
       • T5 encoder    (DeepFloyd/t5-v1_1-xxl)
  5. Runs a 1-second smoke test to verify the pipeline loads

Usage
-----
  python setup_opensora.py
  python setup_opensora.py --skip-weights     # skip HF download (if you have weights)
  python setup_opensora.py --opensora-dir /path/to/Open-Sora
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys


OPENSORA_REPO = "https://github.com/hpcaitech/Open-Sora.git"
OPENSORA_TAG  = "v1.3.0"   # pin to a stable release

HF_MODELS = [
    "hpcai-tech/OpenSora-STDiT-v3",
    "hpcai-tech/OpenSora-VAE-v1.2",
    "DeepFloyd/t5-v1_1-xxl",
]

MIN_VRAM_GB = 16   # recommended minimum


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(msg: str):
    w = max(60, len(msg) + 4)
    print(f"\n{'─'*w}\n  {msg}\n{'─'*w}")


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    """Run a shell command, stream output, return exit code."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        sys.exit(f"[Error] Command failed with exit code {result.returncode}")
    return result.returncode


def pip_install(*packages: str):
    run([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — system checks
# ─────────────────────────────────────────────────────────────────────────────

def check_requirements():
    banner("Checking system requirements")

    # Python version
    major, minor = sys.version_info[:2]
    print(f"  Python : {major}.{minor}")
    if (major, minor) < (3, 10):
        sys.exit("[Error] Python 3.10+ is required for Open-Sora.")

    # PyTorch + CUDA
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"  CUDA   : {'✓ ' + torch.version.cuda if cuda_ok else '✗ not available (CPU only — very slow)'}")
        if cuda_ok:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram  = props.total_memory / 1024**3
                print(f"  GPU {i}  : {props.name}  {vram:.1f} GB VRAM")
                if vram < MIN_VRAM_GB:
                    print(f"  [Warning] GPU {i} has {vram:.1f} GB VRAM; "
                          f"{MIN_VRAM_GB} GB recommended for 720p generation.")
    except ImportError:
        print("  [Warning] PyTorch not found — will be installed.")

    # ffmpeg
    rc = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode
    print(f"  ffmpeg : {'✓' if rc == 0 else '✗ NOT FOUND — please install ffmpeg'}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — clone / update the repo
# ─────────────────────────────────────────────────────────────────────────────

def clone_or_update(opensora_dir: str):
    banner(f"Setting up Open-Sora repo → {opensora_dir}")

    if os.path.isdir(opensora_dir):
        if os.path.isdir(os.path.join(opensora_dir, ".git")):
            print("  Repo already exists — pulling latest …")
            run(["git", "fetch", "--tags"], cwd=opensora_dir)
            run(["git", "checkout", OPENSORA_TAG], cwd=opensora_dir, check=False)
            return
        else:
            print(f"  Directory {opensora_dir} exists but is not a git repo — skipping clone.")
            return

    print(f"  Cloning {OPENSORA_REPO} (tag {OPENSORA_TAG}) …")
    parent = os.path.dirname(opensora_dir)
    os.makedirs(parent, exist_ok=True)
    run(["git", "clone",
         "--depth", "1",
         "--branch", OPENSORA_TAG,
         OPENSORA_REPO,
         opensora_dir])


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────

def install_dependencies(opensora_dir: str):
    banner("Installing Python dependencies")

    # Core torch stack (skip if already correct version)
    try:
        import torch
        if not torch.cuda.is_available():
            print("  Installing PyTorch with CUDA 12.1 …")
            pip_install(
                "torch==2.3.0", "torchvision==0.18.0",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            )
    except ImportError:
        print("  Installing PyTorch with CUDA 12.1 …")
        pip_install(
            "torch==2.3.0", "torchvision==0.18.0",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        )

    # Open-Sora package itself
    req_file = os.path.join(opensora_dir, "requirements", "requirements.txt")
    if os.path.isfile(req_file):
        run([sys.executable, "-m", "pip", "install", "-r", req_file])
    else:
        # Fallback: install from setup.py
        run([sys.executable, "-m", "pip", "install", "-e", opensora_dir])

    # Flash Attention 2 (optional but strongly recommended for speed)
    try:
        import flash_attn  # noqa: F401
        print("  flash-attn: already installed")
    except ImportError:
        print("  Installing flash-attn (this may take a few minutes) …")
        rc = run(
            [sys.executable, "-m", "pip", "install",
             "flash-attn", "--no-build-isolation"],
            check=False,
        )
        if rc != 0:
            print("  [Warning] flash-attn failed to install — "
                  "generation will work but be slower.")

    # xformers (memory-efficient attention fallback)
    try:
        import xformers  # noqa: F401
        print("  xformers: already installed")
    except ImportError:
        rc = run(
            [sys.executable, "-m", "pip", "install", "xformers"],
            check=False,
        )
        if rc != 0:
            print("  [Warning] xformers not installed.")

    # Other required packages
    pip_install(
        "transformers>=4.39.0",
        "diffusers>=0.27.0",
        "accelerate>=0.28.0",
        "huggingface_hub>=0.22.0",
        "einops",
        "rotary_embedding_torch",
        "timm",
        "colossalai",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — download model weights
# ─────────────────────────────────────────────────────────────────────────────

def download_weights(opensora_dir: str):
    banner("Downloading model weights from HuggingFace")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pip_install("huggingface_hub")
        from huggingface_hub import snapshot_download

    cache_dir = os.path.join(opensora_dir, "pretrained_models")
    os.makedirs(cache_dir, exist_ok=True)

    for repo_id in HF_MODELS:
        print(f"\n  Downloading {repo_id} …")
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
            )
            print(f"  ✓ {repo_id} → {local_path}")
        except Exception as e:
            print(f"  [Warning] Could not download {repo_id}: {e}")
            print("  You can download it manually with:")
            print(f"    huggingface-cli download {repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(opensora_dir: str):
    banner("Running smoke test (import check)")

    if opensora_dir not in sys.path:
        sys.path.insert(0, opensora_dir)

    try:
        import opensora                                         # noqa: F401
        from opensora.models.stdit.stdit3 import STDiT3        # noqa: F401
        from opensora.models.vae.vae import VideoAutoencoderPipeline  # noqa: F401
        from opensora.schedulers.rf import RFLOW               # noqa: F401
        print("  ✓ Open-Sora imports OK")
    except ImportError as e:
        print(f"  [Warning] Import check failed: {e}")
        print("  This may be OK if CUDA/GPU drivers aren't installed in this env.")
        return

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  [Info] No CUDA GPU — inference will run on CPU.")
    except ImportError:
        pass

    print("\n  Open-Sora setup complete!")
    print("  Usage:  python main.py script.json --use-opensora")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Set up Open-Sora for text-to-video generation.")
    parser.add_argument(
        "--opensora-dir",
        default=os.path.join(os.path.dirname(__file__), "Open-Sora"),
        help="Where to clone/install the Open-Sora repo (default: ./Open-Sora)",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip downloading HuggingFace model weights",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip pip dependency installation (useful if already installed)",
    )
    args = parser.parse_args()

    check_requirements()
    clone_or_update(args.opensora_dir)

    if not args.skip_deps:
        install_dependencies(args.opensora_dir)

    if not args.skip_weights:
        download_weights(args.opensora_dir)

    smoke_test(args.opensora_dir)

    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  Open-Sora setup complete!                                  │
│                                                             │
│  To use Open-Sora in the pipeline:                          │
│    python main.py script.json --use-opensora                │
│                                                             │
│  Optional flags:                                            │
│    --opensora-resolution 480p    (240p/360p/480p/720p)      │
│    --opensora-frames 49          (17/33/49/65/97 frames)    │
│    --opensora-bg                 (also generate background) │
│    --opensora-cli                (use CLI instead of API)   │
└─────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
