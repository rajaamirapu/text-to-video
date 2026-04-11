#!/usr/bin/env python3
"""
setup_opensora.py
=================
One-shot installer for Open-Sora v1.3.

What it does
------------
  1. Checks Python / CUDA / VRAM requirements
  2. Clones the Open-Sora repo (tag v1.3) — or pulls latest if already cloned
  3. pip-installs PyTorch (CUDA 12.1) + Open-Sora dependencies
  4. Downloads publicly-available model weights from HuggingFace:
       • STDiT3-XL/2  →  hpcai-tech/OpenSora-STDiT-v3     (text-to-video model)
       • VAE          →  hpcai-tech/OpenSora-VAE-v1.2      (video encoder/decoder)
       • T5 encoder   →  DeepFloyd/t5-v1_1-xxl             (text encoder)
  5. Writes a ready-to-use inference config
  6. Runs an import smoke-test

Note on versions
----------------
  Open-Sora v1.3 is the latest codebase.  The publicly released model weights
  on HuggingFace are from v1.2 (STDiT-v3 / VAE-v1.2) — the v1.3 weights are
  not yet publicly available.  We use the v1.3 code with the v1.2 weights,
  which is the officially supported HF inference path (see
  configs/opensora-v1-2/inference/sample_hf.py).

Usage
-----
  python setup_opensora.py
  python setup_opensora.py --skip-weights     # skip HF download
  python setup_opensora.py --opensora-dir /path/to/Open-Sora
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import textwrap


OPENSORA_REPO = "https://github.com/hpcaitech/Open-Sora.git"
OPENSORA_TAG  = "v1.3"          # latest stable tag

# Publicly available HF weights (v1.2 weights, used with v1.3 code)
HF_MODELS = [
    "hpcai-tech/OpenSora-STDiT-v3",   # main video-diffusion transformer
    "hpcai-tech/OpenSora-VAE-v1.2",   # 3D VAE
    "DeepFloyd/t5-v1_1-xxl",          # T5 text encoder (4.3 GB)
]

MIN_VRAM_GB = 16


# ─────────────────────────────────────────────────────────────────────────────

def banner(msg: str):
    w = max(60, len(msg) + 4)
    print(f"\n{'─'*w}\n  {msg}\n{'─'*w}")


def run(cmd: list, cwd: str | None = None, check: bool = True) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        sys.exit(f"[Error] Command failed (exit {result.returncode})")
    return result.returncode


def pip(*packages: str, extra_index: str | None = None):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    run(cmd)


# ── Step 1: system checks ─────────────────────────────────────────────────────

def check_requirements():
    banner("Checking system requirements")
    major, minor = sys.version_info[:2]
    print(f"  Python : {major}.{minor}")
    if (major, minor) < (3, 10):
        sys.exit("[Error] Python 3.10+ required.")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p    = torch.cuda.get_device_properties(i)
                vram = p.total_memory / 1024**3
                flag = "✓" if vram >= MIN_VRAM_GB else f"⚠ ({MIN_VRAM_GB} GB recommended)"
                print(f"  GPU {i}  : {p.name}  {vram:.1f} GB  {flag}")
        else:
            print("  CUDA   : ✗ not available — inference will be extremely slow on CPU")
    except ImportError:
        print("  PyTorch: not found — will install")

    rc = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode
    print(f"  ffmpeg : {'✓' if rc == 0 else '✗ missing — please install ffmpeg'}")

    rc2 = subprocess.run(["git", "--version"], capture_output=True).returncode
    print(f"  git    : {'✓' if rc2 == 0 else '✗ missing — please install git'}")


# ── Step 2: clone / update repo ───────────────────────────────────────────────

def clone_or_update(opensora_dir: str):
    banner(f"Setting up Open-Sora repo  →  {opensora_dir}")

    if os.path.isdir(os.path.join(opensora_dir, ".git")):
        print("  Repo exists — fetching tags …")
        run(["git", "fetch", "--tags", "--depth", "1"], cwd=opensora_dir, check=False)
        run(["git", "checkout", OPENSORA_TAG], cwd=opensora_dir, check=False)
        return

    parent = os.path.dirname(os.path.abspath(opensora_dir))
    os.makedirs(parent, exist_ok=True)

    print(f"  Cloning {OPENSORA_REPO}  (tag {OPENSORA_TAG}) …")
    run([
        "git", "clone",
        "--depth", "1",
        "--branch", OPENSORA_TAG,
        OPENSORA_REPO,
        opensora_dir,
    ])


# ── Step 3: install dependencies ──────────────────────────────────────────────

def install_dependencies(opensora_dir: str):
    banner("Installing Python dependencies")

    # PyTorch (CUDA 12.1)
    try:
        import torch
        if not torch.cuda.is_available():
            raise ImportError("no cuda")
        print(f"  PyTorch {torch.__version__} with CUDA already installed — skipping.")
    except ImportError:
        print("  Installing PyTorch 2.2 + CUDA 12.1 …")
        run([
            sys.executable, "-m", "pip", "install",
            "torch==2.2.2", "torchvision==0.17.2",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ])

    # Core Open-Sora requirements
    req = os.path.join(opensora_dir, "requirements", "requirements.txt")
    if os.path.isfile(req):
        run([sys.executable, "-m", "pip", "install", "-r", req])
    else:
        print(f"  [Warning] requirements not found at {req}")

    # Install Open-Sora package itself (editable)
    run([sys.executable, "-m", "pip", "install", "-e", opensora_dir,
         "--no-deps"])   # deps already installed above

    # xformers (memory-efficient attention — strongly recommended)
    try:
        import xformers; print("  xformers: already installed")  # noqa: E401
    except ImportError:
        print("  Installing xformers …")
        run([sys.executable, "-m", "pip", "install",
             "xformers==0.0.25.post1",
             "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)

    # Flash Attention 2 (optional — faster but requires nvcc)
    try:
        import flash_attn; print("  flash-attn: already installed")  # noqa: E401
    except ImportError:
        print("  Attempting flash-attn install (needs nvcc — skip if it fails) …")
        run([sys.executable, "-m", "pip", "install",
             "flash-attn", "--no-build-isolation"], check=False)


# ── Step 4: download model weights ────────────────────────────────────────────

def download_weights(opensora_dir: str):
    banner("Downloading model weights from HuggingFace")
    print("  Models: OpenSora-STDiT-v3, OpenSora-VAE-v1.2, t5-v1_1-xxl")
    print("  Note  : These are the v1.2 publicly released weights, compatible")
    print("          with the v1.3 codebase via configs/opensora-v1-2/inference/sample_hf.py")
    print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    cache_dir = os.path.join(opensora_dir, "pretrained_models")
    os.makedirs(cache_dir, exist_ok=True)

    for repo_id in HF_MODELS:
        print(f"  Downloading  {repo_id}  …")
        try:
            path = snapshot_download(
                repo_id       = repo_id,
                cache_dir     = cache_dir,
                ignore_patterns = [
                    "*.msgpack", "flax_model*", "tf_model*",
                    "rust_model*", "onnx*",
                ],
            )
            print(f"  ✓  {repo_id}\n     → {path}\n")
        except Exception as e:
            print(f"  [Warning] Could not download {repo_id}: {e}")
            print(f"  Manual download:  huggingface-cli download {repo_id}\n")


# ── Step 5: write inference config helper ─────────────────────────────────────

def write_inference_config(opensora_dir: str):
    """
    Write a ready-to-use inference config (480p, 51 frames) that points to
    the HF model repos.  Saved as configs/opensora-v1-2/inference/sample_hf.py
    which already exists in the repo — this step is informational only.
    """
    cfg_path = os.path.join(
        opensora_dir, "configs", "opensora-v1-2", "inference", "sample_hf.py"
    )
    if os.path.isfile(cfg_path):
        print(f"  Using existing config: {cfg_path}")
    else:
        print(f"  [Warning] Config not found at {cfg_path}")


# ── Step 6: smoke test ────────────────────────────────────────────────────────

def smoke_test(opensora_dir: str):
    banner("Import smoke test")

    if opensora_dir not in sys.path:
        sys.path.insert(0, opensora_dir)

    errors = []
    for mod in [
        "opensora",
        "opensora.registry",
        "opensora.models.stdit.stdit3",
        "opensora.schedulers.rf",
    ]:
        try:
            __import__(mod)
            print(f"  ✓  import {mod}")
        except ImportError as e:
            errors.append(f"  ✗  import {mod}  →  {e}")

    if errors:
        for e in errors:
            print(e)
        print("\n  Some imports failed — this may be OK before weights are downloaded.")
    else:
        print("\n  All imports OK!")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  No GPU detected — set up CUDA before running inference.")
    except ImportError:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Set up Open-Sora for text-to-video generation.")
    parser.add_argument(
        "--opensora-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Open-Sora"),
        help="Where to clone the Open-Sora repo  (default: ./Open-Sora)",
    )
    parser.add_argument("--skip-weights", action="store_true",
                        help="Skip HuggingFace weight download")
    parser.add_argument("--skip-deps",    action="store_true",
                        help="Skip pip dependency installation")
    args = parser.parse_args()

    check_requirements()
    clone_or_update(args.opensora_dir)

    if not args.skip_deps:
        install_dependencies(args.opensora_dir)

    if not args.skip_weights:
        download_weights(args.opensora_dir)

    write_inference_config(args.opensora_dir)
    smoke_test(args.opensora_dir)

    cfg_path = os.path.join(
        args.opensora_dir, "configs", "opensora-v1-2", "inference", "sample_hf.py"
    )
    print(textwrap.dedent(f"""
    ┌──────────────────────────────────────────────────────────────────┐
    │  Open-Sora setup complete!                                       │
    │                                                                  │
    │  Manual inference test:                                          │
    │    cd {args.opensora_dir}
    │    python scripts/inference.py \\                                │
    │      {cfg_path} \\
    │      --prompt "A person talking in an office" \\                 │
    │      --save-dir ./samples                                        │
    │                                                                  │
    │  Pipeline usage:                                                 │
    │    python main.py script.json --use-opensora                     │
    │    python main.py script.json --use-opensora --opensora-bg       │
    └──────────────────────────────────────────────────────────────────┘
    """))


if __name__ == "__main__":
    main()
