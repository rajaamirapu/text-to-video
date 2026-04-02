#!/usr/bin/env python3
"""
setup_models.py

Run this ONCE before using the pipeline:
    python setup_models.py

What it does
------------
1. Clones the Wav2Lip repository
2. Downloads the Wav2Lip GAN model weights  (via gdown from Google Drive)
3. Downloads the face-detection model weights (s3fd)
4. Installs Python dependencies
5. (Optional) Pre-downloads the Stable Diffusion model from HuggingFace

After this script completes, run:
    python main.py example_dialogue.json
"""

from __future__ import annotations
import os
import subprocess
import sys

# ── model IDs / URLs ──────────────────────────────────────────────────────────

WAV2LIP_REPO   = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_DIR    = "Wav2Lip"

# Google Drive file IDs for Wav2Lip weights
# Source: https://github.com/Rudrabha/Wav2Lip#getting-the-weights
WAV2LIP_GAN_GDRIVE_ID  = "1H8cjvMi7pqCz7vdCjMbS4n3f5qJamzqW"   # wav2lip_gan.pth (~420 MB)
WAV2LIP_GDRIVE_ID      = "1ZwMQvzBf3IbQHePkZAi4RvpBGYsJ2n2w"   # wav2lip.pth (~420 MB)

# Face detection weights (s3fd – used inside Wav2Lip)
S3FD_URL = (
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
)

# Stable Diffusion model (cached by diffusers automatically on first use)
SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "SG161222/Realistic_Vision_V5.1_noVAE")


# ── helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str):
    line = "─" * min(72, len(msg) + 4)
    print(f"\n{line}\n  {msg}\n{line}")


def run(cmd: list[str], **kwargs):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def pip_install(*packages: str):
    run([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


# ── steps ─────────────────────────────────────────────────────────────────────

def step_clone_wav2lip():
    banner("1 / 5  Clone Wav2Lip repository")
    if os.path.isdir(os.path.join(WAV2LIP_DIR, ".git")):
        print("  Already cloned – pulling latest …")
        run(["git", "-C", WAV2LIP_DIR, "pull"])
    else:
        run(["git", "clone", WAV2LIP_REPO, WAV2LIP_DIR])
    print("  ✓ Wav2Lip repository ready")


def step_install_wav2lip_deps():
    banner("2 / 5  Install Wav2Lip Python dependencies")
    req = os.path.join(WAV2LIP_DIR, "requirements.txt")
    if os.path.isfile(req):
        run([sys.executable, "-m", "pip", "install", "-r", req])
    # Additional packages used by our pipeline
    pip_install(
        "gdown",            # Google Drive downloader
        "diffusers",        # Stable Diffusion
        "transformers",
        "accelerate",
        "xformers",         # optional memory optimisation
        "gtts",
        "pyttsx3",
        "librosa",
        "soundfile",
        "scipy",
        "moviepy",
        "Pillow",
        "numpy",
        "requests",
        "tqdm",
    )
    print("  ✓ Dependencies installed")


def step_download_wav2lip_weights():
    banner("3 / 5  Download Wav2Lip model weights")
    checkpoints_dir = os.path.join(WAV2LIP_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    gan_path = os.path.join(checkpoints_dir, "wav2lip_gan.pth")
    std_path = os.path.join(checkpoints_dir, "wav2lip.pth")

    import importlib.util
    has_gdown = importlib.util.find_spec("gdown") is not None
    if not has_gdown:
        pip_install("gdown")

    import gdown  # type: ignore

    if not os.path.isfile(gan_path):
        print("  Downloading wav2lip_gan.pth (~420 MB) …")
        gdown.download(id=WAV2LIP_GAN_GDRIVE_ID, output=gan_path, quiet=False)
    else:
        print(f"  ✓ {gan_path} already exists")

    if not os.path.isfile(std_path):
        print("  Downloading wav2lip.pth (~420 MB) …")
        gdown.download(id=WAV2LIP_GDRIVE_ID, output=std_path, quiet=False)
    else:
        print(f"  ✓ {std_path} already exists")

    print("  ✓ Wav2Lip weights ready")


def step_download_face_detection():
    banner("4 / 5  Download face-detection model (s3fd)")
    dest_dir = os.path.join(
        WAV2LIP_DIR, "face_detection", "detection", "sfd"
    )
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "s3fd-619a316812.pth")

    if os.path.isfile(dest):
        print(f"  ✓ {dest} already exists")
        return

    print("  Downloading s3fd face detection model …")
    try:
        import urllib.request, tqdm  # type: ignore
        from tqdm import tqdm as TQDM

        class _TQDMHook(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with _TQDMHook(unit="B", unit_scale=True, miniters=1, desc="s3fd") as t:
            urllib.request.urlretrieve(S3FD_URL, dest, reporthook=t.update_to)
    except Exception:
        # Fall back to wget / curl if urllib fails
        if shutil.which("wget"):
            run(["wget", "-O", dest, S3FD_URL])
        elif shutil.which("curl"):
            run(["curl", "-L", "-o", dest, S3FD_URL])
        else:
            print(
                f"\n  [Warning] Could not auto-download s3fd.\n"
                f"  Please manually download:\n  {S3FD_URL}\n"
                f"  and save it to:\n  {dest}\n"
            )
            return

    print(f"  ✓ s3fd saved → {dest}")


def step_predownload_sd(model_id: str = SD_MODEL_ID, skip: bool = False):
    banner("5 / 5  Pre-download Stable Diffusion model")
    if skip:
        print("  Skipped (will download on first run)")
        return

    print(f"  Downloading '{model_id}' from HuggingFace …")
    print("  (This may take several minutes — ~4-6 GB depending on the model)")
    try:
        from diffusers import StableDiffusionPipeline  # type: ignore
        import torch
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        print(f"  ✓ '{model_id}' cached")
    except Exception as e:
        print(f"  [Warning] SD pre-download failed: {e}")
        print("  It will be downloaded automatically on first run.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse, shutil
    parser = argparse.ArgumentParser(description="Set up Text-to-Video models")
    parser.add_argument(
        "--skip-sd", action="store_true",
        help="Skip Stable Diffusion pre-download (downloads on first run instead)"
    )
    parser.add_argument(
        "--sd-model", default=SD_MODEL_ID,
        help=f"HuggingFace model ID for face generation (default: {SD_MODEL_ID})"
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  Text-to-Video — Model Setup")
    print("═" * 60)
    print(f"  Wav2Lip dir : {os.path.abspath(WAV2LIP_DIR)}")
    print(f"  SD model    : {args.sd_model}")

    step_clone_wav2lip()
    step_install_wav2lip_deps()
    step_download_wav2lip_weights()
    step_download_face_detection()
    step_predownload_sd(args.sd_model, skip=args.skip_sd)

    print("\n" + "═" * 60)
    print("  ✓ Setup complete!  Run:")
    print("    python main.py example_dialogue.json")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    import shutil
    main()
