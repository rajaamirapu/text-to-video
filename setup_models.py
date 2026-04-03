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
import shutil
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
    run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--only-binary=numpy,numba",   # never compile numpy/numba from source
        *packages,
    ])


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

    # ── Python 3.12: fix distutils/CCompiler BEFORE installing anything ───────
    # setuptools provides the distutils shim that numba/librosa 0.8 need.
    # numba>=0.59 and librosa>=0.10 are the first versions with native 3.12 support.
    print("  Applying Python 3.12 compatibility fixes …")
    pip_install("setuptools", "wheel")
    pip_install("numba>=0.59.0", "librosa>=0.10.0", "soundfile>=0.12.1")

    # ── Patch Wav2Lip's pinned requirements BEFORE pip reads them ─────────────
    req = os.path.join(WAV2LIP_DIR, "requirements.txt")
    _patch_wav2lip_requirements(req)

    # ── Now install Wav2Lip deps (uses our patched requirements.txt) ──────────
    if os.path.isfile(req):
        run([sys.executable, "-m", "pip", "install", "-r", req])

    # ── Patch Wav2Lip/audio.py for librosa 0.10 API changes ──────────────────
    _patch_wav2lip_audio()

    # ── Additional pipeline packages ──────────────────────────────────────────
    pip_install(
        "gdown",            # Google Drive downloader
        "diffusers>=0.29.0",        # Stable Diffusion
        "transformers>=4.41.0",     # must match diffusers (CLIPImageProcessor)
        "accelerate>=0.30.0",
        "xformers",         # optional memory optimisation
        "gtts",
        "pyttsx3",
        "moviepy",
        "Pillow",
        "numpy>=1.26.0,<2.0",
        "requests",
        "tqdm",
        "huggingface_hub",
    )
    print("  ✓ All dependencies installed")


def _patch_wav2lip_requirements(req_path: str):
    """Replace Wav2Lip's pinned librosa==0.8 with a Python 3.12-safe range."""
    if not os.path.isfile(req_path):
        return
    original = open(req_path).read()
    if "librosa>=0.10" in original:
        return  # already patched

    # Back up only once
    backup = req_path + ".bak"
    if not os.path.exists(backup):
        open(backup, "w").write(original)

    import re
    patched = original
    # Replace any pinned librosa version with a safe range
    patched = re.sub(r"librosa[=<>!~][^\n]*", "librosa>=0.10.0", patched)
    # Replace any pinned numba version
    patched = re.sub(r"numba[=<>!~][^\n]*", "numba>=0.59.0", patched)
    open(req_path, "w").write(patched)
    print("  ✓ Patched Wav2Lip/requirements.txt (librosa + numba pins removed)")


def _patch_wav2lip_audio():
    """Remove the res_type= kwarg that librosa 0.10 dropped."""
    import re
    audio_py = os.path.join(WAV2LIP_DIR, "audio.py")
    if not os.path.isfile(audio_py):
        return
    src = open(audio_py, encoding="utf-8").read()
    patched = re.sub(r",\s*res_type\s*=\s*['\"][^'\"]*['\"]", "", src)
    if patched != src:
        backup = audio_py + ".bak"
        if not os.path.exists(backup):
            open(backup, "w").write(src)
        open(audio_py, "w", encoding="utf-8").write(patched)
        print("  ✓ Patched Wav2Lip/audio.py (removed deprecated res_type kwarg)")


def step_download_wav2lip_weights():
    banner("3 / 5  Download Wav2Lip model weights")
    checkpoints_dir = os.path.join(WAV2LIP_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    gan_path = os.path.join(checkpoints_dir, "wav2lip_gan.pth")
    std_path = os.path.join(checkpoints_dir, "wav2lip.pth")

    _download_weight(
        name="wav2lip_gan.pth",
        dest=gan_path,
        gdrive_id=WAV2LIP_GAN_GDRIVE_ID,
        gdrive_share_url=f"https://drive.google.com/file/d/{WAV2LIP_GAN_GDRIVE_ID}/view?usp=sharing",
    )
    _download_weight(
        name="wav2lip.pth",
        dest=std_path,
        gdrive_id=WAV2LIP_GDRIVE_ID,
        gdrive_share_url=f"https://drive.google.com/file/d/{WAV2LIP_GDRIVE_ID}/view?usp=sharing",
    )
    print("  ✓ Wav2Lip weights ready")


def _download_weight(name: str, dest: str, gdrive_id: str, gdrive_share_url: str):
    """
    Try multiple strategies to download a Wav2Lip model weight file.
    Falls back to clear manual instructions if all automated methods fail.
    """
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  ✓ {name} already present  ({os.path.getsize(dest) // (1024*1024)} MB)")
        return

    print(f"\n  Downloading {name} …")
    downloaded = False

    # ── Method 1: gdown fuzzy (handles virus-scan redirect) ──────────────────
    try:
        import gdown  # type: ignore
        print("  [1/3] Trying gdown (fuzzy mode) …")
        gdown.download(gdrive_share_url, dest, quiet=False, fuzzy=True)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"  ✓ {name} downloaded via Google Drive")
            downloaded = True
    except Exception as e:
        print(f"  [1/3] gdown failed: {type(e).__name__}: {str(e)[:120]}")

    # ── Method 2: requests + manual cookie bypass ─────────────────────────────
    if not downloaded:
        print("  [2/3] Trying requests with cookie bypass …")
        try:
            import requests as _req
            session = _req.Session()
            direct  = f"https://drive.google.com/uc?export=download&id={gdrive_id}"
            r = session.get(direct, stream=True, timeout=30)

            # Google shows a confirm page for large files — find the token
            confirm_token = None
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    confirm_token = v
                    break
            # Also search HTML for confirm token
            if not confirm_token and "confirm" in r.text:
                import re
                m = re.search(r'confirm=([0-9A-Za-z_]+)', r.text)
                if m:
                    confirm_token = m.group(1)

            if confirm_token:
                url2 = f"{direct}&confirm={confirm_token}"
                r = session.get(url2, stream=True, timeout=60)

            # Stream to disk
            total = int(r.headers.get("content-length", 0))
            done  = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(32768):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"    {done*100//total}%  ({done//(1024*1024)} MB)",
                                  end="\r", flush=True)
            print()
            if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
                print(f"  ✓ {name} downloaded via requests")
                downloaded = True
            else:
                os.unlink(dest)   # remove partial/HTML stub
        except Exception as e:
            print(f"  [2/3] requests failed: {e}")

    # ── Method 3: wget with cookies ───────────────────────────────────────────
    if not downloaded and shutil.which("wget"):
        print("  [3/3] Trying wget …")
        try:
            cookie_jar = dest + ".cookies"
            # Step 1: get confirmation cookie
            subprocess.run([
                "wget", "--quiet", "--save-cookies", cookie_jar,
                "--keep-session-cookies", "--no-check-certificate",
                f"https://drive.google.com/uc?export=download&id={gdrive_id}",
                "-O", "/dev/null",
            ], check=True)
            # Step 2: actual download
            subprocess.run([
                "wget", "--load-cookies", cookie_jar,
                "--no-check-certificate", "--content-disposition",
                f"https://drive.google.com/uc?export=download&confirm=t&id={gdrive_id}",
                "-O", dest,
            ], check=True)
            if os.path.isfile(cookie_jar):
                os.unlink(cookie_jar)
            if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
                print(f"  ✓ {name} downloaded via wget")
                downloaded = True
        except Exception as e:
            print(f"  [3/3] wget failed: {e}")

    # ── All automated methods failed — clear manual instructions ──────────────
    if not downloaded:
        abs_dest = os.path.abspath(dest)
        print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  ⚠  MANUAL DOWNLOAD NEEDED: {name:<35}│
  ├─────────────────────────────────────────────────────────────────┤
  │  Google Drive is rate-limiting automated downloads.             │
  │                                                                 │
  │  Step 1 — Open this link in your browser:                       │
  │  {gdrive_share_url:<65}│
  │                                                                 │
  │  Step 2 — Click "Download anyway" if Google warns about size.   │
  │                                                                 │
  │  Step 3 — Move the downloaded file here:                        │
  │  {abs_dest:<65}│
  │                                                                 │
  │  Then re-run:  python setup_models.py                           │
  └─────────────────────────────────────────────────────────────────┘
""")


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
    import argparse
    parser = argparse.ArgumentParser(description="Set up Text-to-Video models")
    parser.add_argument(
        "--skip-sd", action="store_true",
        help="Skip Stable Diffusion pre-download (downloads on first run instead)"
    )
    parser.add_argument(
        "--skip-weights", action="store_true",
        help="Skip Wav2Lip weight download (use if you already ran download_weights.py)"
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

    if args.skip_weights:
        print("\n  [3/5] Wav2Lip weights — skipped (--skip-weights)")
        print("        Run  python download_weights.py  if weights are missing.")
    else:
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
