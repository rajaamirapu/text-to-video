#!/usr/bin/env python3
"""
download_weights.py
────────────────────────────────────────────────────────────────────
Standalone script to download Wav2Lip model weights.
Completely replaces the gdown-based approach (which hits Google Drive
quota limits).

Run:
    python download_weights.py

Tries in order:
  1. HuggingFace Hub  (huggingface_hub library, no quota)
  2. Direct HTTP with cookie bypass  (requests)
  3. wget with two-pass cookie handling
  4. Prints clear manual download instructions
────────────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
import subprocess
import urllib.request

# ── targets ──────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR = os.path.join("Wav2Lip", "checkpoints")

WEIGHTS = [
    {
        "name":         "wav2lip_gan.pth",          # higher quality (GAN)
        "dest":         os.path.join(CHECKPOINTS_DIR, "wav2lip_gan.pth"),
        "gdrive_id":    "1H8cjvMi7pqCz7vdCjMbS4n3f5qJamzqW",
        "hf_repo":      "numz/wav2lip_studio",
        "hf_filename":  "Wav2Lip/wav2lip_gan.pth",
    },
    {
        "name":         "wav2lip.pth",              # faster (no GAN)
        "dest":         os.path.join(CHECKPOINTS_DIR, "wav2lip.pth"),
        "gdrive_id":    "1ZwMQvzBf3IbQHePkZAi4RvpBGYsJ2n2w",
        "hf_repo":      "numz/wav2lip_studio",
        "hf_filename":  "Wav2Lip/wav2lip.pth",
    },
]

MIN_SIZE = 100 * 1024 * 1024   # 100 MB – sanity check (real files are ~420 MB)


# ── helpers ───────────────────────────────────────────────────────────────────

def ok(dest: str) -> bool:
    """Return True if the file already exists and looks valid."""
    return os.path.isfile(dest) and os.path.getsize(dest) > MIN_SIZE


def progress(count, block, total):
    if total > 0:
        pct = min(100, count * block * 100 // total)
        mb  = count * block // (1024 * 1024)
        print(f"    {pct:3d}%  {mb} MB downloaded", end="\r", flush=True)


# ── method 1: huggingface_hub ─────────────────────────────────────────────────

def try_hf_hub(w: dict) -> bool:
    print("  [1/3] HuggingFace Hub …")
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        path = hf_hub_download(
            repo_id=w["hf_repo"],
            filename=w["hf_filename"],
            local_dir=CHECKPOINTS_DIR,
            local_dir_use_symlinks=False,
        )
        # hf_hub_download saves with subdirectory structure; move if needed
        if not os.path.isfile(w["dest"]):
            # look for the file anywhere under CHECKPOINTS_DIR
            for root, _, files in os.walk(CHECKPOINTS_DIR):
                if w["name"] in files:
                    shutil.move(os.path.join(root, w["name"]), w["dest"])
                    break
        if ok(w["dest"]):
            print(f"\n  ✓ {w['name']} via HuggingFace Hub")
            return True
    except Exception as e:
        print(f"\n  [!] HF Hub failed: {e}")
    return False


# ── method 2: requests + Google Drive cookie bypass ───────────────────────────

def try_requests_gdrive(w: dict) -> bool:
    print("  [2/3] requests + Google Drive cookie bypass …")
    try:
        import requests as req
        session = req.Session()
        base    = f"https://drive.google.com/uc?export=download&id={w['gdrive_id']}"

        # First request: may get a confirm page for large files
        r = session.get(base, stream=False, timeout=30)
        token = None

        # Check cookies
        for k, v in r.cookies.items():
            if "download_warning" in k:
                token = v
                break

        # Check HTML for confirm param
        if not token:
            import re
            for pattern in [r'confirm=([0-9A-Za-z_\-]+)', r'"confirm":"([^"]+)"']:
                m = re.search(pattern, r.text)
                if m:
                    token = m.group(1)
                    break

        url2 = f"{base}&confirm={token}" if token else base
        r2   = session.get(url2, stream=True, timeout=600)

        if r2.status_code != 200:
            print(f"\n  [!] HTTP {r2.status_code}")
            return False

        total = int(r2.headers.get("content-length", 0))
        done  = 0
        with open(w["dest"], "wb") as f:
            for chunk in r2.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"    {pct:3d}%  {done//(1024*1024)} MB", end="\r", flush=True)
        print()

        if ok(w["dest"]):
            print(f"  ✓ {w['name']} via requests")
            return True
        else:
            # File too small → probably got an HTML error page, not the model
            sz = os.path.getsize(w["dest"])
            print(f"  [!] Downloaded file is too small ({sz} bytes) — likely an error page")
            os.unlink(w["dest"])
    except Exception as e:
        print(f"\n  [!] requests failed: {e}")
    return False


# ── method 3: wget two-pass ───────────────────────────────────────────────────

def try_wget(w: dict) -> bool:
    if not shutil.which("wget"):
        return False
    print("  [3/3] wget (two-pass cookie) …")
    cookie_file = "/tmp/gdrive_cookies.txt"
    gid = w["gdrive_id"]
    try:
        # Pass 1: get confirm cookie
        subprocess.run([
            "wget", "--quiet",
            "--save-cookies", cookie_file,
            "--keep-session-cookies",
            "--no-check-certificate",
            f"https://drive.google.com/uc?export=download&id={gid}",
            "-O", "/dev/null",
        ], check=True, timeout=30)

        # Extract confirm token from saved cookie (t= parameter)
        confirm = "t"
        try:
            with open(cookie_file) as cf:
                for line in cf:
                    if "download_warning" in line:
                        confirm = line.strip().split("\t")[-1]
                        break
        except Exception:
            pass

        # Pass 2: actual download
        subprocess.run([
            "wget",
            "--load-cookies", cookie_file,
            "--no-check-certificate",
            "--content-disposition",
            f"https://drive.google.com/uc?export=download&confirm={confirm}&id={gid}",
            "-O", w["dest"],
        ], check=True, timeout=3600)

        if os.path.exists(cookie_file):
            os.unlink(cookie_file)

        if ok(w["dest"]):
            print(f"  ✓ {w['name']} via wget")
            return True
    except Exception as e:
        print(f"  [!] wget failed: {e}")
    return False


# ── manual fallback ───────────────────────────────────────────────────────────

def print_manual_instructions(w: dict):
    abs_dest = os.path.abspath(w["dest"])
    gid      = w["gdrive_id"]
    print(f"""
  ┌────────────────────────────────────────────────────────────────────────┐
  │  ⚠  All automated downloads failed for: {w['name']:<30}│
  ├────────────────────────────────────────────────────────────────────────┤
  │  Google Drive has hit its public download quota for this file.         │
  │                                                                        │
  │  OPTION A — Download in your browser (easiest):                        │
  │    https://drive.google.com/file/d/{gid}/view      │
  │    Click "Download anyway" when Google warns about file size.          │
  │                                                                        │
  │  OPTION B — gdown with your Google account cookie:                     │
  │    1. Install gdown:  pip install gdown                                │
  │    2. Run:  gdown --fuzzy \\                                            │
  │      "https://drive.google.com/file/d/{gid}/view"  │
  │                                                                        │
  │  OPTION C — Direct pip approach:                                       │
  │    pip install huggingface_hub                                         │
  │    python -c "from huggingface_hub import hf_hub_download; \\           │
  │    hf_hub_download('numz/wav2lip_studio', \\                            │
  │    'Wav2Lip/{w['name']}', local_dir='Wav2Lip/checkpoints')"            │
  │                                                                        │
  │  After downloading, place the file at:                                 │
  │    {abs_dest:<70}│
  │                                                                        │
  │  Then re-run:  python download_weights.py                              │
  └────────────────────────────────────────────────────────────────────────┘
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print("\n" + "═" * 70)
    print("  Wav2Lip Weight Downloader")
    print("  (bypasses gdown Google Drive quota limits)")
    print("═" * 70)

    all_ok = True
    for w in WEIGHTS:
        print(f"\n── {w['name']} ──")

        if ok(w["dest"]):
            sz = os.path.getsize(w["dest"]) // (1024 * 1024)
            print(f"  ✓ Already present ({sz} MB) — skipping")
            continue

        downloaded = (
            try_hf_hub(w)
            or try_requests_gdrive(w)
            or try_wget(w)
        )

        if not downloaded:
            print_manual_instructions(w)
            all_ok = False

    print("\n" + "═" * 70)
    if all_ok:
        print("  ✓ All weights downloaded successfully!")
        print("  Next step:  python setup_models.py --skip-weights")
    else:
        print("  ⚠  Some weights need manual download (see instructions above).")
        print("  After placing them, re-run:  python download_weights.py")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
