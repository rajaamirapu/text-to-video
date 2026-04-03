#!/usr/bin/env python3
"""
fix_python312.py
────────────────────────────────────────────────────────────────────
One-shot fix for:
  NameError: name 'CCompiler' is not defined

Root cause
----------
Python 3.12 removed distutils entirely. Wav2Lip pins librosa==0.8.0
which pulls in an old numba that still imports distutils.ccompiler.CCompiler.

What this script does
---------------------
1. Installs setuptools  (restores distutils shim for legacy packages)
2. Upgrades numba  → 0.59+  (first version with full Python 3.12 support)
3. Upgrades librosa → 0.10+ (compatible with new numba)
4. Patches Wav2Lip/audio.py to use the librosa 0.10 API
   (load() no longer accepts 'res_type'; mel/stft API is unchanged)
5. Rewrites Wav2Lip/requirements.txt with compatible pin ranges
   so re-installing won't downgrade the packages again

Run once:
    python fix_python312.py
"""

from __future__ import annotations
import os
import re
import subprocess
import sys


WAV2LIP_DIR = "Wav2Lip"


def run(cmd: list[str], **kw):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def pip(*packages: str):
    run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--only-binary=numpy,numba",   # never compile from source
        *packages,
    ])


# ── step 1: core fixes ────────────────────────────────────────────────────────

def fix_packages():
    print("\n── Step 1/3: Install Python 3.12-compatible packages ──────────────")
    pip(
        "setuptools",          # restores distutils shim
        "wheel",
        "numba>=0.59.0",       # first numba release with Python 3.12 support
        "librosa>=0.10.0",     # compatible with new numba
        "soundfile>=0.12.1",   # our primary audio loader (no numba at all)
        "numpy>=1.26.0,<2.0",    # stay on 1.x for Wav2Lip compatibility
    )
    print("  ✓ Packages updated")


# ── step 2: patch Wav2Lip/audio.py ───────────────────────────────────────────

AUDIO_PY_PATH = os.path.join(WAV2LIP_DIR, "audio.py")

# librosa 0.10 removed the `res_type` param from load()
# and changed some internal APIs. Patch only the breaking calls.
PATCHES = [
    # load(..., res_type='kaiser_best')  →  load(...)
    (
        r",\s*res_type\s*=\s*['\"]kaiser_best['\"]",
        "",
    ),
    (
        r",\s*res_type\s*=\s*['\"]kaiser_fast['\"]",
        "",
    ),
    # librosa.filters.mel signature changed in 0.10 (sr, n_fft, ...)
    # nothing needed — keyword args still work
]


def patch_wav2lip_audio():
    print("\n── Step 2/3: Patch Wav2Lip/audio.py for librosa 0.10 ──────────────")
    if not os.path.isfile(AUDIO_PY_PATH):
        print(f"  [!] {AUDIO_PY_PATH} not found — skipping (run after cloning Wav2Lip)")
        return

    original = open(AUDIO_PY_PATH, encoding="utf-8").read()
    patched  = original

    for pattern, replacement in PATCHES:
        patched = re.sub(pattern, replacement, patched)

    if patched == original:
        print("  ✓ No changes needed (already compatible)")
        return

    # Back up original
    backup = AUDIO_PY_PATH + ".bak"
    if not os.path.exists(backup):
        open(backup, "w").write(original)
        print(f"  Backed up original → {backup}")

    open(AUDIO_PY_PATH, "w", encoding="utf-8").write(patched)
    print("  ✓ Patched Wav2Lip/audio.py")


# ── step 3: rewrite Wav2Lip/requirements.txt ─────────────────────────────────

WAV2LIP_REQ = os.path.join(WAV2LIP_DIR, "requirements.txt")

COMPATIBLE_REQS = """\
# Wav2Lip requirements — patched for Python 3.12 compatibility
librosa>=0.10.0
numpy>=1.26.0,<2.0
scipy>=1.11
opencv-python>=4.8
tqdm>=4.65
numba>=0.59.0
torch>=2.0
torchvision>=0.15
"""


def fix_wav2lip_requirements():
    print("\n── Step 3/3: Update Wav2Lip/requirements.txt ───────────────────────")
    if not os.path.isdir(WAV2LIP_DIR):
        print(f"  [!] {WAV2LIP_DIR}/ not found — skipping (clone it first)")
        return

    if os.path.isfile(WAV2LIP_REQ):
        backup = WAV2LIP_REQ + ".bak"
        if not os.path.exists(backup):
            import shutil
            shutil.copy(WAV2LIP_REQ, backup)
            print(f"  Backed up original → {backup}")

    open(WAV2LIP_REQ, "w").write(COMPATIBLE_REQS)
    print("  ✓ Wav2Lip/requirements.txt updated")


# ── step 4: quick smoke test ──────────────────────────────────────────────────

def smoke_test():
    print("\n── Smoke test ──────────────────────────────────────────────────────")
    tests = [
        ("setuptools / distutils shim",
         "import setuptools; import distutils; print('  distutils OK')"),
        ("numba",
         "import numba; print(f'  numba {numba.__version__} OK')"),
        ("librosa",
         "import librosa; print(f'  librosa {librosa.__version__} OK')"),
        ("soundfile",
         "import soundfile; print(f'  soundfile OK')"),
        ("lip_sync (our module)",
         "import sys; sys.path.insert(0,'.'); from lip_sync import extract_mouth_openings; print('  lip_sync OK')"),
    ]
    all_ok = True
    for label, code in tests:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout.rstrip())
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown"
            print(f"  [!] {label}: {err}")
            all_ok = False
    return all_ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 65)
    print("  Python 3.12 Compatibility Fix for Wav2Lip + librosa/numba")
    print("═" * 65)

    fix_packages()
    patch_wav2lip_audio()
    fix_wav2lip_requirements()
    ok = smoke_test()

    print("\n" + "═" * 65)
    if ok:
        print("  ✓ All fixes applied successfully!")
        print("  You can now run:  python main.py example_dialogue.json")
    else:
        print("  ⚠  Some issues remain — check errors above.")
        print("  Try:  pip install setuptools numba>=0.59 librosa>=0.10")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
