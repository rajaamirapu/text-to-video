#!/usr/bin/env python3
"""
fix_wav2lip_audio.py — Patch Wav2Lip/audio.py for librosa >= 0.10
==================================================================

librosa 0.10 made two breaking changes that crash Wav2Lip:

  1. librosa.load()         — dropped the  res_type=  keyword argument.
  2. librosa.filters.mel()  — sr and n_fft became keyword-only; passing
                              them positionally raises TypeError.

This script patches Wav2Lip/audio.py in-place (backs up the original first).

Run once, then retry:
    python fix_wav2lip_audio.py
    python main.py example_dialogue.json --output my_video.mp4
"""

import os
import re
import sys

WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "Wav2Lip")
audio_py    = os.path.join(WAV2LIP_DIR, "audio.py")

if not os.path.isfile(audio_py):
    sys.exit(f"[Error] {audio_py} not found. Run from the text-to-video directory.")

src    = open(audio_py, encoding="utf-8").read()
backup = audio_py + ".bak"

if not os.path.exists(backup):
    open(backup, "w", encoding="utf-8").write(src)
    print(f"  Backed up original → {backup}")
else:
    print(f"  Backup already exists → {backup}")

patched = src

# ── Fix 1: remove  res_type='kaiser_best'  (or any res_type=…) ──────────────
before = patched
patched = re.sub(r",\s*res_type\s*=\s*['\"][^'\"]*['\"]", "", patched)
if patched != before:
    print("  ✓ Removed deprecated res_type= argument from librosa.load()")
else:
    print("  – res_type= already removed or not present")

# ── Fix 2: librosa.filters.mel(sr, n_fft, …) → mel(sr=sr, n_fft=n_fft, …) ──
# Old:  librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=…
# New:  librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=…
before = patched
patched = re.sub(
    r"librosa\.filters\.mel\(\s*([^,]+),\s*([^,]+),",
    r"librosa.filters.mel(sr=\1, n_fft=\2,",
    patched,
)
if patched != before:
    print("  ✓ Fixed librosa.filters.mel() positional → keyword arguments")
else:
    print("  – mel() call already uses keyword arguments or not found")

if patched != src:
    open(audio_py, "w", encoding="utf-8").write(patched)
    print(f"\n  Saved patched file → {audio_py}")
else:
    print("\n  No changes needed — file already up-to-date.")

print("\nDone. Retry:  python main.py example_dialogue.json --output my_video.mp4")
