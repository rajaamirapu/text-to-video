#!/usr/bin/env python3
"""
fix_numpy2.py — Fix "numpy.core.multiarray failed to import"
=============================================================

NumPy 2.x removed the internal numpy.core package.
scipy, librosa, and Wav2Lip were compiled against NumPy 1.x and crash
with that ImportError when NumPy ≥ 2.0 is installed.

This script:
  1. Downgrades numpy to the latest 1.x wheel (pre-built binary, no compilation)
  2. Reinstalls scipy against the pinned numpy
  3. Reinstalls librosa / soundfile for good measure

Run once, then retry:
    python fix_numpy2.py
    python main.py example_dialogue.json --output my_video.mp4
"""

import subprocess
import sys


def run(*args, **kwargs):
    print("  $", " ".join(args[0]))
    subprocess.run(*args, check=True, **kwargs)


def pip(*packages, extra_flags=None):
    cmd = [sys.executable, "-m", "pip", "install"] + list(packages)
    if extra_flags:
        cmd += extra_flags
    run(cmd)


print("=" * 60)
print("  Fixing NumPy 2.x incompatibility")
print("=" * 60)

# Step 1 — force a binary numpy 1.x install
print("\n[1/3] Downgrading numpy to <2.0 (binary wheel only) …")
pip(
    "numpy>=1.26.0,<2.0",
    extra_flags=["--force-reinstall", "--only-binary=numpy"],
)

# Step 2 — reinstall scipy (binary) so it links against the correct numpy
print("\n[2/3] Reinstalling scipy …")
pip(
    "scipy>=1.11.0",
    extra_flags=["--force-reinstall", "--only-binary=scipy"],
)

# Step 3 — reinstall librosa + soundfile
print("\n[3/3] Reinstalling librosa + soundfile …")
pip(
    "librosa>=0.10.0",
    "soundfile>=0.12.1",
    extra_flags=["--force-reinstall"],
)

print("\n" + "=" * 60)
print("  Done — numpy.core error should be resolved.")
print("  Run:  python main.py example_dialogue.json --output my_video.mp4")
print("=" * 60)
