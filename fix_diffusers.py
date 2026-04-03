#!/usr/bin/env python3
"""
fix_diffusers.py
────────────────────────────────────────────────────────────────────
Fixes:
  RuntimeError: Could not import module 'CLIPImageProcessor'.
               Are this object's requirements defined correctly?

Root cause
----------
`CLIPImageProcessor` lives in `transformers`.  diffusers uses lazy
loading to import it, but if `transformers` is missing, outdated, or
its version doesn't match what the installed `diffusers` expects, the
lazy import silently records a broken reference — which then blows up
the first time you actually call the pipeline.

Fix
---
Install diffusers and transformers together at a pinned compatible
pairing, plus accelerate which is required for device dispatch.

Run:
    python fix_diffusers.py
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import subprocess
import sys


def run(cmd: list[str]):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pip(*pkgs: str):
    run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--only-binary=:all:",    # wheels only — no source compilation
        *pkgs,
    ])


def pip_src(*pkgs: str):
    """Fallback: allow source builds for packages without wheels."""
    run([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])


def current_version(pkg: str) -> str | None:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", pkg],
        capture_output=True, text=True,
    )
    for line in r.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def smoke_test() -> bool:
    print("\n── Smoke test ──────────────────────────────────────────────────────")
    tests = [
        ("transformers + CLIPImageProcessor",
         "from transformers import CLIPImageProcessor; "
         "print(f'  CLIPImageProcessor OK  (transformers {__import__(\"transformers\").__version__})')"),
        ("diffusers StableDiffusionPipeline",
         "from diffusers import StableDiffusionPipeline; "
         "print(f'  StableDiffusionPipeline OK  (diffusers {__import__(\"diffusers\").__version__})')"),
        ("accelerate",
         "import accelerate; print(f'  accelerate {accelerate.__version__} OK')"),
    ]
    ok = True
    for label, code in tests:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(r.stdout.rstrip())
        else:
            err = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
            print(f"  [!] {label}: {err}")
            ok = False
    return ok


def main():
    print("═" * 65)
    print("  Diffusers / Transformers Compatibility Fix")
    print("═" * 65)

    # Show what's currently installed
    print("\n  Current versions:")
    for pkg in ("diffusers", "transformers", "accelerate", "tokenizers"):
        v = current_version(pkg)
        print(f"    {pkg:<20} {v or 'NOT INSTALLED'}")

    print("\n── Installing compatible versions ──────────────────────────────────")
    print("  Target: diffusers>=0.29.0  +  transformers>=4.41.0")
    print("  (CLIPImageProcessor is in transformers; both must match)\n")

    # Install tokenizers first (rust-based; wheels available)
    try:
        pip("tokenizers>=0.19.0")
    except Exception:
        pip_src("tokenizers>=0.19.0")

    # Install the core trio together so pip resolves their mutual constraints
    try:
        pip(
            "diffusers>=0.29.0",
            "transformers>=4.41.0",
            "accelerate>=0.30.0",
        )
    except subprocess.CalledProcessError:
        print("  [!] Binary install failed — retrying without --only-binary ...")
        pip_src(
            "diffusers>=0.29.0",
            "transformers>=4.41.0",
            "accelerate>=0.30.0",
        )

    # xformers is optional — don't let it block the fix
    print("\n  Installing xformers (optional, speeds up GPU inference) …")
    try:
        pip("xformers")
        print("  ✓ xformers installed")
    except Exception as e:
        print(f"  [skipped] xformers: {e}")

    print("\n  Updated versions:")
    for pkg in ("diffusers", "transformers", "accelerate"):
        v = current_version(pkg)
        print(f"    {pkg:<20} {v or 'NOT INSTALLED'}")

    ok = smoke_test()

    print("\n" + "═" * 65)
    if ok:
        print("  ✓ Fix applied — re-run:")
        print("    python main.py example_dialogue.json --output my_video.mp4")
    else:
        print("  ⚠  Some imports still failing. Try:")
        print("    pip install --upgrade diffusers transformers accelerate")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
