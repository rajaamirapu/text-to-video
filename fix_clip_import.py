#!/usr/bin/env python3
"""
fix_clip_import.py
────────────────────────────────────────────────────────────────────
Fixes:
  RuntimeError: Could not import module 'CLIPImageProcessor'

The patch
---------
In older transformers, the class was called CLIPFeatureExtractor.
It was renamed to CLIPImageProcessor in transformers 4.27+.
When diffusers and transformers versions don't agree on the name,
the lazy import in diffusers silently breaks.

This script:
  1. Adds a compatibility shim to transformers so BOTH names always work
  2. Patches every diffusers file that has the broken import
  3. Verifies the fix with a live import test

Run:
    python fix_clip_import.py
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import importlib
import os
import re
import subprocess
import sys


# ── locate packages ───────────────────────────────────────────────────────────

def pkg_dir(name: str) -> str | None:
    """Return the filesystem directory of an installed package."""
    try:
        mod = importlib.import_module(name)
        return os.path.dirname(mod.__file__)
    except Exception:
        return None


# ── step 1: transformers shim ─────────────────────────────────────────────────

def patch_transformers_init():
    """
    Make BOTH CLIPFeatureExtractor and CLIPImageProcessor importable
    regardless of which transformers version is installed.
    """
    print("\n── Step 1/3: transformers compatibility shim ───────────────────────")

    tr_dir = pkg_dir("transformers")
    if not tr_dir:
        print("  [!] transformers not installed — skipping")
        return

    # Check current state
    result = subprocess.run(
        [sys.executable, "-c",
         "from transformers import CLIPImageProcessor; print('CLIPImageProcessor OK')"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  ✓ CLIPImageProcessor already importable from transformers")
        return

    # Try the alias approach
    result2 = subprocess.run(
        [sys.executable, "-c",
         "from transformers import CLIPFeatureExtractor; print('CLIPFeatureExtractor OK')"],
        capture_output=True, text=True,
    )

    if result2.returncode == 0:
        # CLIPFeatureExtractor exists but CLIPImageProcessor doesn't.
        # Patch transformers/__init__.py to expose the alias.
        init_path = os.path.join(tr_dir, "__init__.py")
        if not os.path.isfile(init_path):
            print(f"  [!] {init_path} not found")
            return

        src = open(init_path, encoding="utf-8").read()

        # Don't double-patch
        if "CLIPImageProcessor" in src:
            print("  ✓ Already patched")
            return

        # Append alias at end of file
        alias = (
            "\n# ── compatibility shim (added by fix_clip_import.py) ──\n"
            "try:\n"
            "    from transformers.models.clip.feature_extraction_clip import "
            "CLIPFeatureExtractor as CLIPImageProcessor\n"
            "except ImportError:\n"
            "    pass\n"
        )
        backup = init_path + ".bak"
        if not os.path.exists(backup):
            open(backup, "w", encoding="utf-8").write(src)
        open(init_path, "a", encoding="utf-8").write(alias)
        print(f"  ✓ Appended CLIPImageProcessor alias to {init_path}")
    else:
        print("  [!] Neither CLIPImageProcessor nor CLIPFeatureExtractor found.")
        print("      Run:  pip install transformers>=4.41.0")


# ── step 2: patch diffusers source files ─────────────────────────────────────

IMPORT_PATTERNS = [
    # from transformers import CLIPImageProcessor
    (
        r"from transformers import (.*?)CLIPImageProcessor",
        lambda m: m.group(0).replace(
            "CLIPImageProcessor",
            "CLIPFeatureExtractor as CLIPImageProcessor"
        ) if "CLIPFeatureExtractor" not in m.group(0) else m.group(0),
    ),
    # "CLIPImageProcessor" inside _import_structure dicts (lazy loading)
    # These are string keys — they don't need patching
]

# Safer: insert a try/except alias at the top of every failing file
SHIM = '''\
try:
    from transformers import CLIPImageProcessor
except (ImportError, RuntimeError):
    try:
        from transformers import CLIPFeatureExtractor as CLIPImageProcessor
    except ImportError:
        CLIPImageProcessor = None
'''


def patch_diffusers_files():
    """
    Find every diffusers Python file that imports CLIPImageProcessor
    and add a try/except shim so it works with any transformers version.
    """
    print("\n── Step 2/3: Patch diffusers source files ──────────────────────────")

    diff_dir = pkg_dir("diffusers")
    if not diff_dir:
        print("  [!] diffusers not installed — skipping")
        return

    patched = 0
    for root, _, files in os.walk(diff_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                src = open(fpath, encoding="utf-8").read()
            except Exception:
                continue

            # Only touch files that reference CLIPImageProcessor as an import
            if "CLIPImageProcessor" not in src:
                continue
            if "from transformers import" not in src and \
               "CLIPImageProcessor" not in src:
                continue

            # Replace direct imports with the safe shim
            new_src = re.sub(
                r"^from transformers import ([^\n]*CLIPImageProcessor[^\n]*)\n",
                SHIM + "\n",
                src,
                flags=re.MULTILINE,
            )

            if new_src == src:
                continue  # nothing changed

            backup = fpath + ".bak"
            if not os.path.exists(backup):
                open(backup, "w", encoding="utf-8").write(src)
            open(fpath, "w", encoding="utf-8").write(new_src)
            rel = os.path.relpath(fpath, diff_dir)
            print(f"  ✓ Patched diffusers/{rel}")
            patched += 1

    if patched == 0:
        print("  ✓ No diffusers files needed patching")
    else:
        print(f"  ✓ Patched {patched} file(s)")


# ── step 3: verify ────────────────────────────────────────────────────────────

def verify():
    print("\n── Step 3/3: Verification ──────────────────────────────────────────")
    tests = [
        ("CLIPImageProcessor from transformers",
         "from transformers import CLIPImageProcessor; print('  ✓ CLIPImageProcessor OK')"),
        ("StableDiffusionPipeline from diffusers",
         "from diffusers import StableDiffusionPipeline; print('  ✓ StableDiffusionPipeline OK')"),
        ("face_generator module",
         "import sys; sys.path.insert(0,'.'); import face_generator; print('  ✓ face_generator OK')"),
    ]
    all_ok = True
    for label, code in tests:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(r.stdout.rstrip())
        else:
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            print(f"  [!] {label}: {err}")
            all_ok = False
    return all_ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 65)
    print("  CLIPImageProcessor Import Fix")
    print("═" * 65)

    patch_transformers_init()
    patch_diffusers_files()
    ok = verify()

    print("\n" + "═" * 65)
    if ok:
        print("  ✓ Fix applied successfully!")
        print("  Re-run:")
        print("    python main.py example_dialogue.json --output my_video.mp4")
    else:
        print("  ⚠  Still failing. Run the nuclear option:")
        print("    pip install --upgrade --force-reinstall \\")
        print("        'diffusers>=0.29.0' 'transformers>=4.41.0' 'accelerate>=0.30.0'")
        print("  Then re-run: python fix_clip_import.py")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
