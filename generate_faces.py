#!/usr/bin/env python3
"""
generate_faces.py

Generate face images for all characters using a specific Ollama image model.

Usage:
    python generate_faces.py                          # uses x/z-image-turbo
    python generate_faces.py --model some-other-model
    python generate_faces.py --script my_dialogue.json
"""

import argparse
import base64
import io
import json
import os
import sys

import requests
from PIL import Image


OLLAMA_URL = "http://localhost:11434"


def generate_image(prompt: str, model: str, width: int, height: int) -> Image.Image:
    """Call Ollama /api/generate and return a PIL image."""
    print(f"  → Prompt: {prompt[:80]}…")
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()

    images = data.get("images") or []
    if not images:
        raise RuntimeError(
            f"Model '{model}' returned no images. "
            "Make sure it is an image-generation model (e.g. x/z-image-turbo)."
        )

    img_bytes = base64.b64decode(images[0])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img.resize((width, height), Image.LANCZOS)


def portrait_prompt(name: str, role: str, gender: str) -> str:
    gender_word  = "man" if gender == "male" else "woman"
    facing       = "slightly right" if gender == "female" else "slightly left"
    return (
        f"photorealistic portrait headshot of a {gender_word}, {role}, named {name}, "
        f"facing {facing}, natural expression, professional business casual attire, "
        f"soft studio lighting, shallow depth of field, sharp focus on face, "
        f"plain light grey background, 4K DSLR quality, ultra realistic"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="x/z-image-turbo",
                        help="Ollama image model to use")
    parser.add_argument("--script", default="example_dialogue.json",
                        help="Dialogue JSON (to read character list)")
    parser.add_argument("--faces-dir", default="faces",
                        help="Output directory for face images")
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--regen", action="store_true",
                        help="Regenerate even if cached image exists")
    args = parser.parse_args()

    # Load characters from JSON
    with open(args.script, encoding="utf-8") as f:
        data = json.load(f)
    characters = data.get("characters", {})

    os.makedirs(args.faces_dir, exist_ok=True)

    print(f"\nOllama model : {args.model}")
    print(f"Output dir   : {args.faces_dir}/")
    print(f"Characters   : {list(characters.keys())}\n")

    for name, info in characters.items():
        role   = info.get("role",   "person")
        gender = info.get("gender", "neutral")
        safe   = name.lower().replace(" ", "_")
        out    = os.path.join(args.faces_dir, f"face_{safe}.png")

        if os.path.isfile(out) and not args.regen:
            print(f"[{name}] Skipping — already exists: {out}")
            print(f"         (use --regen to regenerate)")
            continue

        print(f"[{name}] Generating {gender} {role} portrait …")
        try:
            prompt = portrait_prompt(name, role, gender)
            img    = generate_image(prompt, args.model, args.width, args.height)
            img.save(out)
            print(f"[{name}] ✓ Saved → {out}  ({img.size[0]}×{img.size[1]})\n")
        except Exception as e:
            print(f"[{name}] ✗ Failed: {e}\n")
            sys.exit(1)

    print("Done! All face images generated.")
    print(f"\nRun the pipeline:")
    print(f"  python main.py example_dialogue.json --output my_video.mp4")


if __name__ == "__main__":
    main()
