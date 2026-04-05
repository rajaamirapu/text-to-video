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
import urllib.parse

import requests
from PIL import Image


OLLAMA_URL = "http://localhost:11434"


def generate_image(prompt: str, model: str, width: int, height: int) -> Image.Image:
    """
    Call Ollama and return a PIL image.
    Tries multiple request formats to handle different model APIs.
    """
    print(f"  → Prompt: {prompt[:90]}…")

    # ── Attempt 1: standard /api/generate with JSON response ─────────────────
    for payload in [
        # format A — plain generate
        {"model": model, "prompt": prompt, "stream": False},
        # format B — with explicit size options (some models support this)
        {"model": model, "prompt": prompt, "stream": False,
         "options": {"width": width, "height": height}},
        # format C — images key in request (some vision/gen models)
        {"model": model, "prompt": prompt, "stream": False,
         "format": "json"},
    ]:
        try:
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
            print(f"  → HTTP {r.status_code}  content-type: {r.headers.get('Content-Type','?')}")

            # Check if response is raw image bytes (some models return binary)
            ct = r.headers.get("Content-Type", "")
            if "image" in ct:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                return img.resize((width, height), Image.LANCZOS)

            # Try JSON parse
            raw = r.text.strip()
            if not raw:
                print("  → Empty response body, trying next format …")
                continue

            print(f"  → Response preview: {raw[:200]}")
            data = r.json()

            # images[] field (standard Ollama image gen response)
            images = data.get("images") or []
            if images:
                img_bytes = base64.b64decode(images[0])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return img.resize((width, height), Image.LANCZOS)

            # response field might contain base64 image directly
            resp_field = data.get("response", "")
            if resp_field and len(resp_field) > 100:
                try:
                    img_bytes = base64.b64decode(resp_field)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    return img.resize((width, height), Image.LANCZOS)
                except Exception:
                    pass

            print(f"  → No images in response. Keys: {list(data.keys())}")
            break   # got a valid JSON response but no images — stop trying formats

        except requests.exceptions.HTTPError as e:
            print(f"  → HTTP error: {e}")
            break
        except json.JSONDecodeError:
            # Not JSON — maybe raw binary image?
            if len(r.content) > 1000:
                try:
                    img = Image.open(io.BytesIO(r.content)).convert("RGB")
                    return img.resize((width, height), Image.LANCZOS)
                except Exception:
                    pass
            print(f"  → Non-JSON response ({len(r.content)} bytes), trying next …")
            continue

    # ── Attempt 2: /api/chat with image generation role ──────────────────────
    print("  → Trying /api/chat endpoint …")
    try:
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=chat_payload, timeout=300)
        print(f"  → HTTP {r.status_code}")
        raw = r.text.strip()
        if raw:
            print(f"  → Response preview: {raw[:200]}")
            data = r.json()
            images = (data.get("message", {}).get("images") or
                      data.get("images") or [])
            if images:
                img_bytes = base64.b64decode(images[0])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return img.resize((width, height), Image.LANCZOS)
    except Exception as e:
        print(f"  → /api/chat failed: {e}")

    raise RuntimeError(
        f"\nModel '{model}' did not return an image.\n"
        f"Check that it is an image-generation model and is fully loaded.\n"
        f"Run:  ollama run {model} 'generate image of a red apple'\n"
        f"to verify it works before using this script."
    )


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
