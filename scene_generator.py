"""
scene_generator.py

Uses Ollama image-generation models to produce:
  1. A photorealistic room/lounge background
  2. A seated full-body character image for each person

Falls back to PIL rendering if no image-capable model is found.

Supported Ollama image models (auto-detected):
  • x/stable-diffusion   (SD via Ollama)
  • hf.co/…/FLUX…        (FLUX models pulled from HuggingFace)
  • Any model whose /api/generate response contains an "images" field
"""

from __future__ import annotations
import base64
import hashlib
import io
import os
from typing import Optional

import requests
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Ollama image generation
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_generate_image(
    prompt: str,
    model: str,
    ollama_url: str,
    width: int,
    height: int,
    timeout: int = 180,
) -> Optional[Image.Image]:
    """
    Call Ollama /api/generate and return the first image in the response.
    Returns None if the model doesn't support image output.
    """
    try:
        payload = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": -1,
            },
        }
        r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        raw_images = data.get("images") or []
        if not raw_images:
            return None

        img_bytes = base64.b64decode(raw_images[0])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        return img

    except Exception as e:
        print(f"  [SceneGen] Ollama image attempt failed: {e}")
        return None


def find_image_model(ollama_url: str) -> Optional[str]:
    """
    Return the best Ollama model name for image generation, or None.

    Prefers models whose names hint at image generation
    (flux, stable-diffusion, imagen, sd, draw).
    Falls back to probing ALL installed models with a tiny test prompt.
    """
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None

    hints    = ["flux", "stable", "diffusion", "sd", "imagen", "draw", "turbo", "z-image", "x-image"]
    priority = [m for m in models if any(h in m.lower() for h in hints)]
    rest     = [m for m in models if m not in priority]

    test_prompt = "a red circle"
    for model in priority + rest:
        print(f"  [SceneGen] Probing '{model}' for image generation …")
        img = _ollama_generate_image(test_prompt, model, ollama_url, 64, 64, timeout=30)
        if img is not None:
            print(f"  [SceneGen] ✓ Image model found: {model}")
            return model

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scene prompts
# ─────────────────────────────────────────────────────────────────────────────

ROOM_PROMPT = (
    "photorealistic modern office lounge interior, empty room no people, "
    "large floor-to-ceiling windows with lush green foliage visible outside, "
    "monstera plant on left and right side, light grey comfortable sofas, "
    "coffee table with two white mugs, warm diffused natural lighting, "
    "soft shadows, high detail, 4K, cinematic still, architectural photography"
)


def _character_prompt(name: str, role: str, gender: str, appearance: dict) -> str:
    sk    = appearance.get("skin_rgb",  [220, 185, 155])
    hr    = appearance.get("hair_rgb",  [80,  50,  20])
    style = appearance.get("hair_style", "medium")
    skin_word = (
        "fair" if sk[0] > 210 else
        "medium" if sk[0] > 175 else
        "olive" if sk[0] > 140 else "dark"
    )
    facing = "slightly to the right" if name else "slightly to the left"
    return (
        f"photorealistic upper-body portrait of a {gender} {role}, "
        f"{skin_word} skin, {style} hairstyle, "
        f"business casual attire, seated on a light grey sofa, "
        f"facing {facing}, relaxed friendly expression, "
        f"holding a white coffee mug, soft natural window lighting, "
        f"shallow depth of field, 4K DSLR quality, plain background"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_room_background(
    width: int,
    height: int,
    ollama_url: str,
    image_model: Optional[str],
    cache_dir: str = "faces",
    room_bg_path: Optional[str] = None,
) -> Image.Image:
    """
    Return a photorealistic room background image.

    Priority order:
      1. User-supplied --room-bg image
      2. Cached Ollama-generated background (faces/room_bg.png)
      3. Fresh Ollama generation
      4. PIL fallback
    """
    # 1. User-supplied image
    if room_bg_path and os.path.isfile(room_bg_path):
        print(f"  [SceneGen] Using user-supplied room background: {room_bg_path}")
        return Image.open(room_bg_path).convert("RGB").resize((width, height), Image.LANCZOS)

    # 2. Cached generated background
    cache_path = os.path.join(cache_dir, "room_bg.png")
    if os.path.isfile(cache_path):
        print(f"  [SceneGen] Using cached room background: {cache_path}")
        img = Image.open(cache_path).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        return img

    # 3. Ollama image generation
    if image_model:
        print(f"  [SceneGen] Generating room with Ollama model '{image_model}' …")
        img = _ollama_generate_image(ROOM_PROMPT, image_model, ollama_url, width, height)
        if img:
            os.makedirs(cache_dir, exist_ok=True)
            img.save(cache_path)
            print(f"  [SceneGen] ✓ Room background saved → {cache_path}")
            return img

    # 4. PIL fallback
    print("  [SceneGen] No Ollama image model — using PIL room background")
    from video_composer import _make_room_background
    return _make_room_background(width, height)


def generate_character_body(
    name: str,
    role: str,
    gender: str,
    appearance: dict,
    width: int,
    height: int,
    ollama_url: str,
    image_model: Optional[str],
    cache_dir: str = "faces",
    idx: int = 0,
) -> Optional[Image.Image]:
    """
    Generate a full upper-body seated image of a character.
    Returns None if generation fails (caller will use face headshot instead).
    """
    safe  = name.lower().replace(" ", "_")
    cpath = os.path.join(cache_dir, f"body_{safe}.png")

    if os.path.isfile(cpath):
        print(f"  [SceneGen] Using cached body image for '{name}'")
        img = Image.open(cpath).convert("RGB")
        return img.resize((width, height), Image.LANCZOS)

    if not image_model:
        return None

    prompt = _character_prompt(name, role, gender, appearance)
    print(f"  [SceneGen] Generating body for '{name}' with Ollama …")
    img = _ollama_generate_image(prompt, image_model, ollama_url, width, height)
    if img:
        os.makedirs(cache_dir, exist_ok=True)
        img.save(cpath)
        print(f"  [SceneGen] ✓ Body image saved → {cpath}")
    return img
