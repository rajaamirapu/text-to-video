"""
scene_generator.py

Uses Stable Diffusion (via diffusers) to generate:
  1. A photorealistic room/lounge background
  2. A seated full-body character image for each person

Falls back to PIL rendering if diffusers / torch is not available.
"""

from __future__ import annotations
import io
import os
from typing import Optional

from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Stable Diffusion helpers
# ─────────────────────────────────────────────────────────────────────────────

_sd_pipe = None   # module-level singleton so we only load the model once


def _get_sd_pipe(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Return a cached StableDiffusionPipeline, loading it on first call."""
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"  [SceneGen] Loading SD model '{model_id}' on {device} …")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        _sd_pipe = pipe
        print(f"  [SceneGen] ✓ SD pipeline ready")
        return _sd_pipe
    except Exception as e:
        print(f"  [SceneGen] Failed to load SD model: {e}")
        return None


def _sd_generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int = 30,
    guidance: float = 7.5,
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> Optional[Image.Image]:
    """Run SD inference; returns PIL image or None on failure."""
    pipe = _get_sd_pipe(model_id)
    if pipe is None:
        return None

    # SD requires dimensions divisible by 8
    w = (width  // 8) * 8
    h = (height // 8) * 8

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            num_inference_steps=steps,
            guidance_scale=guidance,
        )
        return result.images[0].convert("RGB").resize((width, height), Image.LANCZOS)
    except Exception as e:
        print(f"  [SceneGen] SD generation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

ROOM_PROMPT = (
    "photorealistic modern office lounge interior, empty room no people, "
    "large floor-to-ceiling panoramic windows with lush green foliage outside, "
    "monstera plant on left and right, light grey comfortable sofas, "
    "coffee table with two white mugs, warm diffused natural lighting, "
    "soft shadows, ultra high detail, 4K, cinematic still, architectural photography"
)

ROOM_NEGATIVE = (
    "people, person, human, figure, cartoon, anime, painting, sketch, "
    "dark, gloomy, blurry, low quality, watermark, text"
)

PORTRAIT_NEGATIVE = (
    "cartoon, anime, illustration, painting, sketch, drawing, unrealistic, "
    "deformed, disfigured, bad anatomy, extra limbs, watermark, logo, text, "
    "blurry, low quality, low resolution, ugly"
)


def _character_prompt(name: str, role: str, gender: str, appearance: dict) -> str:
    skin_rgb  = appearance.get("skin_rgb",  [220, 185, 155])
    skin_word = (
        "fair"   if skin_rgb[0] > 210 else
        "medium" if skin_rgb[0] > 175 else
        "olive"  if skin_rgb[0] > 140 else "dark"
    )
    hair_style = appearance.get("hair_style", "medium")
    facing     = "slightly to the right" if gender == "female" else "slightly to the left"
    return (
        f"photorealistic upper-body portrait of a {gender} {role}, "
        f"{skin_word} skin, {hair_style} hairstyle, "
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
    model_id: str = "runwayml/stable-diffusion-v1-5",
    cache_dir: str = "faces",
    room_bg_path: Optional[str] = None,
) -> Image.Image:
    """
    Return a photorealistic room background image.

    Priority:
      1. User-supplied room_bg_path
      2. Cached image at cache_dir/room_bg.png
      3. Fresh SD generation
      4. PIL fallback
    """
    # 1. User-supplied image
    if room_bg_path and os.path.isfile(room_bg_path):
        print(f"  [SceneGen] Using user-supplied room background: {room_bg_path}")
        return Image.open(room_bg_path).convert("RGB").resize((width, height), Image.LANCZOS)

    # 2. Cached
    cache_path = os.path.join(cache_dir, "room_bg.png")
    if os.path.isfile(cache_path):
        print(f"  [SceneGen] Using cached room background: {cache_path}")
        img = Image.open(cache_path).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        return img

    # 3. SD generation
    print(f"  [SceneGen] Generating room background with Stable Diffusion …")
    img = _sd_generate(ROOM_PROMPT, ROOM_NEGATIVE, width, height, model_id=model_id)
    if img:
        os.makedirs(cache_dir, exist_ok=True)
        img.save(cache_path)
        print(f"  [SceneGen] ✓ Room background saved → {cache_path}")
        return img

    # 4. PIL fallback
    print("  [SceneGen] SD not available — using PIL room background")
    from video_composer import _make_room_background
    return _make_room_background(width, height)


def generate_character_body(
    name: str,
    role: str,
    gender: str,
    appearance: dict,
    width: int,
    height: int,
    model_id: str = "runwayml/stable-diffusion-v1-5",
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

    prompt = _character_prompt(name, role, gender, appearance)
    print(f"  [SceneGen] Generating body for '{name}' with Stable Diffusion …")
    img = _sd_generate(prompt, PORTRAIT_NEGATIVE, width, height, model_id=model_id)
    if img:
        os.makedirs(cache_dir, exist_ok=True)
        img.save(cpath)
        print(f"  [SceneGen] ✓ Body image saved → {cpath}")
    return img
