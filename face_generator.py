"""
face_generator.py

Generates character portrait images using Ollama.

Strategy (tried in order)
--------------------------
1. Ollama image generation  — POST /api/generate → response.images[]
   Works if the user has a model with image-output capability installed
   (e.g. a GGUF Stable-Diffusion model loaded via Ollama).

2. Ollama + enhanced PIL    — LLM describes appearance in structured JSON,
   PIL renders a photorealistic-style portrait using those exact colours,
   proportions, and style attributes. Always works regardless of which
   Ollama model is installed.

No diffusers / HuggingFace / Stable Diffusion dependency required.
"""

from __future__ import annotations
import base64
import io
import json
import math
import os
import random
import sys
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFilter


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ── Ollama image-generation attempt
# ─────────────────────────────────────────────────────────────────────────────

def _try_ollama_image_gen(
    prompt: str,
    model: str,
    ollama_url: str,
    width: int,
    height: int,
) -> Optional[Image.Image]:
    """
    Ask Ollama to generate an image.
    Returns a PIL Image if the model supports image output, else None.
    """
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": -1},
        }
        r = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()

        raw_images = data.get("images") or []
        if raw_images:
            img_bytes = base64.b64decode(raw_images[0])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            print(f"  [Ollama] Image generated via model '{model}'")
            return img

    except Exception as e:
        pass   # model doesn't support image output — fall through to PIL

    return None


def _build_image_prompt(name: str, role: str, gender: str, appearance: dict) -> str:
    """Build a detailed portrait prompt from appearance data."""
    sk = appearance.get("skin_rgb", [220, 185, 155])
    hr = appearance.get("hair_rgb", [80, 50, 20])
    ey = appearance.get("eye_rgb", [70, 130, 180])
    style = appearance.get("hair_style", "medium")
    skin_word = "fair" if sk[0] > 210 else "medium" if sk[0] > 170 else "dark"
    return (
        f"photorealistic portrait headshot of {name}, a {gender} {role}, "
        f"{skin_word} skin, {style} hair with rgb({hr[0]},{hr[1]},{hr[2]}) colour, "
        f"rgb({ey[0]},{ey[1]},{ey[2]}) eyes, professional attire, "
        f"natural studio lighting, high detail, 4K, sharp focus, "
        f"facing the camera, neutral background"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ── Enhanced PIL portrait renderer (Ollama-driven colours)
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def _clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))


def _darken(rgb, amt=30):
    return tuple(_clamp(c - amt) for c in rgb)


def _lighten(rgb, amt=30):
    return tuple(_clamp(c + amt) for c in rgb)


def _draw_gradient_ellipse(
    img: Image.Image,
    bbox: tuple,
    color_centre: tuple,
    color_edge: tuple,
    steps: int = 20,
):
    """Fill an ellipse with a radial gradient for skin-depth effect."""
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    for i in range(steps, 0, -1):
        t   = (i - 1) / (steps - 1)
        col = _lerp(color_centre, color_edge, t)
        sx  = rx * i / steps
        sy  = ry * i / steps
        draw.ellipse(
            [cx - sx, cy - sy, cx + sx, cy + sy],
            fill=col,
        )


def _render_portrait(
    width: int,
    height: int,
    appearance: dict,
    name: str,
    position: str = "left",
) -> Image.Image:
    """
    Render a photorealistic-style portrait bust using PIL.
    Colours are driven by Ollama's appearance JSON.
    """
    skin   = tuple(appearance.get("skin_rgb",  [220, 185, 155]))
    hair   = tuple(appearance.get("hair_rgb",  [60,  40,  20]))
    eye    = tuple(appearance.get("eye_rgb",   [70, 130, 180]))
    shirt  = tuple(appearance.get("shirt_rgb", [70, 100, 160]))
    style  = appearance.get("hair_style", "medium")
    gender = appearance.get("gender", "neutral")

    img  = Image.new("RGB", (width, height), (24, 28, 38))
    draw = ImageDraw.Draw(img)

    cx   = width  // 2
    head_r  = int(min(width, height) * 0.22)
    head_cy = int(height * 0.36)
    head_cx = cx

    # ── background gradient ──────────────────────────────────────────────────
    for y in range(height):
        t   = y / height
        col = _lerp((38, 44, 58), (18, 22, 32), t)
        draw.line([(0, y), (width, y)], fill=col)

    # ── body / shirt ─────────────────────────────────────────────────────────
    body_top = head_cy + int(head_r * 1.05)
    sw = int(head_r * 2.0)
    body_pts = [
        (cx - sw,            body_top),
        (cx + sw,            body_top),
        (cx + int(sw * 1.5), height + 10),
        (cx - int(sw * 1.5), height + 10),
    ]
    draw.polygon(body_pts, fill=shirt)
    # shirt highlight
    draw.polygon(body_pts, outline=_lighten(shirt, 20))
    # collar
    draw.line([(cx - 18, body_top), (cx,  body_top + 40)], fill=_lighten(shirt, 40), width=2)
    draw.line([(cx + 18, body_top), (cx,  body_top + 40)], fill=_lighten(shirt, 40), width=2)

    # ── neck ─────────────────────────────────────────────────────────────────
    nw = int(head_r * 0.25)
    ny = head_cy + int(head_r * 0.84)
    _draw_gradient_ellipse(
        img, [cx - nw, ny, cx + nw, ny + int(head_r * 0.5)],
        _lighten(skin, 8), _darken(skin, 15),
    )

    # ── hair (behind head) ───────────────────────────────────────────────────
    _draw_gradient_ellipse(
        img,
        [head_cx - int(head_r * 1.10), head_cy - int(head_r * 1.28),
         head_cx + int(head_r * 1.10), head_cy + int(head_r * 0.32)],
        _lighten(hair, 15),
        _darken(hair, 20),
    )
    if style == "long":
        for side in (-1, 1):
            pts = [
                (head_cx + side * int(head_r * 0.90), head_cy - int(head_r * 0.15)),
                (head_cx + side * int(head_r * 1.22), head_cy + int(head_r * 0.60)),
                (head_cx + side * int(head_r * 1.08), head_cy + int(head_r * 1.30)),
                (head_cx + side * int(head_r * 0.68), head_cy + int(head_r * 1.10)),
            ]
            draw.polygon(pts, fill=hair)

    # ── head (face) ──────────────────────────────────────────────────────────
    _draw_gradient_ellipse(
        img,
        [head_cx - head_r, head_cy - int(head_r * 1.12),
         head_cx + head_r, head_cy + int(head_r * 0.94)],
        _lighten(skin, 18),
        _darken(skin, 12),
        steps=30,
    )

    # ── hairline (front) ─────────────────────────────────────────────────────
    draw.arc(
        [head_cx - int(head_r * 1.08), head_cy - int(head_r * 1.24),
         head_cx + int(head_r * 1.08), head_cy - int(head_r * 0.20)],
        start=202, end=338,
        fill=hair,
        width=int(head_r * 0.30),
    )

    # ── ears ─────────────────────────────────────────────────────────────────
    ear_cy = head_cy - int(head_r * 0.06)
    for side in (-1, 1):
        ex = head_cx + side * int(head_r * 0.88)
        ew, eh = int(head_r * 0.14), int(head_r * 0.20)
        _draw_gradient_ellipse(img, [ex - ew, ear_cy - eh, ex + ew, ear_cy + eh],
                               skin, _darken(skin, 22))
        draw.ellipse(
            [ex - int(ew * 0.45), ear_cy - int(eh * 0.45),
             ex + int(ew * 0.45), ear_cy + int(eh * 0.45)],
            fill=_darken(skin, 30),
        )

    # ── eyebrows ─────────────────────────────────────────────────────────────
    brow_y  = head_cy - int(head_r * 0.30)
    spacing = int(head_r * 0.36)
    brow_w  = int(head_r * 0.25)
    brow_col = _darken(hair, 20)
    for side in (-1, 1):
        bx = head_cx + side * spacing
        draw.arc([bx - brow_w, brow_y - int(head_r * 0.06),
                  bx + brow_w, brow_y + int(head_r * 0.04)],
                 start=208, end=332, fill=brow_col, width=3)

    # ── eyes ─────────────────────────────────────────────────────────────────
    eye_y   = head_cy - int(head_r * 0.14)
    ew, eh  = int(head_r * 0.24), int(head_r * 0.15)
    for side in (-1, 1):
        ex = head_cx + side * spacing

        # sclera
        draw.ellipse([ex - ew, eye_y - eh, ex + ew, eye_y + eh], fill=(252, 252, 250))
        # iris gradient
        ir = int(eh * 0.80)
        _draw_gradient_ellipse(img, [ex - ir, eye_y - ir, ex + ir, eye_y + ir],
                               _lighten(eye, 20), _darken(eye, 10), steps=10)
        # pupil
        pr = int(ir * 0.46)
        draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr], fill=(12, 10, 10))
        # specular
        draw.ellipse([ex + 2, eye_y - pr + 1,
                      ex + int(pr * 0.65) + 2, eye_y],
                     fill=(255, 255, 255))
        # eyelid
        draw.arc([ex - ew, eye_y - eh, ex + ew, eye_y + eh],
                 start=205, end=335, fill=_darken(skin, 50), width=2)
        # lower lash line
        draw.arc([ex - ew, eye_y - eh, ex + ew, eye_y + eh],
                 start=20, end=160, fill=_darken(skin, 35), width=1)

    # ── nose ─────────────────────────────────────────────────────────────────
    nose_cy = head_cy + int(head_r * 0.12)
    nw2     = int(head_r * 0.09)
    shadow  = _darken(skin, 28)
    for side in (-1, 1):
        nx = head_cx + side * int(nw2 * 0.88)
        draw.ellipse([nx - int(head_r * 0.042), nose_cy - int(head_r * 0.030),
                      nx + int(head_r * 0.042), nose_cy + int(head_r * 0.042)],
                     fill=shadow)
    # bridge shading
    draw.line(
        [(head_cx, head_cy - int(head_r * 0.10)), (head_cx, nose_cy + int(head_r * 0.04))],
        fill=_darken(skin, 18), width=1,
    )

    # ── mouth ─────────────────────────────────────────────────────────────────
    mouth_cy = head_cy + int(head_r * 0.42)
    mw       = int(head_r * 0.34)
    lip_col  = (
        _clamp(skin[0] + 20),
        _clamp(skin[1] - 45),
        _clamp(skin[2] - 45),
    )
    # lip shadow
    draw.ellipse([head_cx - int(mw * 0.6), mouth_cy - 3,
                  head_cx + int(mw * 0.6), mouth_cy + 3],
                 fill=_darken(skin, 22))
    # smile arc
    draw.arc([head_cx - mw, mouth_cy - int(head_r * 0.06),
              head_cx + mw, mouth_cy + int(head_r * 0.06)],
             start=8, end=172, fill=lip_col, width=3)
    # lower lip fullness
    draw.arc([head_cx - int(mw * 0.78), mouth_cy,
              head_cx + int(mw * 0.78), mouth_cy + int(head_r * 0.08)],
             start=5, end=175, fill=_lighten(lip_col, 12), width=2)

    # ── subtle skin texture (fine noise) ─────────────────────────────────────
    # Add a very slight noise layer over the face area for texture
    face_mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(face_mask)
    mask_draw.ellipse(
        [head_cx - head_r, head_cy - int(head_r * 1.12),
         head_cx + head_r, head_cy + int(head_r * 0.94)],
        fill=255,
    )
    rng = random.Random(42)
    noise = Image.new("RGB", (width, height), (0, 0, 0))
    noise_pixels = noise.load()
    for py in range(head_cy - int(head_r * 1.12), head_cy + int(head_r * 0.94)):
        for px in range(head_cx - head_r, head_cx + head_r):
            if 0 <= px < width and 0 <= py < height:
                v = rng.randint(-6, 6)
                noise_pixels[px, py] = (v, v, v)
    img = Image.composite(
        Image.blend(img, Image.eval(img, lambda x: _clamp(x + rng.randint(-4, 4))), 0.3),
        img,
        face_mask,
    )

    # ── name label ────────────────────────────────────────────────────────────
    draw2 = ImageDraw.Draw(img)
    lw, lh = 160, 26
    lx, ly = cx - lw // 2, height - 45
    draw2.rounded_rectangle([lx, ly, lx + lw, ly + lh], radius=7,
                             fill=(40, 50, 70, 200))
    draw2.text((cx, ly + lh // 2), name, fill=(210, 220, 240), anchor="mm")

    # Soft vignette
    img = img.filter(ImageFilter.GaussianBlur(0))

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ── Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_face_image(
    output_path: str,
    name: str          = "Person",
    role: str          = "professional",
    gender: str        = "female",
    appearance: dict   | None = None,
    ollama_url: str    = "http://localhost:11434",
    ollama_model: str  | None = None,
    width: int         = 512,
    height: int        = 512,
    position: str      = "left",
) -> str:
    """
    Generate and save a portrait image.
    Tries Ollama image generation first; falls back to enhanced PIL rendering.

    Returns the saved output_path.
    """
    if appearance is None:
        appearance = {}

    # ── 1. Try Ollama image generation ───────────────────────────────────────
    if ollama_model:
        prompt = _build_image_prompt(name, role, gender, appearance)
        img = _try_ollama_image_gen(prompt, ollama_model, ollama_url, width, height)
        if img:
            img.save(output_path)
            print(f"  [FaceGen] '{name}' saved via Ollama image gen → {output_path}")
            return output_path

    # ── 2. Enhanced PIL rendering (always works) ──────────────────────────────
    print(f"  [FaceGen] Rendering '{name}' portrait with PIL (Ollama-driven colours)")
    img = _render_portrait(width, height, appearance, name, position)
    img.save(output_path)
    print(f"  [FaceGen] Saved → {output_path}")
    return output_path


def generate_all_faces(
    characters: dict,
    appearances: dict,
    output_dir: str    = "faces",
    ollama_url: str    = "http://localhost:11434",
    ollama_model: str  | None = None,
    width: int         = 512,
    height: int        = 512,
    regen: bool        = False,
) -> dict[str, str]:
    """
    Generate portrait images for all characters.

    Parameters
    ----------
    characters  : {name: {role, gender}} from dialogue JSON
    appearances : {name: appearance_dict} from OllamaClient
    output_dir  : directory to save PNGs
    ollama_url  : Ollama server URL
    ollama_model: model to try for image generation (None = skip, use PIL)
    regen       : force regeneration even if file already exists

    Returns {name: image_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    face_paths: dict[str, str] = {}
    positions = ["left", "right"]

    for idx, (name, info) in enumerate(characters.items()):
        safe = name.lower().replace(" ", "_")
        out  = os.path.join(output_dir, f"face_{safe}.png")

        if os.path.isfile(out) and not regen:
            print(f"  [FaceGen] Using cached portrait for '{name}' → {out}")
            face_paths[name] = out
            continue

        generate_face_image(
            output_path  = out,
            name         = name,
            role         = info.get("role", "professional"),
            gender       = info.get("gender", "neutral"),
            appearance   = appearances.get(name, {}),
            ollama_url   = ollama_url,
            ollama_model = ollama_model,
            width        = width,
            height       = height,
            position     = positions[idx % len(positions)],
        )
        face_paths[name] = out

    return face_paths


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ── Ollama model detection helper
# ─────────────────────────────────────────────────────────────────────────────

def find_image_gen_model(ollama_url: str = "http://localhost:11434") -> str | None:
    """
    Probe available Ollama models to find one that can generate images.
    Returns the first model name that returns images, or None.
    """
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None

    # Keywords that hint a model might support image generation
    image_hints = ["sd", "stable", "diffusion", "flux", "imagen", "dall", "draw"]

    candidates = [
        m for m in models
        if any(h in m.lower() for h in image_hints)
    ] + models   # try all if none match hints

    test_prompt = "a simple white circle on black background"
    for model in candidates:
        img = _try_ollama_image_gen(test_prompt, model, ollama_url, 64, 64)
        if img:
            print(f"  [FaceGen] Found image-generation model: '{model}'")
            return model

    return None
