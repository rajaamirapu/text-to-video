"""
face_generator.py

Generates realistic character portrait images.

Strategy (tried in order)
--------------------------
1. Ollama image generation  — POST /api/generate → response.images[]
   Works if the user has a model with image-output capability installed
   (e.g. a FLUX or SD GGUF model loaded via Ollama).

2. Pollinations.ai (free, no API key) — sends a detailed prompt to
   https://image.pollinations.ai and receives a photorealistic JPEG.
   Requires an internet connection. Powered by FLUX.

3. Enhanced PIL fallback — always works offline, produces a stylised
   portrait driven by Ollama appearance colours.

No diffusers / HuggingFace / Stable Diffusion dependency required.
"""

from __future__ import annotations
import base64
import io
import os
import random
import urllib.parse
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFilter


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 ── Ollama image-generation
# ─────────────────────────────────────────────────────────────────────────────

def _try_ollama_image_gen(
    prompt: str,
    model: str,
    ollama_url: str,
    width: int,
    height: int,
) -> Optional[Image.Image]:
    """Ask Ollama to generate an image. Returns PIL Image or None."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": -1},
        }
        r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        raw_images = data.get("images") or []
        if raw_images:
            img_bytes = base64.b64decode(raw_images[0])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            print(f"  [FaceGen] Ollama image generated via model '{model}'")
            return img
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 ── Pollinations.ai (free, photorealistic FLUX images)
# ─────────────────────────────────────────────────────────────────────────────

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"


def _try_pollinations(
    prompt: str,
    width: int,
    height: int,
    seed: int = 42,
    timeout: int = 60,
) -> Optional[Image.Image]:
    """
    Fetch a photorealistic image from Pollinations.ai (free, no API key).
    Returns a PIL Image on success, None on any error.
    """
    encoded = urllib.parse.quote(prompt)
    url = (
        f"{POLLINATIONS_BASE}/{encoded}"
        f"?width={width}&height={height}&seed={seed}"
        f"&model=flux&nologo=true&enhance=true"
    )
    try:
        print(f"  [FaceGen] Requesting photorealistic portrait from Pollinations.ai …")
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "image" not in content_type:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if img.size[0] < 64:          # suspiciously small → probably an error page
            return None
        img = img.resize((width, height), Image.LANCZOS)
        print(f"  [FaceGen] ✓ Photorealistic portrait received ({img.size[0]}×{img.size[1]})")
        return img
    except Exception as e:
        print(f"  [FaceGen] Pollinations.ai unavailable ({e}) — using PIL portrait")
        return None


def _build_portrait_prompt(name: str, role: str, gender: str, appearance: dict, seed: int) -> str:
    """Build a rich photorealistic prompt from appearance data."""
    sk    = appearance.get("skin_rgb",  [220, 185, 155])
    hr    = appearance.get("hair_rgb",  [80,  50,  20])
    ey    = appearance.get("eye_rgb",   [70, 130, 180])
    style = appearance.get("hair_style", "medium length")

    skin_word = (
        "fair, light" if sk[0] > 210 else
        "medium, warm" if sk[0] > 175 else
        "olive, tan" if sk[0] > 140 else
        "deep brown, dark"
    )
    hair_hex  = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in hr])
    eye_color = _rgb_to_color_word(ey)

    return (
        f"professional headshot portrait photograph of a real {gender} {role} named {name}, "
        f"{skin_word} skin tone, {style} hair in shade {hair_hex}, {eye_color} eyes, "
        f"neutral expression, business casual attire, "
        f"soft studio lighting, shallow depth of field, "
        f"sharp focus on face, photorealistic, 8K, DSLR quality, "
        f"plain light grey background, facing camera directly, "
        f"no makeup excess, natural appearance, seed {seed}"
    )


def _rgb_to_color_word(rgb) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    if r > 150 and g > 150 and b > 150:
        return "light grey"
    if b > r and b > g:
        return "blue" if b > 150 else "dark blue"
    if g > r and g > b:
        return "green"
    if r > g and r > b:
        return "brown" if r < 160 else "hazel"
    return "dark brown"


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 ── PIL fallback portrait renderer
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))

def _clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(v)))

def _darken(rgb, amt=30):
    return tuple(_clamp(c - amt) for c in rgb)

def _lighten(rgb, amt=30):
    return tuple(_clamp(c + amt) for c in rgb)

def _draw_gradient_ellipse(img, bbox, color_centre, color_edge, steps=20):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    for i in range(steps, 0, -1):
        t   = (i - 1) / (steps - 1)
        col = _lerp(color_centre, color_edge, t)
        sx  = rx * i / steps
        sy  = ry * i / steps
        draw.ellipse([cx - sx, cy - sy, cx + sx, cy + sy], fill=col)


def _render_portrait(width, height, appearance, name, position="left"):
    """Stylised PIL fallback — used only when all AI options are unavailable."""
    skin  = tuple(appearance.get("skin_rgb",  [220, 185, 155]))
    hair  = tuple(appearance.get("hair_rgb",  [60,  40,  20]))
    eye   = tuple(appearance.get("eye_rgb",   [70, 130, 180]))
    shirt = tuple(appearance.get("shirt_rgb", [70, 100, 160]))
    style = appearance.get("hair_style", "medium")

    img  = Image.new("RGB", (width, height), (24, 28, 38))
    draw = ImageDraw.Draw(img)
    cx   = width  // 2
    head_r  = int(min(width, height) * 0.22)
    head_cy = int(height * 0.36)
    head_cx = cx

    # background gradient
    for y in range(height):
        t   = y / height
        col = _lerp((38, 44, 58), (18, 22, 32), t)
        draw.line([(0, y), (width, y)], fill=col)

    # body / shirt
    body_top = head_cy + int(head_r * 1.05)
    sw = int(head_r * 2.0)
    body_pts = [
        (cx - sw,            body_top),
        (cx + sw,            body_top),
        (cx + int(sw * 1.5), height + 10),
        (cx - int(sw * 1.5), height + 10),
    ]
    draw.polygon(body_pts, fill=shirt)
    draw.polygon(body_pts, outline=_lighten(shirt, 20))
    draw.line([(cx - 18, body_top), (cx, body_top + 40)], fill=_lighten(shirt, 40), width=2)
    draw.line([(cx + 18, body_top), (cx, body_top + 40)], fill=_lighten(shirt, 40), width=2)

    # neck
    nw = int(head_r * 0.25)
    ny = head_cy + int(head_r * 0.84)
    _draw_gradient_ellipse(img, [cx - nw, ny, cx + nw, ny + int(head_r * 0.5)],
                           _lighten(skin, 8), _darken(skin, 15))

    # hair (behind)
    _draw_gradient_ellipse(
        img,
        [head_cx - int(head_r * 1.10), head_cy - int(head_r * 1.28),
         head_cx + int(head_r * 1.10), head_cy + int(head_r * 0.32)],
        _lighten(hair, 15), _darken(hair, 20),
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

    # face
    _draw_gradient_ellipse(
        img,
        [head_cx - head_r, head_cy - int(head_r * 1.12),
         head_cx + head_r, head_cy + int(head_r * 0.94)],
        _lighten(skin, 18), _darken(skin, 12), steps=30,
    )

    # hairline
    draw.arc(
        [head_cx - int(head_r * 1.08), head_cy - int(head_r * 1.24),
         head_cx + int(head_r * 1.08), head_cy - int(head_r * 0.20)],
        start=202, end=338, fill=hair, width=int(head_r * 0.30),
    )

    # ears
    ear_cy = head_cy - int(head_r * 0.06)
    for side in (-1, 1):
        ex = head_cx + side * int(head_r * 0.88)
        ew, eh = int(head_r * 0.14), int(head_r * 0.20)
        _draw_gradient_ellipse(img, [ex - ew, ear_cy - eh, ex + ew, ear_cy + eh],
                               skin, _darken(skin, 22))
        draw.ellipse([ex - int(ew * 0.45), ear_cy - int(eh * 0.45),
                      ex + int(ew * 0.45), ear_cy + int(eh * 0.45)],
                     fill=_darken(skin, 30))

    # eyebrows
    brow_y  = head_cy - int(head_r * 0.30)
    spacing = int(head_r * 0.36)
    brow_w  = int(head_r * 0.25)
    brow_col = _darken(hair, 20)
    for side in (-1, 1):
        bx = head_cx + side * spacing
        draw.arc([bx - brow_w, brow_y - int(head_r * 0.06),
                  bx + brow_w, brow_y + int(head_r * 0.04)],
                 start=208, end=332, fill=brow_col, width=3)

    # eyes
    eye_y  = head_cy - int(head_r * 0.14)
    ew, eh = int(head_r * 0.24), int(head_r * 0.15)
    for side in (-1, 1):
        ex = head_cx + side * spacing
        draw.ellipse([ex - ew, eye_y - eh, ex + ew, eye_y + eh], fill=(252, 252, 250))
        ir = int(eh * 0.80)
        _draw_gradient_ellipse(img, [ex - ir, eye_y - ir, ex + ir, eye_y + ir],
                               _lighten(eye, 20), _darken(eye, 10), steps=10)
        pr = int(ir * 0.46)
        draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr], fill=(12, 10, 10))
        draw.ellipse([ex + 2, eye_y - pr + 1, ex + int(pr * 0.65) + 2, eye_y],
                     fill=(255, 255, 255))
        draw.arc([ex - ew, eye_y - eh, ex + ew, eye_y + eh],
                 start=205, end=335, fill=_darken(skin, 50), width=2)
        draw.arc([ex - ew, eye_y - eh, ex + ew, eye_y + eh],
                 start=20, end=160, fill=_darken(skin, 35), width=1)

    # nose
    nose_cy = head_cy + int(head_r * 0.12)
    shadow  = _darken(skin, 28)
    for side in (-1, 1):
        nx = head_cx + side * int(head_r * 0.08)
        draw.ellipse([nx - int(head_r * 0.042), nose_cy - int(head_r * 0.030),
                      nx + int(head_r * 0.042), nose_cy + int(head_r * 0.042)],
                     fill=shadow)
    draw.line([(head_cx, head_cy - int(head_r * 0.10)),
               (head_cx, nose_cy + int(head_r * 0.04))],
              fill=_darken(skin, 18), width=1)

    # mouth
    mouth_cy = head_cy + int(head_r * 0.42)
    mw       = int(head_r * 0.34)
    lip_col  = (_clamp(skin[0] + 20), _clamp(skin[1] - 45), _clamp(skin[2] - 45))
    draw.ellipse([head_cx - int(mw * 0.6), mouth_cy - 3,
                  head_cx + int(mw * 0.6), mouth_cy + 3],
                 fill=_darken(skin, 22))
    draw.arc([head_cx - mw, mouth_cy - int(head_r * 0.06),
              head_cx + mw, mouth_cy + int(head_r * 0.06)],
             start=8, end=172, fill=lip_col, width=3)
    draw.arc([head_cx - int(mw * 0.78), mouth_cy,
              head_cx + int(mw * 0.78), mouth_cy + int(head_r * 0.08)],
             start=5, end=175, fill=_lighten(lip_col, 12), width=2)

    # name label
    draw2 = ImageDraw.Draw(img)
    lw, lh = 160, 26
    lx, ly = cx - lw // 2, height - 45
    draw2.rounded_rectangle([lx, ly, lx + lw, ly + lh], radius=7, fill=(40, 50, 70))
    draw2.text((cx, ly + lh // 2), name, fill=(210, 220, 240), anchor="mm")

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 ── Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_face_image(
    output_path: str,
    name: str         = "Person",
    role: str         = "professional",
    gender: str       = "female",
    appearance: dict  | None = None,
    ollama_url: str   = "http://localhost:11434",
    ollama_model: str | None = None,
    width: int        = 512,
    height: int       = 512,
    position: str     = "left",
    seed: int         = 42,
    use_pollinations: bool = True,
) -> str:
    """
    Generate and save a realistic portrait image.

    Tries (in order):
      1. Ollama image-generation model (if ollama_model provided)
      2. Pollinations.ai free FLUX API  (if use_pollinations=True and internet available)
      3. Enhanced PIL portrait          (always works, offline)

    Returns the saved output_path.
    """
    if appearance is None:
        appearance = {}

    prompt = _build_portrait_prompt(name, role, gender, appearance, seed)

    # ── 1. Ollama image gen ───────────────────────────────────────────────────
    if ollama_model:
        img = _try_ollama_image_gen(prompt, ollama_model, ollama_url, width, height)
        if img:
            img.save(output_path)
            print(f"  [FaceGen] '{name}' → {output_path}  (Ollama)")
            return output_path

    # ── 2. Pollinations.ai (photorealistic FLUX, free, no API key) ────────────
    if use_pollinations:
        img = _try_pollinations(prompt, width, height, seed=seed)
        if img:
            img.save(output_path)
            print(f"  [FaceGen] '{name}' → {output_path}  (Pollinations.ai)")
            return output_path

    # ── 3. PIL fallback ───────────────────────────────────────────────────────
    print(f"  [FaceGen] Rendering '{name}' with PIL portrait (offline fallback)")
    img = _render_portrait(width, height, appearance, name, position)
    img.save(output_path)
    print(f"  [FaceGen] '{name}' → {output_path}  (PIL)")
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
    use_pollinations: bool = True,
) -> dict[str, str]:
    """
    Generate portrait images for all characters.

    Returns {name: image_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    face_paths: dict[str, str] = {}
    positions  = ["left", "right"]
    base_seed  = 1337   # deterministic but distinct seed per character

    for idx, (name, info) in enumerate(characters.items()):
        safe = name.lower().replace(" ", "_")
        out  = os.path.join(output_dir, f"face_{safe}.png")

        if os.path.isfile(out) and not regen:
            print(f"  [FaceGen] Cached portrait → '{name}'  ({out})")
            face_paths[name] = out
            continue

        generate_face_image(
            output_path      = out,
            name             = name,
            role             = info.get("role", "professional"),
            gender           = info.get("gender", "neutral"),
            appearance       = appearances.get(name, {}),
            ollama_url       = ollama_url,
            ollama_model     = ollama_model,
            width            = width,
            height           = height,
            position         = positions[idx % len(positions)],
            seed             = base_seed + idx * 17,
            use_pollinations = use_pollinations,
        )
        face_paths[name] = out

    return face_paths


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 ── Ollama model detection helper
# ─────────────────────────────────────────────────────────────────────────────

def find_image_gen_model(ollama_url: str = "http://localhost:11434") -> str | None:
    """
    Probe available Ollama models to find one that can generate images.
    Returns the first model name that returns image data, or None.
    """
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None

    image_hints = ["sd", "stable", "diffusion", "flux", "imagen", "dall", "draw"]
    candidates  = [m for m in models if any(h in m.lower() for h in image_hints)]
    candidates  = candidates + [m for m in models if m not in candidates]

    for model in candidates:
        img = _try_ollama_image_gen(
            "a simple white circle on black background", model, ollama_url, 64, 64
        )
        if img:
            print(f"  [FaceGen] Ollama image model found: '{model}'")
            return model

    return None
