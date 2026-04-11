#!/usr/bin/env python3
"""
generate_faces.py

Generate face portraits and room background for all characters using
Stable Diffusion (via the diffusers library).

Usage:
    python generate_faces.py                          # uses default SD model
    python generate_faces.py --model "runwayml/stable-diffusion-v1-5"
    python generate_faces.py --script my_dialogue.json
    python generate_faces.py --regen                  # force re-generate
"""

import argparse
import json
import os
import sys

from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Stable Diffusion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_sd_pipeline(model_id: str):
    """Load a StableDiffusionPipeline (float16 on CUDA, float32 on CPU)."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError:
        sys.exit(
            "[Error] diffusers / torch not installed.\n"
            "  Run:  pip install diffusers transformers accelerate torch"
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"  → Loading SD model '{model_id}' on {device} ({dtype}) …")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,          # disable for portraits
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """Run SD inference and return a PIL image."""
    # SD requires dimensions divisible by 8
    width  = (width  // 8) * 8
    height = (height // 8) * 8

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    return result.images[0].convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE = (
    "cartoon, anime, illustration, painting, sketch, drawing, unrealistic, "
    "deformed, disfigured, bad anatomy, extra limbs, watermark, logo, text, "
    "blurry, low quality, low resolution"
)

ROOM_PROMPT = (
    "photorealistic modern office lounge interior, empty room no people, "
    "large floor-to-ceiling panoramic windows with lush green foliage outside, "
    "monstera plant on left and right, light grey comfortable sofas, "
    "coffee table with two white mugs, warm diffused natural lighting, "
    "soft shadows, ultra high detail, 4K, cinematic still, architectural photography"
)

ROOM_NEGATIVE = (
    "people, person, human, figure, cartoon, anime, painting, sketch, "
    "dark, gloomy, blurry, low quality"
)


def portrait_prompt(name: str, role: str, gender: str, panel_idx: int = 0) -> str:
    """
    Build a portrait prompt.

    panel_idx : 0 = left panel  → character should face RIGHT (toward partner)
                1 = right panel → character should face LEFT  (toward partner)
    """
    gender_word = "man" if gender == "male" else "woman"
    # Direction is driven by panel position so they always face each other,
    # regardless of gender.
    if panel_idx == 0:
        facing = "slightly right, looking toward the right side of frame"
    else:
        facing = "slightly left, looking toward the left side of frame"
    return (
        f"photorealistic portrait headshot of a {gender_word}, {role}, "
        f"facing {facing}, natural conversational expression, "
        f"professional business casual attire, "
        f"soft studio lighting, shallow depth of field, sharp focus on face, "
        f"plain neutral dark background, 4K DSLR quality, ultra realistic skin texture"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate character face portraits using Stable Diffusion."
    )
    parser.add_argument(
        "--model", default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID or local path (default: runwayml/stable-diffusion-v1-5)"
    )
    parser.add_argument(
        "--script", default="example_dialogue.json",
        help="Dialogue JSON (to read character list)"
    )
    parser.add_argument("--faces-dir", default="faces",  help="Output directory for face images")
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps",  type=int, default=30,  help="SD inference steps")
    parser.add_argument("--cfg",    type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--regen",  action="store_true",  help="Regenerate even if cached")
    parser.add_argument("--room-bg", action="store_true", help="Also generate room background")
    args = parser.parse_args()

    # ── Load characters ───────────────────────────────────────────────────────
    with open(args.script, encoding="utf-8") as f:
        data = json.load(f)
    characters = data.get("characters", {})

    os.makedirs(args.faces_dir, exist_ok=True)

    print(f"\nStable Diffusion model : {args.model}")
    print(f"Output dir             : {args.faces_dir}/")
    print(f"Characters             : {list(characters.keys())}")
    if args.room_bg:
        print("Room background        : will be generated")
    print()

    # ── Load SD pipeline ──────────────────────────────────────────────────────
    pipe = _load_sd_pipeline(args.model)

    # ── Generate room background (optional) ───────────────────────────────────
    if args.room_bg:
        room_path = os.path.join(args.faces_dir, "room_bg.png")
        if os.path.isfile(room_path) and not args.regen:
            print(f"[Room BG] Skipping — already exists: {room_path}")
            print("          (use --regen to regenerate)")
        else:
            print("[Room BG] Generating room background …")
            room_w = max(args.width  * 2, 1024)
            room_h = max(args.height * 2, 576)
            img = generate_image(
                pipe, ROOM_PROMPT, ROOM_NEGATIVE,
                room_w, room_h, args.steps, args.cfg
            )
            img.save(room_path)
            print(f"[Room BG] ✓ Saved → {room_path}  ({img.size[0]}×{img.size[1]})\n")

    # ── Generate character faces ───────────────────────────────────────────────
    char_list = list(characters.items())
    for panel_idx, (name, info) in enumerate(char_list):
        role   = info.get("role",   "person")
        gender = info.get("gender", "neutral")
        safe   = name.lower().replace(" ", "_")
        out    = os.path.join(args.faces_dir, f"face_{safe}.png")

        if os.path.isfile(out) and not args.regen:
            print(f"[{name}] Skipping — already exists: {out}")
            print(f"         (use --regen to regenerate)")
            continue

        print(f"[{name}] Generating {gender} {role} portrait (panel {panel_idx}) …")
        prompt = portrait_prompt(name, role, gender, panel_idx=panel_idx)
        print(f"  → Prompt: {prompt[:90]}…")

        try:
            img = generate_image(
                pipe, prompt, NEGATIVE,
                args.width, args.height, args.steps, args.cfg
            )
        except Exception as e:
            print(f"[{name}] ⚠ SD failed ({e}) — falling back to PIL placeholder.")
            img = None

        if img is None:
            from main import _make_placeholder_face
            img = _make_placeholder_face(name, gender, panel_idx,
                                         args.width, args.height)
            print(f"[{name}] ⚠ PIL placeholder generated.")

        img.save(out)
        print(f"[{name}] ✓ Saved → {out}  ({img.size[0]}×{img.size[1]})\n")

    print("Done! All face images generated.")
    print(f"\nNext step — run the full pipeline:")
    print(f"  python main.py example_dialogue.json --output my_video.mp4")


if __name__ == "__main__":
    main()
