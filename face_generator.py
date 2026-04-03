"""
face_generator.py

Generates photorealistic portrait images using Stable Diffusion (GPU).

Flow
----
1. OllamaClient provides a character description (role, gender, style).
2. We turn that description into a high-quality SD portrait prompt.
3. diffusers runs SD on the GPU and saves a PNG face image.

Default model: SG161222/Realistic_Vision_V5.1_noVAE
  – Excellent skin texture and facial detail, well-suited for portraits.

Alternative (set SD_MODEL_ID in config or env):
  runwayml/stable-diffusion-v1-5  (smaller, ~4 GB)
"""

from __future__ import annotations
import os
import sys

# ── configurable ─────────────────────────────────────────────────────────────
DEFAULT_SD_MODEL   = os.environ.get(
    "SD_MODEL_ID", "SG161222/Realistic_Vision_V5.1_noVAE"
)
IMAGE_SIZE         = (512, 512)   # must be divisible by 8; bump to 768 for SDXL
INFERENCE_STEPS    = 35
GUIDANCE_SCALE     = 7.5

POSITIVE_SUFFIX = (
    "photorealistic portrait, professional headshot, 4k, sharp focus, "
    "natural lighting, high detail, RAW photo"
)
NEGATIVE_PROMPT = (
    "cartoon, anime, painting, illustration, drawing, blurry, deformed, "
    "extra limbs, disfigured, ugly, low quality, watermark, text, logo, "
    "sunglasses, hat, hands, body below shoulders"
)


# ── pipeline cache (load model once per process) ──────────────────────────────
_pipe = None


def _get_pipeline(model_id: str = DEFAULT_SD_MODEL):
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"  [SD] Loading model '{model_id}' …")
    try:
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    except (ImportError, RuntimeError) as e:
        err = str(e)
        hint = (
            "  Run:  python fix_diffusers.py\n"
            "  Then: pip install diffusers>=0.29.0 transformers>=4.41.0 accelerate>=0.30.0"
        )
        if "CLIPImageProcessor" in err:
            sys.exit(
                f"[Error] diffusers/transformers version mismatch:\n  {err}\n{hint}"
            )
        sys.exit(f"[Error] Missing dependency: {err}\n{hint}")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    dtype    = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,        # disable NSFW filter (portraits only)
        requires_safety_checker=False,
    )
    # Faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers optional

    _pipe = pipe
    print(f"  [SD] Model loaded on {device.upper()}")
    return pipe


# ── prompt helpers ────────────────────────────────────────────────────────────

def _build_prompt(
    name: str,
    role: str,
    gender: str,
    appearance: dict | None,
    ollama_description: str | None,
) -> str:
    """Construct the full positive SD prompt for a character."""
    gender_word = {"female": "woman", "male": "man"}.get(gender.lower(), "person")

    # Base subject
    subject = f"close-up portrait of a {gender_word}"

    # Role / profession
    if role and role.lower() not in ("person", "neutral", ""):
        subject += f", {role}"

    # Ollama-provided prose description (e.g. "brown hair, blue eyes, warm smile")
    details = ""
    if ollama_description:
        details = f", {ollama_description}"
    elif appearance:
        # Build from structured JSON if prose not available
        hr, hg, hb = appearance.get("hair_rgb", [80, 50, 20])
        hair_style  = appearance.get("hair_style", "medium-length")
        sr, sg, sb  = appearance.get("skin_rgb", [255, 220, 185])
        skin_desc   = "light" if sr > 200 else "medium" if sr > 160 else "dark"
        details     = f", {hair_style} hair, {skin_desc} skin tone"

    prompt = f"{subject}{details}, {POSITIVE_SUFFIX}"
    return prompt


# ── public API ────────────────────────────────────────────────────────────────

def generate_face_image(
    output_path: str,
    name: str          = "Person",
    role: str          = "professional",
    gender: str        = "female",
    appearance: dict   | None = None,
    ollama_description: str | None = None,
    seed: int          | None = None,
    model_id: str      = DEFAULT_SD_MODEL,
) -> str:
    """
    Generate a photorealistic portrait PNG and save it to *output_path*.

    Returns *output_path* on success.

    Parameters
    ----------
    name               : character name (for logging)
    role               : e.g. "scientist", "journalist"
    gender             : "female" | "male" | "neutral"
    appearance         : structured dict from OllamaClient (optional)
    ollama_description : free-text description from Ollama (optional)
    seed               : for reproducibility; random if None
    model_id           : HuggingFace model repo ID
    """
    import torch

    pipe   = _get_pipeline(model_id)
    prompt = _build_prompt(name, role, gender, appearance, ollama_description)
    neg    = NEGATIVE_PROMPT

    print(f"  [SD] Generating '{name}' ({gender}, {role})")
    print(f"       prompt: {prompt[:80]}…")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    result = pipe(
        prompt,
        negative_prompt=neg,
        width=IMAGE_SIZE[0],
        height=IMAGE_SIZE[1],
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        num_images_per_prompt=1,
    )

    image = result.images[0]
    image.save(output_path)
    print(f"  [SD] Saved → {output_path}")
    return output_path


def generate_all_faces(
    characters: dict,
    appearances: dict,
    output_dir: str = "faces",
    model_id: str   = DEFAULT_SD_MODEL,
) -> dict[str, str]:
    """
    Generate portrait images for all characters.

    Parameters
    ----------
    characters : {name: {role, gender}} dict from dialogue JSON
    appearances: {name: appearance_dict} from OllamaClient
    output_dir : directory to save face PNGs

    Returns
    -------
    {name: image_path} mapping
    """
    os.makedirs(output_dir, exist_ok=True)
    face_paths: dict[str, str] = {}
    seeds = [42, 137, 271, 999]   # deterministic per character index

    for idx, (name, info) in enumerate(characters.items()):
        out_path = os.path.join(output_dir, f"face_{name.lower().replace(' ', '_')}.png")
        if os.path.exists(out_path):
            print(f"  [SD] Using cached face for '{name}' → {out_path}")
            face_paths[name] = out_path
            continue

        app = appearances.get(name)
        generate_face_image(
            output_path=out_path,
            name=name,
            role=info.get("role", "professional"),
            gender=info.get("gender", "neutral"),
            appearance=app,
            seed=seeds[idx % len(seeds)],
            model_id=model_id,
        )
        face_paths[name] = out_path

    return face_paths
