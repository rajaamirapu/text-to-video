"""
opensora_generator.py
=====================
Open-Sora (v1.3 codebase, v1.2 public weights) text-to-video integration.

Architecture used
-----------------
  Model     : STDiT3-XL/2  (hpcai-tech/OpenSora-STDiT-v3)
  VAE       : OpenSoraVAE_V1_2  (hpcai-tech/OpenSora-VAE-v1.2)
  Text enc  : T5-XXL  (DeepFloyd/t5-v1_1-xxl)
  Scheduler : rflow (rectified flow, 30 steps)

Two inference modes
-------------------
  CLI mode  (default, recommended)
      Runs Open-Sora's  scripts/inference.py  as a subprocess.
      Uses mmengine config files — exactly the same path as the official docs.
      Works correctly with colossalai single-GPU launch.

  API mode  (--opensora-cli NOT set, for advanced use)
      Loads the pipeline in-process via the opensora registry.
      Avoids subprocess overhead but requires all deps importable in the
      current Python environment.

Public API
----------
  generate_character_video(name, role, gender, facing, output_path, ...)
  generate_background_video(scene_description, output_path, ...)
  extract_still_from_video(video_path, frame_idx)
  make_looping_clip(video_path, target_duration)
  is_opensora_ready()

Setup
-----
  python setup_opensora.py
"""

from __future__ import annotations
import os
import sys
import subprocess
import tempfile
import textwrap
from typing import Optional

from PIL import Image


# ── locate the Open-Sora repo ─────────────────────────────────────────────────

def _find_opensora_root() -> str | None:
    env = os.environ.get("OPENSORA_ROOT")
    if env and os.path.isdir(env):
        return env
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Open-Sora"),
        os.path.expanduser("~/Open-Sora"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "setup.py")):
            return c
    return None


def is_opensora_ready() -> bool:
    """True if the Open-Sora repo is cloned and importable."""
    root = _find_opensora_root()
    if root is None:
        return False
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import opensora  # noqa: F401
        return True
    except ImportError:
        return False


# ── shared negative prompt ────────────────────────────────────────────────────

_NEGATIVE = (
    "blurry, low quality, watermark, text overlay, deformed, cartoon, anime, "
    "unrealistic skin, artifacts, flickering, overexposed, underexposed, nsfw"
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI-based inference  (primary method)
# ─────────────────────────────────────────────────────────────────────────────

def _write_temp_config(
    prompt: str,
    num_frames: int,
    fps: int,
    resolution: str,
    aspect_ratio: str,
    save_dir: str,
    seed: int,
    opensora_dir: str,
) -> str:
    """
    Write a temporary mmengine-style Python config that overrides the base
    sample_hf.py with our runtime parameters.

    Open-Sora's inference.py accepts  --cfg-options  or a separate config
    file; we use the config-file approach for cleanliness.
    """
    # Base config (HF weights, rflow scheduler)
    base_cfg = os.path.join(
        opensora_dir,
        "configs", "opensora-v1-2", "inference", "sample_hf.py",
    )
    if not os.path.isfile(base_cfg):
        raise FileNotFoundError(
            f"Base config not found: {base_cfg}\n"
            "Make sure Open-Sora was cloned correctly via setup_opensora.py"
        )

    # Resolution → (height, width)
    res_map = {
        "144p": (144, 256), "240p": (240, 426), "360p": (360, 640),
        "480p": (480, 854), "720p": (720, 1280), "1080p": (1080, 1920),
    }
    base_h, base_w = res_map.get(resolution, (480, 854))

    # Swap for portrait aspect ratio
    if aspect_ratio in ("9:16", "portrait"):
        h, w = max(base_h, base_w), min(base_h, base_w)
    elif aspect_ratio in ("1:1", "square"):
        h = w = min(base_h, base_w)
    else:
        h, w = base_h, base_w   # 16:9 landscape default

    # Round to multiples of 16 (VAE requirement)
    h = (h // 16) * 16
    w = (w // 16) * 16

    # num_frames must be 4k+1 for Open-Sora (17, 33, 49, 65, 97, 113 …)
    nf = num_frames
    if (nf - 1) % 4 != 0:
        nf = ((nf - 1) // 4) * 4 + 1
        print(f"  [OpenSora] Adjusted num_frames to {nf} (must be 4k+1)")

    config_str = textwrap.dedent(f"""\
        # Auto-generated config for opensora_generator.py
        # Base: configs/opensora-v1-2/inference/sample_hf.py

        _base_ = ["{base_cfg}"]

        num_frames    = {nf}
        fps           = {fps}
        frame_interval = 1
        resolution    = "{resolution}"
        aspect_ratio  = "{aspect_ratio}"
        image_size    = ({h}, {w})

        save_dir = "{save_dir}"
        seed     = {seed}
        batch_size = 1
        dtype    = "bf16"

        # Prompt is passed on the command line via --prompt
        prompt = None
    """)

    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="opensora_cfg_", delete=False
    )
    tmp_cfg.write(config_str)
    tmp_cfg.close()
    return tmp_cfg.name


def _run_cli(
    prompt: str,
    output_path: str,
    num_frames: int    = 49,
    fps: int           = 24,
    resolution: str    = "480p",
    aspect_ratio: str  = "16:9",
    seed: int          = 42,
    opensora_dir: str | None = None,
) -> str:
    """
    Run Open-Sora inference via subprocess using scripts/inference.py.
    This is the recommended path — mirrors the official docs exactly.

    Returns output_path after renaming the generated file.
    """
    root = opensora_dir or _find_opensora_root()
    if root is None:
        raise RuntimeError(
            "Open-Sora repo not found. Run:  python setup_opensora.py"
        )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    tmp_save_dir = tempfile.mkdtemp(prefix="opensora_out_")
    cfg_file     = _write_temp_config(
        prompt, num_frames, fps, resolution, aspect_ratio,
        tmp_save_dir, seed, root,
    )

    script = os.path.join(root, "scripts", "inference.py")

    cmd = [
        sys.executable, script,
        cfg_file,
        "--prompt", prompt,
    ]

    print(f"  [OpenSora] Running inference …")
    print(f"  [OpenSora] Prompt      : {prompt[:80]}")
    print(f"  [OpenSora] Frames/fps  : {num_frames} / {fps}")
    print(f"  [OpenSora] Resolution  : {resolution}  aspect={aspect_ratio}")

    env = os.environ.copy()
    if root not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=root, env=env)

    # Clean up temp config
    try:
        os.unlink(cfg_file)
    except OSError:
        pass

    if result.returncode != 0:
        raise RuntimeError(
            f"Open-Sora inference failed (exit {result.returncode}).\n"
            "Check GPU memory and that all weights were downloaded."
        )

    # Find the newest MP4 in the output dir
    mp4s = sorted(
        [f for f in os.listdir(tmp_save_dir) if f.endswith(".mp4")],
        key=lambda f: os.path.getmtime(os.path.join(tmp_save_dir, f)),
        reverse=True,
    )
    if not mp4s:
        raise RuntimeError(f"No MP4 found in {tmp_save_dir} after inference.")

    generated = os.path.join(tmp_save_dir, mp4s[0])
    import shutil
    shutil.move(generated, output_path)
    try:
        shutil.rmtree(tmp_save_dir, ignore_errors=True)
    except Exception:
        pass

    print(f"  [OpenSora] ✓  Saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# In-process Python API  (alternative to CLI)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_cache = None


def _get_pipeline_api(
    model_id:  str = "hpcai-tech/OpenSora-STDiT-v3",
    vae_id:    str = "hpcai-tech/OpenSora-VAE-v1.2",
    t5_id:     str = "DeepFloyd/t5-v1_1-xxl",
    dtype_str: str = "bf16",
    opensora_dir: str | None = None,
):
    """
    Load and cache the Open-Sora pipeline in-process using the registry API.
    Only used when use_cli=False.
    """
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    root = opensora_dir or _find_opensora_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)

    try:
        import torch
        from opensora.registry    import MODELS, SCHEDULERS, build_module
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import Open-Sora: {e}\n"
            "Run: python setup_opensora.py"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"  [OpenSora] Loading pipeline in-process on {device} ({dtype_str}) …")

    text_encoder = build_module(
        dict(type="t5", from_pretrained=t5_id, model_max_length=300),
        MODELS, device=device,
    )
    vae = build_module(
        dict(type="OpenSoraVAE_V1_2",
             from_pretrained=vae_id,
             micro_frame_size=17, micro_batch_size=4,
             force_huggingface=True),
        MODELS,
    ).to(device, dtype)

    model = build_module(
        dict(type="STDiT3-XL/2",
             from_pretrained=model_id,
             qk_norm=True,
             enable_flash_attn=True,
             enable_layernorm_kernel=True,
             force_huggingface=True),
        MODELS,
    ).to(device, dtype).eval()

    scheduler = build_module(
        dict(type="rflow",
             use_timestep_transform=True,
             num_sampling_steps=30,
             cfg_scale=7.0),
        SCHEDULERS,
    )

    _pipeline_cache = dict(
        model=model, vae=vae,
        text_encoder=text_encoder,
        scheduler=scheduler,
        device=device, dtype=dtype,
    )
    print("  [OpenSora] ✓ Pipeline ready (in-process)")
    return _pipeline_cache


def _run_api(
    prompt: str,
    output_path: str,
    num_frames: int    = 49,
    fps: int           = 24,
    resolution: str    = "480p",
    aspect_ratio: str  = "16:9",
    seed: int          = 42,
    opensora_dir: str | None = None,
) -> str:
    """
    In-process inference via the Open-Sora registry API.
    Less reliable than CLI but avoids subprocess overhead.
    """
    import torch, numpy as np  # noqa: E401

    pipe = _get_pipeline_api(opensora_dir=opensora_dir)
    model, vae, text_encoder, scheduler = (
        pipe["model"], pipe["vae"], pipe["text_encoder"], pipe["scheduler"]
    )
    device, dtype = pipe["device"], pipe["dtype"]

    res_map = {
        "240p": (240, 426), "360p": (360, 640),
        "480p": (480, 854), "720p": (720, 1280),
    }
    base_h, base_w = res_map.get(resolution, (480, 854))
    if aspect_ratio in ("9:16", "portrait"):
        h, w = max(base_h, base_w), min(base_h, base_w)
    else:
        h, w = base_h, base_w
    h = (h // 16) * 16; w = (w // 16) * 16

    nf = num_frames
    if (nf - 1) % 4 != 0:
        nf = ((nf - 1) // 4) * 4 + 1

    generator = torch.Generator(device=device).manual_seed(seed)

    # Encode text
    with torch.no_grad():
        text_tokens   = text_encoder.encode([prompt])
        neg_tokens    = text_encoder.encode([_NEGATIVE])

        # Sample latents via scheduler
        latent_shape  = (1, vae.out_channels, nf, h // 8, w // 8)
        z             = torch.randn(latent_shape, device=device,
                                    dtype=dtype, generator=generator)
        samples       = scheduler.sample(
            model,
            text_encoder  = text_encoder,
            z             = z,
            prompts       = [prompt],
            device        = device,
            additional_args = dict(
                height           = torch.tensor([h], device=device, dtype=dtype),
                width            = torch.tensor([w], device=device, dtype=dtype),
                num_frames       = torch.tensor([nf], device=device, dtype=dtype),
                ar               = torch.tensor([w/h], device=device, dtype=dtype),
                fps              = torch.tensor([fps], device=device, dtype=dtype),
            ),
        )

        # Decode latents → pixel frames
        video_tensor = vae.decode(samples[0].unsqueeze(0))   # [1, C, T, H, W]

    # [1, C, T, H, W] → [T, H, W, C]  uint8
    vt   = video_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy()
    vt   = ((vt + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    _save_frames_to_mp4(vt, output_path, fps)
    print(f"  [OpenSora] ✓  Saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_frames_to_mp4(frames_uint8, output_path: str, fps: int = 24) -> None:
    """Save numpy array [T, H, W, 3] uint8 → MP4 via ffmpeg PNG pipe."""
    import numpy as np

    tmp_dir = tempfile.mkdtemp(prefix="opensora_frames_")
    try:
        for i, frame in enumerate(frames_uint8):
            Image.fromarray(frame).save(os.path.join(tmp_dir, f"f{i:05d}.png"))
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "f%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {r.stderr[:400]}")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_character_video(
    character_name: str,
    role:           str,
    gender:         str,
    facing:         str  = "forward",
    output_path:    str  = "character.mp4",
    num_frames:     int  = 49,
    fps:            int  = 24,
    resolution:     str  = "480p",
    seed:           int  = 42,
    use_cli:        bool = True,
) -> str:
    """
    Generate a short animated portrait video of one character.

    The clip shows the character with a natural neutral expression and
    slight ambient motion.  It is used as:
      • A looping "listening" animation while the other character speaks
      • A still frame (frame 10) fed to Wav2Lip as the face input

    Parameters
    ----------
    character_name : Used in the prompt.
    role           : e.g. "software engineer", "doctor".
    gender         : "male" or "female".
    facing         : "left", "right", or "forward".
    output_path    : Where to write the MP4.
    num_frames     : Frames in the clip (must be 4k+1: 17/33/49/65/97).
    fps            : Frame rate (24 recommended).
    resolution     : "240p", "360p", "480p", or "720p".
    seed           : RNG seed.
    use_cli        : True → subprocess CLI (recommended).

    Returns
    -------
    Absolute path of the generated MP4.
    """
    gw = "man" if gender.lower() in ("male", "m") else "woman"
    facing_phrases = {
        "left":    "slightly facing left, looking left toward conversation partner",
        "right":   "slightly facing right, looking right toward conversation partner",
        "forward": "facing forward, natural eye contact",
    }
    facing_str = facing_phrases.get(facing, "facing forward")

    prompt = (
        f"Photorealistic talking head portrait of a {gw} {role}, {facing_str}, "
        f"calm attentive expression, slight natural head movement, "
        f"professional business casual clothing, soft studio lighting, "
        f"plain neutral dark background, shallow depth of field, 4K quality"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fn = _run_cli if use_cli else _run_api
    return fn(
        prompt       = prompt,
        output_path  = output_path,
        num_frames   = num_frames,
        fps          = fps,
        resolution   = resolution,
        aspect_ratio = "9:16",   # portrait: taller than wide
        seed         = seed,
    )


def generate_background_video(
    scene_description: str = "modern office lounge with warm ambient lighting",
    output_path:       str = "background.mp4",
    num_frames:        int = 97,    # ~4 s at 24 fps
    fps:               int = 24,
    resolution:        str = "720p",
    seed:              int = 0,
    use_cli:           bool = True,
) -> str:
    """
    Generate a looping animated background video.

    The clip is composited behind the character panels, giving the scene
    subtle life (light rays, ambient drift, etc.).

    Parameters
    ----------
    scene_description : Free-text description of the environment.
    output_path       : Where to write the MP4.
    num_frames        : Frames (4k+1).  97 ≈ 4 s at 24 fps.
    fps               : Frame rate.
    resolution        : "480p" or "720p".
    seed              : RNG seed.
    use_cli           : True → subprocess CLI (recommended).

    Returns
    -------
    Absolute path of the generated MP4.
    """
    prompt = (
        f"{scene_description}, "
        f"cinematic background, subtle ambient motion, soft light rays, "
        f"no people, no text, high quality, photorealistic, 4K"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fn = _run_cli if use_cli else _run_api
    return fn(
        prompt       = prompt,
        output_path  = output_path,
        num_frames   = num_frames,
        fps          = fps,
        resolution   = resolution,
        aspect_ratio = "16:9",
        seed         = seed,
    )


def extract_still_from_video(video_path: str, frame_idx: int = 10) -> Image.Image:
    """
    Pull a single frame from a generated video as a PIL Image.
    Used to extract a face still for Wav2Lip input.

    Parameters
    ----------
    video_path : Path to the MP4 file.
    frame_idx  : Which frame to extract (0-indexed; default 10).
    """
    try:
        from moviepy import VideoFileClip  # v2
    except ImportError:
        from moviepy.editor import VideoFileClip  # v1

    clip  = VideoFileClip(video_path)
    t     = frame_idx / max(1, clip.fps)
    frame = clip.get_frame(min(t, clip.duration - 0.01))
    clip.close()
    return Image.fromarray(frame.astype("uint8"))


def make_looping_clip(video_path: str, target_duration: float) -> str:
    """
    Loop a generated clip to fill a target duration.
    The output is saved alongside the source as  <name>_loop.mp4.

    Parameters
    ----------
    video_path      : Source MP4 path.
    target_duration : Desired output length in seconds.
    """
    out_path = video_path.replace(".mp4", "_loop.mp4")

    # ffmpeg stream-loop (cleanest method)
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", video_path,
        "-t", f"{target_duration:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return out_path

    # Fallback: concat copies with moviepy
    try:
        from moviepy import VideoFileClip, concatenate_videoclips  # v2
    except ImportError:
        from moviepy.editor import VideoFileClip, concatenate_videoclips  # v1

    clip   = VideoFileClip(video_path)
    copies = max(1, int(target_duration / clip.duration) + 1)
    looped = concatenate_videoclips([clip] * copies).subclipped(0, target_duration)
    looped.write_videofile(out_path, codec="libx264", audio=False, logger=None)
    clip.close()
    return out_path
