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

# ── NCCL compatibility shim ───────────────────────────────────────────────────
# Must be set BEFORE torch is imported anywhere in this process.
# Prevents "ncclCommWindowDeregister: undefined symbol" on hosts where
# the system NCCL (2.18-) is older than what PyTorch (2.4+) was compiled against.
# These flags disable peer-to-peer and InfiniBand NCCL transports; shared-memory
# and socket transports remain active — fine for single-GPU inference.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE",  "1")

from PIL import Image


# ── Flash Attention compatibility check ───────────────────────────────────────

def _flash_attn_ok() -> bool:
    """
    Return True only if flash-attn is importable AND its C++ kernel schema
    matches the running PyTorch version.

    Flash Attention must be compiled for the *exact* torch+CUDA combo.
    Common mismatch:  flash-attn 2.5.x compiled for torch 2.0/2.1, but
    torch 2.2+ changed the  aten::_flash_attention_forward  return schema
    from 4 tensors to 5 tensors.  Importing the wheel succeeds; using it
    raises  "does not have a compatible aten::_flash_attention_forward schema".
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
            import os
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"]  = "1"
            import torch
            import flash_attn
            # Trigger the schema validation
            from flash_attn import flash_attn_func
            q = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.float16)
            flash_attn_func(q, q, q)
            print("ok")
        """)],
        capture_output=True, text=True,
    )
    ok = result.returncode == 0 and "ok" in result.stdout
    if not ok and result.stderr:
        needle = "compatible aten::_flash_attention_forward"
        if needle in result.stderr or "schema" in result.stderr.lower():
            print(
                "  [OpenSora] flash-attn schema mismatch detected — "
                "will use xformers / standard attention instead."
            )
    return ok


# Module-level cached result (computed once per process)
_FLASH_ATTN_AVAILABLE: bool | None = None


def _use_flash_attn() -> bool:
    """Cached wrapper around _flash_attn_ok()."""
    global _FLASH_ATTN_AVAILABLE
    if _FLASH_ATTN_AVAILABLE is None:
        try:
            _FLASH_ATTN_AVAILABLE = _flash_attn_ok()
        except Exception:
            _FLASH_ATTN_AVAILABLE = False
    return _FLASH_ATTN_AVAILABLE


def _xformers_ok() -> bool:
    """
    Return True only if xformers can be imported cleanly with the running torch.
    xformers built for a different (torch, CUDA, Python) triple will raise an
    ImportError from xformers.ops.fmha.torch_attention_compat.ensure_pt_flash_ok
    which propagates through diffusers and crashes the whole inference process.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
            import os, warnings
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"]  = "1"
            warnings.filterwarnings("ignore")
            import xformers
            import xformers.ops
            print("ok")
        """)],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and "ok" in result.stdout


_XFORMERS_OK: bool | None = None


def _xformers_available() -> bool:
    """Cached wrapper around _xformers_ok()."""
    global _XFORMERS_OK
    if _XFORMERS_OK is None:
        try:
            _XFORMERS_OK = _xformers_ok()
        except Exception:
            _XFORMERS_OK = False
    return _XFORMERS_OK


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

    # Check flash-attn once; disable it in the config if it's broken
    flash_ok = _use_flash_attn()
    attn_override = "" if flash_ok else textwrap.dedent("""\

        # flash-attn schema mismatch — disable to fall back to xformers/standard attn
        model = dict(
            enable_flash_attn=False,
            enable_layernorm_kernel=False,
        )
    """)
    if not flash_ok:
        print("  [OpenSora] flash-attn disabled in config (schema incompatible with torch).")

    # Choose dtype: bf16 needs Ampere+ (A100, 3090, 4090 …).
    # T4 / V100 / older GPUs only support fp16 natively; bf16 silently falls
    # back to fp32 on those cards, doubling memory use.  We detect support.
    dtype_check = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['NCCL_P2P_DISABLE']='1'; "
         "import torch; "
         "d=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; "
         "print('bf16' if d and d.major>=8 else 'fp16')"],
        capture_output=True, text=True,
    )
    dtype = dtype_check.stdout.strip() if dtype_check.returncode == 0 else "fp16"
    if dtype == "fp16":
        print("  [OpenSora] GPU does not support bf16 natively — using fp16.")

    # Memory budget: for ≤16 GB cards lower sampling steps and VAE micro-batch.
    vram_check = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['NCCL_P2P_DISABLE']='1'; "
         "import torch; "
         "gb=torch.cuda.get_device_properties(0).total_memory/1024**3 "
         "if torch.cuda.is_available() else 0; print(f'{gb:.1f}')"],
        capture_output=True, text=True,
    )
    try:
        vram_gb = float(vram_check.stdout.strip())
    except ValueError:
        vram_gb = 24.0   # assume large if detection fails

    # Scale aggressiveness based on available VRAM
    if vram_gb < 12:
        num_steps, vae_micro = 15, 1
    elif vram_gb < 16:
        num_steps, vae_micro = 20, 1
    elif vram_gb < 24:
        num_steps, vae_micro = 25, 2
    else:
        num_steps, vae_micro = 30, 4   # default; full quality

    print(f"  [OpenSora] VRAM {vram_gb:.1f} GiB → "
          f"steps={num_steps}, vae_micro_batch={vae_micro}, dtype={dtype}")

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

        save_dir   = "{save_dir}"
        seed       = {seed}
        batch_size = 1
        dtype      = "{dtype}"

        # Prompt is passed on the command line via --prompt
        prompt = None

        # ── Memory optimisations ──────────────────────────────────────────────
        # Fewer diffusion steps: slightly lower quality but fits in VRAM.
        # micro_batch_size: VAE decodes this many frames at once; 1 = minimum memory.
        scheduler = dict(
            num_sampling_steps = {num_steps},
            cfg_scale          = 7.0,
        )
        vae = dict(
            micro_frame_size  = 17,
            micro_batch_size  = {vae_micro},
        )
    """) + attn_override

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

    # ── Pre-import wrapper ────────────────────────────────────────────────────
    # Write a tiny launcher that imports torch + torchvision BEFORE inference.py
    # so both packages are fully initialized in the correct order.  Without this,
    # a circular-import race between torchvision sub-modules can cause:
    #   "partially initialized module 'torchvision' has no attribute 'extension'"
    # This is especially likely when colossalai / diffusers trigger torchvision
    # via multiple concurrent import chains.
    wrapper_src = textwrap.dedent(f"""\
        import os, sys, gc
        # ── Safety env vars (must be set before importing torch) ──────────────
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE",  "1")
        # Allow the CUDA allocator to release and re-use fragmented memory
        # instead of OOM-crashing when large contiguous blocks are needed.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True,max_split_size_mb:512")

        # ── Pre-warm torch + torchvision in the correct order ─────────────────
        import torch          # noqa: must come first
        try:
            import torchvision  # noqa
        except Exception:
            pass

        # ── Release any pre-existing GPU memory before inference ──────────────
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"  [OpenSora] GPU free before inference: {{free_gb:.2f}} GiB")

        # ── Hand off to the real inference script with correct sys.argv ───────
        sys.argv = {[script, cfg_file, "--prompt", prompt]!r}
        with open({script!r}) as _f:
            exec(compile(_f.read(), {script!r}, "exec"),
                 {{"__name__": "__main__", "__file__": {script!r}}})
    """)
    wrapper_file = tempfile.NamedTemporaryFile(
        mode="w", suffix="_opensora_launch.py", prefix="ttv_",
        dir=tmp_save_dir, delete=False,
    )
    wrapper_file.write(wrapper_src)
    wrapper_file.close()

    cmd = [sys.executable, wrapper_file.name]

    print(f"  [OpenSora] Running inference …")
    print(f"  [OpenSora] Prompt      : {prompt[:80]}")
    print(f"  [OpenSora] Frames/fps  : {num_frames} / {fps}")
    print(f"  [OpenSora] Resolution  : {resolution}  aspect={aspect_ratio}")

    env = os.environ.copy()
    if root not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")

    # ── NCCL ── prevent "ncclCommWindowDeregister: undefined symbol" ──────────
    env.setdefault("NCCL_P2P_DISABLE",  "1")
    env.setdefault("NCCL_IB_DISABLE",   "1")
    env.setdefault("NCCL_SHM_DISABLE",  "0")   # keep shared-mem transport

    # ── CUDA allocator — prevent OOM from fragmentation ───────────────────────
    # expandable_segments lets PyTorch grow/shrink allocations without
    # reserving a fixed pool — critical for models with varying tensor shapes.
    # max_split_size_mb limits the largest individual block to 512 MB so the
    # allocator can fulfil smaller requests from the gaps.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                   "expandable_segments:True,max_split_size_mb:512")

    # ── xformers ── if xformers is installed but built for a different torch, ─
    # diffusers will crash on import.  We test it here and, if broken, tell    ─
    # diffusers to skip xformers by setting XFORMERS_DISABLED in the env.     ─
    if not _xformers_available():
        env["XFORMERS_DISABLED"] = "1"         # diffusers respects this flag
        # Also tell colossalai not to try loading xformers
        env["COLOSSAL_NO_XFORMERS"] = "1"
        print(
            "  [OpenSora] xformers disabled (schema mismatch) — "
            "will use PyTorch SDPA.  Run setup_opensora.py to fix."
        )

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

    flash_ok = _use_flash_attn()
    model = build_module(
        dict(type="STDiT3-XL/2",
             from_pretrained=model_id,
             qk_norm=True,
             enable_flash_attn=flash_ok,
             enable_layernorm_kernel=flash_ok,   # also requires triton; skip if flash-attn broken
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
    # Set NCCL env vars BEFORE importing torch to prevent the
    # "ncclCommWindowDeregister: undefined symbol" error caused by
    # PyTorch being compiled against a newer NCCL than the system provides.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE",  "1")

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
    num_frames:     int  = 17,    # 17 = minimum 4k+1; ~0.7 s at 24 fps — fits ≤16 GB VRAM
    fps:            int  = 24,
    resolution:     str  = "240p",  # 240p for ≤16 GB; raise to 360p/480p with more VRAM
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
    num_frames:        int = 33,    # 33 = 4k+1; ~1.4 s at 24 fps — looped to fill scene
    fps:               int = 24,
    resolution:        str = "240p",  # upgrade to 480p/720p only on ≥24 GB VRAM
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
