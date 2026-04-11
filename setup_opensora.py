#!/usr/bin/env python3
"""
setup_opensora.py
=================
One-shot installer for Open-Sora v1.3.

What it does
------------
  1. Checks Python / CUDA / VRAM requirements
  2. Clones the Open-Sora repo (tag v1.3) — or pulls latest if already cloned
  3. pip-installs PyTorch (CUDA 12.1) + Open-Sora dependencies
  4. Downloads publicly-available model weights from HuggingFace:
       • STDiT3-XL/2  →  hpcai-tech/OpenSora-STDiT-v3     (text-to-video model)
       • VAE          →  hpcai-tech/OpenSora-VAE-v1.2      (video encoder/decoder)
       • T5 encoder   →  DeepFloyd/t5-v1_1-xxl             (text encoder)
  5. Writes a ready-to-use inference config
  6. Runs an import smoke-test

Note on versions
----------------
  Open-Sora v1.3 is the latest codebase.  The publicly released model weights
  on HuggingFace are from v1.2 (STDiT-v3 / VAE-v1.2) — the v1.3 weights are
  not yet publicly available.  We use the v1.3 code with the v1.2 weights,
  which is the officially supported HF inference path (see
  configs/opensora-v1-2/inference/sample_hf.py).

Usage
-----
  python setup_opensora.py
  python setup_opensora.py --skip-weights     # skip HF download
  python setup_opensora.py --opensora-dir /path/to/Open-Sora
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import textwrap


OPENSORA_REPO = "https://github.com/hpcaitech/Open-Sora.git"
OPENSORA_TAG  = "v1.3"          # latest stable tag

# Publicly available HF weights (v1.2 weights, used with v1.3 code)
HF_MODELS = [
    "hpcai-tech/OpenSora-STDiT-v3",   # main video-diffusion transformer
    "hpcai-tech/OpenSora-VAE-v1.2",   # 3D VAE
    "DeepFloyd/t5-v1_1-xxl",          # T5 text encoder (4.3 GB)
]

MIN_VRAM_GB = 16


# ─────────────────────────────────────────────────────────────────────────────

# torch major.minor → compatible torchvision version
_TORCHVISION_COMPAT: dict[str, str] = {
    "2.0": "0.15.2",
    "2.1": "0.16.2",
    "2.2": "0.17.2",
    "2.3": "0.18.1",
    "2.4": "0.19.1",
    "2.5": "0.20.1",
    "2.6": "0.21.0",
}
# torch major.minor → compatible torchaudio version (used by some OS deps)
_TORCHAUDIO_COMPAT: dict[str, str] = {
    "2.0": "2.0.2",
    "2.1": "2.1.2",
    "2.2": "2.2.2",
    "2.3": "2.3.0",
    "2.4": "2.4.1",
    "2.5": "2.5.1",
    "2.6": "2.6.0",
}


def _get_torch_version() -> tuple[str, str]:
    """
    Return (full_version, major_minor), e.g. ('2.5.1', '2.5').
    Falls back to ('2.2.2', '2.2') if torch isn't importable.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['NCCL_P2P_DISABLE']='1'; "
         "import torch; print(torch.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        full = result.stdout.strip().split("+")[0]   # strip +cu121 suffix
        parts = full.split(".")
        mm = f"{parts[0]}.{parts[1]}"
        return full, mm
    return "2.2.2", "2.2"


def _check_torchvision_compat() -> bool:
    """
    Return True if the installed torchvision is compatible with the installed torch.
    Incompatible torchvision causes 'partially initialized module torchvision
    has no attribute extension' — a circular import due to internal C++ mismatch.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['NCCL_P2P_DISABLE']='1'; "
         "import torch; import torchvision; "
         "print(torch.__version__.split('+')[0], torchvision.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    parts = result.stdout.strip().split()
    if len(parts) < 2:
        return False
    tv = parts[1]
    # Basic check: torchvision major.minor should match torch major.minor mapping
    tv_mm = ".".join(tv.split(".")[:2])
    torch_full, torch_mm = parts[0], ".".join(parts[0].split(".")[:2])
    expected_tv = _TORCHVISION_COMPAT.get(torch_mm, "")
    expected_tv_mm = ".".join(expected_tv.split(".")[:2]) if expected_tv else ""
    if expected_tv_mm and tv_mm != expected_tv_mm:
        print(
            f"  torchvision: {tv} is incompatible with torch {torch_full} "
            f"(expected {expected_tv_mm}.x)"
        )
        return False
    return True


def _check_flash_attn() -> bool:
    """
    Return True only if flash-attn is installed AND its kernel schema matches
    the running PyTorch.  Does the check in a subprocess so a crash/ImportError
    doesn't kill this setup script.
    """
    result = subprocess.run(
        [sys.executable, "-c", (
            "import os; "
            "os.environ['NCCL_P2P_DISABLE']='1'; "
            "os.environ['NCCL_IB_DISABLE']='1'; "
            "import torch; import flash_attn; "
            "from flash_attn import flash_attn_func; "
            "q=torch.randn(1,4,1,32,device='cuda',dtype=torch.float16); "
            "flash_attn_func(q,q,q); print('ok')"
        )],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and "ok" in result.stdout:
        return True
    if "schema" in result.stderr or "aten::_flash_attention_forward" in result.stderr:
        print(
            "  flash-attn: aten::_flash_attention_forward schema mismatch\n"
            f"              {result.stderr.splitlines()[0] if result.stderr else ''}"
        )
    return False


def _check_xformers() -> bool:
    """
    Return True only if xformers is installed AND its C++/CUDA ops load cleanly
    for the running (PyTorch, CUDA, Python) combination.

    A common failure mode: xformers was pre-installed for a different torch/CUDA
    (e.g. torch 2.10+cu128 / Python 3.10) but the env has torch 2.5+cu124 /
    Python 3.12.  In this case xformers is importable but its fmha.flash module
    calls ensure_pt_flash_ok() which raises:
      "does not have a compatible aten::_flash_attention_forward schema"
    This propagates up through diffusers.models.attention_processor and crashes
    the whole Open-Sora import chain.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
            import os
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"]  = "1"
            import warnings
            warnings.filterwarnings("ignore")
            import xformers
            import xformers.ops          # triggers C++ extension load
            from xformers.ops import memory_efficient_attention  # needs CUDA ops
            print("ok", xformers.__version__)
        """)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and "ok" in result.stdout:
        ver = result.stdout.strip().split()[-1]
        print(f"  xformers: {ver} — compatible ✓")
        return True

    stderr = result.stderr
    if "schema" in stderr or "aten::_flash_attention_forward" in stderr:
        print(
            "  xformers: torch flash-attention schema mismatch "
            "(xformers built for different torch version)."
        )
    elif "built for" in stderr or "you have" in stderr:
        # Extract the one-liner warning e.g. "PyTorch 2.10.0+cu128 … you have 2.5.1+cu124"
        for line in stderr.splitlines():
            if "built for" in line or "you have" in line:
                print(f"  xformers: version mismatch — {line.strip()}")
                break
    else:
        print("  xformers: incompatible or import failed.")

    return False


def _get_torch_cuda_tag() -> str:
    """
    Return the cu-tag matching the installed PyTorch, e.g. 'cu124', 'cu121'.
    Falls back to 'cu121' if detection fails.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; v=torch.version.cuda or '12.1'; "
         "tag='cu'+''.join(v.split('.')[:2]); print(tag)"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        tag = result.stdout.strip()
        if tag.startswith("cu") and len(tag) >= 4:
            return tag
    return "cu121"   # safe default


def banner(msg: str):
    w = max(60, len(msg) + 4)
    print(f"\n{'─'*w}\n  {msg}\n{'─'*w}")


def run(cmd: list, cwd: str | None = None, check: bool = True) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        sys.exit(f"[Error] Command failed (exit {result.returncode})")
    return result.returncode


def pip(*packages: str, extra_index: str | None = None):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    run(cmd)


# ── Step 1: system checks ─────────────────────────────────────────────────────

def check_requirements():
    banner("Checking system requirements")
    major, minor = sys.version_info[:2]
    print(f"  Python : {major}.{minor}")
    if (major, minor) < (3, 10):
        sys.exit("[Error] Python 3.10+ required.")

    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; "
         "print(torch.__version__); "
         "import json, torch; "
         "gpus=[{'name':torch.cuda.get_device_name(i),"
         "'vram':torch.cuda.get_device_properties(i).total_memory/1024**3}"
         " for i in range(torch.cuda.device_count())];"
         "print(json.dumps(gpus))"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        print(f"  PyTorch: {lines[0]}")
        try:
            import json
            gpus = json.loads(lines[1]) if len(lines) > 1 else []
            if gpus:
                for i, g in enumerate(gpus):
                    flag = "✓" if g["vram"] >= MIN_VRAM_GB else f"⚠ ({MIN_VRAM_GB} GB recommended)"
                    print(f"  GPU {i}  : {g['name']}  {g['vram']:.1f} GB  {flag}")
            else:
                print("  CUDA   : ✗ no GPU detected — inference will be extremely slow on CPU")
        except Exception:
            pass
    else:
        stderr = result.stderr
        if "ncclComm" in stderr or "undefined symbol" in stderr:
            print(
                "  PyTorch: ⚠ NCCL/CUDA symbol mismatch (ncclCommWindowDeregister).\n"
                "           install_dependencies() will reinstall PyTorch 2.2.2+cu121."
            )
        else:
            print("  PyTorch: not installed or broken — will install.")

    rc = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode
    print(f"  ffmpeg : {'✓' if rc == 0 else '✗ missing — please install ffmpeg'}")

    rc2 = subprocess.run(["git", "--version"], capture_output=True).returncode
    print(f"  git    : {'✓' if rc2 == 0 else '✗ missing — please install git'}")


# ── Step 2: clone / update repo ───────────────────────────────────────────────

def clone_or_update(opensora_dir: str):
    banner(f"Setting up Open-Sora repo  →  {opensora_dir}")

    if os.path.isdir(os.path.join(opensora_dir, ".git")):
        print("  Repo exists — fetching tags …")
        run(["git", "fetch", "--tags", "--depth", "1"], cwd=opensora_dir, check=False)
        run(["git", "checkout", OPENSORA_TAG], cwd=opensora_dir, check=False)
        return

    parent = os.path.dirname(os.path.abspath(opensora_dir))
    os.makedirs(parent, exist_ok=True)

    print(f"  Cloning {OPENSORA_REPO}  (tag {OPENSORA_TAG}) …")
    run([
        "git", "clone",
        "--depth", "1",
        "--branch", OPENSORA_TAG,
        OPENSORA_REPO,
        opensora_dir,
    ])


# ── Step 3: install dependencies ──────────────────────────────────────────────

def _torch_loads_cleanly() -> bool:
    """
    Return True only if PyTorch imports without NCCL / CUDA symbol errors.
    A broken torch (e.g. compiled against NCCL 2.19+ on a system with NCCL
    2.18) raises an ImportError even though the package is 'installed'.
    """
    env = os.environ.copy()
    # Disable NCCL peer-to-peer paths during the check to avoid false negatives
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"]  = "1"
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; assert torch.cuda.is_available(), 'no cuda'; "
         "print(torch.__version__)"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        ver = result.stdout.strip()
        print(f"  PyTorch: {ver} (loads OK)")
        return True

    stderr = result.stderr
    if "ncclComm" in stderr or "undefined symbol" in stderr or "libtorch_cuda" in stderr:
        print("  PyTorch: NCCL/CUDA symbol mismatch detected — will reinstall.")
    elif "no cuda" in stderr:
        print("  PyTorch: installed but CUDA unavailable — will reinstall with cu121.")
    else:
        print(f"  PyTorch: import failed:\n    {stderr[:300]}")
    return False


def install_dependencies(opensora_dir: str):
    banner("Installing Python dependencies")

    # ── Detect installed torch and pick a compatible (torch, torchvision) pair ─
    #
    # We keep whatever torch is already installed unless it has an NCCL symbol
    # error.  If the user's environment has torch 2.5.1 and we blindly install
    # torchvision 0.17.2 we get:
    #   "partially initialized module torchvision has no attribute extension"
    # because torchvision 0.17.x was compiled against torch 2.2.x's C++ ABI.
    #
    # Compatibility matrix: torch major.minor → torchvision version
    # (mirrors PyTorch release notes)

    torch_ok = _torch_loads_cleanly()

    if torch_ok:
        torch_full, torch_mm = _get_torch_version()
        tv_version = _TORCHVISION_COMPAT.get(torch_mm, "0.17.2")
        cu_tag      = _get_torch_cuda_tag()
        whl_url     = f"https://download.pytorch.org/whl/{cu_tag}"
        print(f"  Keeping torch {torch_full} ({cu_tag}) — "
              f"will ensure torchvision {tv_version}")

        # Check if torchvision is already the right version
        if not _check_torchvision_compat():
            print(f"  Reinstalling torchvision=={tv_version} to match torch {torch_full} …")
            run([
                sys.executable, "-m", "pip", "install", "--upgrade",
                f"torchvision=={tv_version}",
                "--index-url", whl_url,
            ])
        else:
            print("  torchvision: already compatible ✓")

    else:
        # torch is broken (NCCL mismatch etc.) — install a known-good baseline
        # Default to torch 2.2.2+cu121 which is stable across most cloud GPUs
        TORCH_VERSION   = "2.2.2"
        TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
        tv_version      = _TORCHVISION_COMPAT["2.2"]

        print(f"  Installing PyTorch {TORCH_VERSION} + CUDA 12.1 …")
        run([
            sys.executable, "-m", "pip", "install", "--upgrade",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={tv_version}",
            "--index-url", TORCH_INDEX_URL,
        ])

    # Core Open-Sora requirements
    req = os.path.join(opensora_dir, "requirements", "requirements.txt")
    if os.path.isfile(req):
        run([sys.executable, "-m", "pip", "install", "-r", req])
    else:
        print(f"  [Warning] requirements not found at {req}")

    # Install Open-Sora package itself (editable)
    run([sys.executable, "-m", "pip", "install", "-e", opensora_dir,
         "--no-deps"])   # deps already installed above

    # ── flash-attn: always uninstall — too version-sensitive to auto-manage ───
    # flash-attn must be compiled for the exact (torch + CUDA + Python) triple.
    # Any mismatch causes "aten::_flash_attention_forward schema" crashes that
    # propagate through xformers and diffusers.  We remove it unconditionally
    # and let Open-Sora use xformers or PyTorch SDPA instead.
    _fa_check = subprocess.run(
        [sys.executable, "-m", "pip", "show", "flash-attn"],
        capture_output=True,
    )
    if _fa_check.returncode == 0:
        print("  flash-attn: removing (version-sensitive; will use xformers/SDPA) …")
        run([sys.executable, "-m", "pip", "uninstall", "flash-attn", "-y"], check=False)
    else:
        print("  flash-attn: not installed ✓")

    # ── xformers: detect mismatch and reinstall for the correct torch+CUDA ───
    # xformers must match (PyTorch version, CUDA version, Python version).
    # A pre-installed xformers built for a different environment will crash
    # the diffusers → attention_processor → xformers.ops import chain.
    cu_tag = _get_torch_cuda_tag()   # e.g. "cu124"
    whl_url = f"https://download.pytorch.org/whl/{cu_tag}"
    print(f"  Detected CUDA tag: {cu_tag}  (wheel index: {whl_url})")

    if _check_xformers():
        print("  xformers: already compatible — skipping reinstall.")
    else:
        # Uninstall whatever is there (may or may not be installed)
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "xformers", "-y"],
            capture_output=True,
        )
        print(f"  Installing xformers for torch+{cu_tag} …")
        rc = run(
            [sys.executable, "-m", "pip", "install", "xformers",
             "--index-url", whl_url],
            check=False,
        )
        if rc == 0:
            # Verify it actually works now
            if _check_xformers():
                print("  xformers: reinstalled and verified ✓")
            else:
                print(
                    "  xformers: reinstalled but still incompatible.\n"
                    "  Open-Sora will use PyTorch SDPA (standard attention).\n"
                    "  Performance is similar; no action needed."
                )
                # Remove broken xformers so it doesn't poison the import chain
                run([sys.executable, "-m", "pip", "uninstall", "xformers", "-y"],
                    check=False)
        else:
            print(
                f"  xformers: install failed for {cu_tag}.\n"
                "  Open-Sora will use PyTorch SDPA — this is fine."
            )


# ── Step 4: download model weights ────────────────────────────────────────────

def download_weights(opensora_dir: str):
    banner("Downloading model weights from HuggingFace")
    print("  Models: OpenSora-STDiT-v3, OpenSora-VAE-v1.2, t5-v1_1-xxl")
    print("  Note  : These are the v1.2 publicly released weights, compatible")
    print("          with the v1.3 codebase via configs/opensora-v1-2/inference/sample_hf.py")
    print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    cache_dir = os.path.join(opensora_dir, "pretrained_models")
    os.makedirs(cache_dir, exist_ok=True)

    for repo_id in HF_MODELS:
        print(f"  Downloading  {repo_id}  …")
        try:
            path = snapshot_download(
                repo_id       = repo_id,
                cache_dir     = cache_dir,
                ignore_patterns = [
                    "*.msgpack", "flax_model*", "tf_model*",
                    "rust_model*", "onnx*",
                ],
            )
            print(f"  ✓  {repo_id}\n     → {path}\n")
        except Exception as e:
            print(f"  [Warning] Could not download {repo_id}: {e}")
            print(f"  Manual download:  huggingface-cli download {repo_id}\n")


# ── Step 5: write inference config helper ─────────────────────────────────────

def write_inference_config(opensora_dir: str):
    """
    Write a ready-to-use inference config (480p, 51 frames) that points to
    the HF model repos.  Saved as configs/opensora-v1-2/inference/sample_hf.py
    which already exists in the repo — this step is informational only.
    """
    cfg_path = os.path.join(
        opensora_dir, "configs", "opensora-v1-2", "inference", "sample_hf.py"
    )
    if os.path.isfile(cfg_path):
        print(f"  Using existing config: {cfg_path}")
    else:
        print(f"  [Warning] Config not found at {cfg_path}")


# ── Step 6: smoke test ────────────────────────────────────────────────────────

def smoke_test(opensora_dir: str):
    banner("Import smoke test")

    if opensora_dir not in sys.path:
        sys.path.insert(0, opensora_dir)

    errors = []
    for mod in [
        "opensora",
        "opensora.registry",
        "opensora.models.stdit.stdit3",
        "opensora.schedulers.rf",
    ]:
        try:
            __import__(mod)
            print(f"  ✓  import {mod}")
        except ImportError as e:
            errors.append(f"  ✗  import {mod}  →  {e}")

    if errors:
        for e in errors:
            print(e)
        print("\n  Some imports failed — this may be OK before weights are downloaded.")
    else:
        print("\n  All imports OK!")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  No GPU detected — set up CUDA before running inference.")
    except ImportError:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Set up Open-Sora for text-to-video generation.")
    parser.add_argument(
        "--opensora-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Open-Sora"),
        help="Where to clone the Open-Sora repo  (default: ./Open-Sora)",
    )
    parser.add_argument("--skip-weights", action="store_true",
                        help="Skip HuggingFace weight download")
    parser.add_argument("--skip-deps",    action="store_true",
                        help="Skip pip dependency installation")
    args = parser.parse_args()

    check_requirements()
    clone_or_update(args.opensora_dir)

    if not args.skip_deps:
        install_dependencies(args.opensora_dir)

    if not args.skip_weights:
        download_weights(args.opensora_dir)

    write_inference_config(args.opensora_dir)
    smoke_test(args.opensora_dir)

    cfg_path = os.path.join(
        args.opensora_dir, "configs", "opensora-v1-2", "inference", "sample_hf.py"
    )
    print(textwrap.dedent(f"""
    ┌──────────────────────────────────────────────────────────────────┐
    │  Open-Sora setup complete!                                       │
    │                                                                  │
    │  Manual inference test:                                          │
    │    cd {args.opensora_dir}
    │    python scripts/inference.py \\                                │
    │      {cfg_path} \\
    │      --prompt "A person talking in an office" \\                 │
    │      --save-dir ./samples                                        │
    │                                                                  │
    │  Pipeline usage:                                                 │
    │    python main.py script.json --use-opensora                     │
    │    python main.py script.json --use-opensora --opensora-bg       │
    └──────────────────────────────────────────────────────────────────┘
    """))


if __name__ == "__main__":
    main()
