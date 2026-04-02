"""
wav2lip_runner.py

Wrapper around Wav2Lip inference for GPU-accelerated lip synchronisation.

Wav2Lip takes:
  • a face image (or video) of the speaker
  • a TTS audio file
and outputs a short video of the face talking in perfect sync with the audio.

Setup
-----
Run setup_models.py once to clone the Wav2Lip repo and download model weights.
Or set WAV2LIP_DIR and WAV2LIP_CHECKPOINT env vars to custom paths.

Reference
---------
  Wav2Lip: Accurately Lip-syncing Videos In The Wild  (Prajwal et al., ACM MM 2020)
  https://github.com/Rudrabha/Wav2Lip
"""

from __future__ import annotations
import os
import sys
import subprocess
import tempfile
import shutil

# ── paths (overridable via env vars) ─────────────────────────────────────────
WAV2LIP_DIR        = os.environ.get("WAV2LIP_DIR",        "Wav2Lip")
WAV2LIP_CHECKPOINT = os.environ.get("WAV2LIP_CHECKPOINT", "Wav2Lip/checkpoints/wav2lip_gan.pth")
FACE_DET_MODEL     = os.environ.get("FACE_DET_MODEL",
                        "Wav2Lip/face_detection/detection/sfd/s3fd-619a316812.pth")


# ── sanity checks ─────────────────────────────────────────────────────────────

def is_wav2lip_ready() -> bool:
    """Return True if Wav2Lip repo and model weights are present."""
    inference = os.path.join(WAV2LIP_DIR, "inference.py")
    return (
        os.path.isfile(inference)
        and os.path.isfile(WAV2LIP_CHECKPOINT)
        and os.path.isfile(FACE_DET_MODEL)
    )


def check_or_abort():
    """Abort with a helpful message if Wav2Lip is not set up."""
    if not is_wav2lip_ready():
        missing = []
        if not os.path.isfile(os.path.join(WAV2LIP_DIR, "inference.py")):
            missing.append(f"  • {WAV2LIP_DIR}/inference.py  (repo not cloned)")
        if not os.path.isfile(WAV2LIP_CHECKPOINT):
            missing.append(f"  • {WAV2LIP_CHECKPOINT}  (model weights)")
        if not os.path.isfile(FACE_DET_MODEL):
            missing.append(f"  • {FACE_DET_MODEL}  (face detection weights)")
        sys.exit(
            "\n[Error] Wav2Lip is not set up. Missing:\n"
            + "\n".join(missing)
            + "\n\nFix: run  python setup_models.py  first.\n"
        )


# ── core inference ────────────────────────────────────────────────────────────

def run_wav2lip(
    face_image_path: str,
    audio_path: str,
    output_path: str,
    use_gan: bool         = True,
    resize_factor: int    = 1,
    fps: int              = 25,
    pad_top: int          = 0,
    pad_bottom: int       = 10,
    pad_left: int         = 0,
    pad_right: int        = 0,
    nosmooth: bool        = False,
) -> str:
    """
    Run Wav2Lip on a still face image + audio file.

    Parameters
    ----------
    face_image_path : path to a portrait PNG/JPG (speaker's face)
    audio_path      : path to a WAV/MP3 audio file (TTS speech)
    output_path     : where to save the resulting MP4 video
    use_gan         : True → wav2lip_gan.pth (sharper), False → wav2lip.pth (faster)
    resize_factor   : downsample by this factor (1 = full res)
    fps             : output video frame rate
    pad_*           : face bounding-box padding (pixels); increase pad_bottom if
                      mouth is clipped at the chin
    nosmooth        : disable temporal smoothing (faster but less smooth)

    Returns
    -------
    output_path on success
    """
    check_or_abort()

    # Choose checkpoint
    checkpoint = WAV2LIP_CHECKPOINT
    if not use_gan:
        alt = WAV2LIP_CHECKPOINT.replace("wav2lip_gan.pth", "wav2lip.pth")
        if os.path.isfile(alt):
            checkpoint = alt

    cmd = [
        sys.executable,                          # same Python that runs this script
        os.path.join(WAV2LIP_DIR, "inference.py"),
        "--checkpoint_path", checkpoint,
        "--face",            face_image_path,
        "--audio",           audio_path,
        "--outfile",         output_path,
        "--resize_factor",   str(resize_factor),
        "--fps",             str(fps),
        "--pads",            str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
        "--face_det_batch_size", "4",
        "--wav2lip_batch_size",  "128",
    ]
    if nosmooth:
        cmd.append("--nosmooth")

    env = os.environ.copy()
    # Add Wav2Lip dir to PYTHONPATH so its internal imports resolve
    env["PYTHONPATH"] = WAV2LIP_DIR + os.pathsep + env.get("PYTHONPATH", "")

    print(f"  [Wav2Lip] Running inference …  face={os.path.basename(face_image_path)}")
    result = subprocess.run(
        cmd,
        cwd=WAV2LIP_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("  [Wav2Lip] STDERR:", result.stderr[-1500:])
        raise RuntimeError(
            f"Wav2Lip inference failed (exit code {result.returncode}).\n"
            f"Check the output above for details."
        )

    if not os.path.exists(output_path):
        # Wav2Lip sometimes writes to cwd/results/result_voice.mp4
        fallback = os.path.join(WAV2LIP_DIR, "results", "result_voice.mp4")
        if os.path.exists(fallback):
            shutil.move(fallback, output_path)
        else:
            raise FileNotFoundError(
                f"Wav2Lip did not produce an output file at {output_path}."
            )

    sz = os.path.getsize(output_path)
    print(f"  [Wav2Lip] Done → {output_path}  ({sz // 1024} KB)")
    return output_path


# ── batch helper ──────────────────────────────────────────────────────────────

def make_listening_video(
    face_image_path: str,
    duration_s: float,
    output_path: str,
    fps: int = 25,
) -> str:
    """
    Create a static-face 'listening' video by looping the portrait image.
    Used for the character that is NOT speaking in a given segment.

    Returns output_path.
    """
    try:
        try:
            from moviepy.editor import ImageClip  # type: ignore  # v1
        except ImportError:
            from moviepy import ImageClip          # type: ignore  # v2

        clip = ImageClip(face_image_path, duration=duration_s)
        clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio=False,
            logger=None,
        )
    except Exception as e:
        # Fallback: ffmpeg directly
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", face_image_path,
            "-t", str(duration_s),
            "-vf", f"fps={fps}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    return output_path
