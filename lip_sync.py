"""
lip_sync.py

Analyses an audio file and returns per-frame mouth-opening values in [0, 1].

Pipeline
--------
1. Load audio with librosa (or soundfile fallback)
2. Compute short-time RMS energy at video frame-rate resolution
3. Smooth with a Gaussian to avoid flicker
4. Normalise → [0, 1]
5. Apply a soft threshold so silence reads as 0.0
"""

from __future__ import annotations
import os
import math


def extract_mouth_openings(audio_path: str, fps: int = 24) -> "list[float]":
    """
    Return a list of mouth-opening values (one per video frame) for the
    duration of *audio_path*.

    Values are in [0.0, 1.0] where:
      0.0 = mouth closed (silence)
      1.0 = mouth fully open (peak loudness)

    Falls back to a simple silence list if the audio cannot be loaded.
    """
    # ── try librosa ───────────────────────────────────────────────────────────
    try:
        import librosa                          # type: ignore
        import numpy as np
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        y, sr = librosa.load(audio_path, sr=None, mono=True)

        hop_length = max(1, int(sr / fps))
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Normalise
        peak = rms.max()
        if peak > 0:
            rms = rms / peak

        # Smooth (~1 frame half-window)
        rms = gaussian_filter1d(rms.astype(float), sigma=1.2)

        # Soft threshold – below 5 % of peak ≈ silence
        threshold = 0.05
        rms = np.where(rms < threshold, 0.0, rms)

        # Clamp
        rms = np.clip(rms, 0.0, 1.0)
        return rms.tolist()

    except Exception as e:
        print(f"  [LipSync] librosa path failed ({e}), trying soundfile …")

    # ── try soundfile + numpy ─────────────────────────────────────────────────
    try:
        import soundfile as sf                  # type: ignore
        import numpy as np
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        data, sr = sf.read(audio_path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)

        hop = max(1, int(sr / fps))
        n_frames = max(1, len(data) // hop)
        rms_list = []
        for i in range(n_frames):
            chunk = data[i * hop : (i + 1) * hop]
            rms_list.append(float(np.sqrt(np.mean(chunk ** 2))))

        arr = np.array(rms_list)
        if arr.max() > 0:
            arr /= arr.max()
        arr = gaussian_filter1d(arr, sigma=1.2)
        arr = np.clip(arr, 0.0, 1.0)
        return arr.tolist()

    except Exception as e:
        print(f"  [LipSync] soundfile path failed ({e}), generating synthetic curve …")

    # ── synthetic fallback ────────────────────────────────────────────────────
    return _synthetic_lipsync(audio_path, fps)


def _synthetic_lipsync(audio_path: str, fps: int) -> "list[float]":
    """
    If we can't read the audio, generate a plausible talking curve based
    on the estimated duration of the audio file.
    """
    import math

    duration = _get_duration_fallback(audio_path)
    n_frames = max(1, int(duration * fps))

    # Sinusoidal syllable rhythm at ~4 syllables/s
    syl_rate = 4.0
    values = []
    for i in range(n_frames):
        t = i / fps
        val = max(0.0, math.sin(2 * math.pi * syl_rate * t)) * 0.8
        values.append(val)
    return values


def _get_duration_fallback(audio_path: str) -> float:
    """Estimate audio duration without librosa/soundfile."""
    try:
        import wave
        with wave.open(audio_path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        pass

    # File-size heuristic for mp3 (~ 128 kbps → 16 KB/s)
    try:
        size = os.path.getsize(audio_path)
        return size / 16_000
    except Exception:
        return 3.0  # assume 3 seconds


def get_audio_duration(audio_path: str) -> float:
    """Return duration of *audio_path* in seconds."""
    try:
        import librosa                          # type: ignore
        y, sr = librosa.load(audio_path, sr=None)
        return float(len(y) / sr)
    except Exception:
        return _get_duration_fallback(audio_path)
