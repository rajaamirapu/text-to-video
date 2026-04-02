"""
lip_sync.py

Analyses an audio file and returns per-frame mouth-opening values in [0, 1].

Primary path  : soundfile + numpy  (no numba / no librosa)
                Works on Python 3.12+ without distutils issues.
Optional path : librosa (used only if soundfile fails)
Fallback      : pydub → wave module → synthetic curve

Pipeline
--------
1. Load audio samples
2. Compute RMS energy at video frame-rate resolution
3. Gaussian-smooth to avoid flicker
4. Normalise → [0, 1] with soft silence threshold
"""

from __future__ import annotations
import math
import os
import struct
import wave


# ── helpers ───────────────────────────────────────────────────────────────────

def _smooth(values: list[float], sigma: float = 1.5) -> list[float]:
    """Simple Gaussian-kernel smoothing (avoids scipy dependency)."""
    if sigma <= 0 or len(values) < 3:
        return values
    radius = max(1, int(3 * sigma))
    kernel = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)]
    k_sum  = sum(kernel)
    kernel = [k / k_sum for k in kernel]

    n      = len(values)
    out    = []
    for i in range(n):
        acc = 0.0
        for j, k in enumerate(kernel):
            idx = i + j - radius
            idx = max(0, min(n - 1, idx))   # clamp at boundaries
            acc += values[idx] * k
        out.append(acc)
    return out


def _rms_per_frame(samples: list[float] | "np.ndarray", sr: int, fps: int) -> list[float]:
    """
    Compute root-mean-square energy in non-overlapping windows
    aligned to video frame rate.
    """
    import math
    hop = max(1, sr // fps)
    rms = []
    n   = len(samples)
    i   = 0
    while i < n:
        chunk = samples[i : i + hop]
        if len(chunk) == 0:
            break
        sq = sum(float(s) ** 2 for s in chunk) / len(chunk)
        rms.append(math.sqrt(sq))
        i += hop
    return rms


def _normalise_and_threshold(rms: list[float], threshold: float = 0.05) -> list[float]:
    """Normalise to [0,1] and zero out values below threshold (silence)."""
    peak = max(rms) if rms else 0.0
    if peak == 0:
        return [0.0] * len(rms)
    normed = [v / peak for v in rms]
    return [0.0 if v < threshold else min(1.0, v) for v in normed]


# ── primary loader: soundfile ─────────────────────────────────────────────────

def _load_soundfile(path: str):
    """Return (samples_as_list_of_float, sample_rate) using soundfile."""
    import soundfile as sf          # type: ignore
    data, sr = sf.read(path, always_2d=False, dtype="float32")
    if data.ndim == 2:              # stereo → mono
        data = data.mean(axis=1)
    return data.tolist(), sr


# ── fallback loaders ──────────────────────────────────────────────────────────

def _load_wave(path: str):
    """Read a .wav file using the stdlib wave module."""
    with wave.open(path, "r") as wf:
        sr        = wf.getframerate()
        n_frames  = wf.getnframes()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw       = wf.readframes(n_frames)

    fmt_map = {1: "b", 2: "h", 4: "i"}
    fmt = fmt_map.get(sampwidth, "h")
    samples_all = list(struct.unpack(f"<{n_frames * n_ch}{fmt}", raw))

    # mix down to mono
    if n_ch > 1:
        samples = [
            sum(samples_all[i : i + n_ch]) / n_ch
            for i in range(0, len(samples_all), n_ch)
        ]
    else:
        samples = samples_all

    # normalise int → float
    max_val = float(2 ** (8 * sampwidth - 1))
    samples = [s / max_val for s in samples]
    return samples, sr


def _load_mp3_via_pydub(path: str):
    """Convert mp3 → wav in memory using pydub, then read with wave."""
    try:
        from pydub import AudioSegment  # type: ignore
        seg  = AudioSegment.from_file(path).set_channels(1)
        sr   = seg.frame_rate
        raw  = seg.raw_data
        sw   = seg.sample_width
        fmt_map = {1: "b", 2: "h", 4: "i"}
        fmt  = fmt_map.get(sw, "h")
        n    = len(raw) // sw
        samples = list(struct.unpack(f"<{n}{fmt}", raw[:n * sw]))
        max_val = float(2 ** (8 * sw - 1))
        samples = [s / max_val for s in samples]
        return samples, sr
    except Exception as e:
        raise RuntimeError(f"pydub failed: {e}")


def _load_librosa(path: str):
    """Load with librosa (requires numba; may fail on Python 3.12)."""
    import librosa          # type: ignore
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.tolist(), int(sr)


def _load_audio(path: str) -> tuple[list[float], int]:
    """
    Try loaders in order from most to least reliable on Python 3.12.
    Returns (samples, sample_rate).
    """
    loaders = [
        ("soundfile",  _load_soundfile),
        ("wave",       _load_wave),
        ("pydub",      _load_mp3_via_pydub),
        ("librosa",    _load_librosa),
    ]
    last_err = None
    for name, fn in loaders:
        try:
            samples, sr = fn(path)
            if samples and sr > 0:
                print(f"  [LipSync] Loaded audio via {name}  ({len(samples)} samples @ {sr} Hz)")
                return samples, sr
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All audio loaders failed. Last error: {last_err}")


# ── public API ────────────────────────────────────────────────────────────────

def extract_mouth_openings(audio_path: str, fps: int = 25) -> list[float]:
    """
    Return a list of mouth-opening values (one per video frame).
    Values are in [0.0, 1.0]:  0 = mouth closed,  1 = wide open.

    Works on Python 3.12+ without librosa / numba.
    """
    try:
        samples, sr = _load_audio(audio_path)
        rms         = _rms_per_frame(samples, sr, fps)
        rms         = _smooth(rms, sigma=1.5)
        rms         = _normalise_and_threshold(rms, threshold=0.04)
        return rms
    except Exception as e:
        print(f"  [LipSync] Audio analysis failed: {e} — using synthetic curve")
        return _synthetic_lipsync(audio_path, fps)


def _synthetic_lipsync(audio_path: str, fps: int) -> list[float]:
    """Fallback: generate a plausible talking rhythm without reading audio."""
    duration = _get_duration_fallback(audio_path)
    n_frames = max(1, int(duration * fps))
    syl_rate = 4.0          # ~4 syllables per second
    return [
        max(0.0, math.sin(2 * math.pi * syl_rate * (i / fps))) * 0.75
        for i in range(n_frames)
    ]


def get_audio_duration(audio_path: str) -> float:
    """Return duration of *audio_path* in seconds."""
    try:
        samples, sr = _load_audio(audio_path)
        return len(samples) / sr
    except Exception:
        return _get_duration_fallback(audio_path)


def _get_duration_fallback(audio_path: str) -> float:
    """Estimate audio duration without any audio library."""
    try:
        with wave.open(audio_path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        pass
    # File-size heuristic for mp3 @ ~128 kbps → ~16 KB/s
    try:
        return os.path.getsize(audio_path) / 16_000
    except Exception:
        return 3.0
