"""
tts.py  —  Text-to-speech with natural voices (open-source first)

Engine priority
---------------
1. piper-tts  (fully open-source, offline after first download — **primary**)
                Uses lightweight ONNX voices from
                https://huggingface.co/rhasspy/piper-voices  (MIT/CC-BY).
                Voices tried in order:
                  male   → en_US-ryan-high, en_US-lessac-medium, en_GB-alan-medium
                  female → en_US-amy-medium, en_US-hfc_female-medium,
                           en_GB-jenny_dioco-medium
2. edge-tts   (Microsoft neural TTS — high quality but requires a network
                round-trip to Azure; kept as a fallback)
3. pyttsx3    (offline system voices — last-line quality fallback)
4. Silent WAV  (final fallback so the pipeline never crashes)

Why Piper?
----------
Piper is an Apache-2.0 licensed neural TTS engine from the Rhasspy project. It
runs completely on CPU via onnxruntime, ships small voice models (~30-60 MB
each), and produces natural-sounding speech without any cloud dependency. That
makes this module genuinely open-source and offline-capable, which was not
true of the previous edge-tts-first implementation (Azure neural voices are
proprietary and require network access to Microsoft's servers on every call).

Install
-------
  pip install piper-tts            # primary engine (plus onnxruntime)
  pip install edge-tts pyttsx3     # optional fallbacks

Configuration
-------------
  PIPER_VOICE_DIR   override the directory used to cache voice models.
                    Default:  ~/.cache/piper/voices
"""

from __future__ import annotations
import asyncio
import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. piper-tts  (open-source neural TTS — primary)
# ─────────────────────────────────────────────────────────────────────────────

# Voices tried in preference order. Names follow Piper's convention:
# "<lang_family>_<lang_region>-<voice_name>-<quality>".
PIPER_MALE_VOICES = [
    "en_US-ryan-high",       # confident American male (high quality, ~60 MB)
    "en_US-lessac-medium",   # neutral American (medium quality, ~30 MB)
    "en_GB-alan-medium",     # British male
]

PIPER_FEMALE_VOICES = [
    "en_US-amy-medium",              # warm American female
    "en_US-hfc_female-medium",       # clear American female
    "en_GB-jenny_dioco-medium",      # British female
]

# Process-wide cache so we don't reload the ONNX model on every line of dialogue.
_PIPER_VOICE_CACHE: dict = {}


def _piper_voice_dir() -> Path:
    """Return (creating if needed) the directory used to cache Piper voices."""
    override = os.environ.get("PIPER_VOICE_DIR")
    root = Path(override) if override else Path.home() / ".cache" / "piper" / "voices"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _piper_ensure_voice(voice_name: str) -> Optional[Path]:
    """
    Ensure the given Piper voice model is available locally.
    Returns the path to the .onnx file, or None if the download failed.
    """
    voice_dir = _piper_voice_dir()
    onnx_path = voice_dir / f"{voice_name}.onnx"
    config_path = voice_dir / f"{voice_name}.onnx.json"
    if onnx_path.is_file() and config_path.is_file():
        return onnx_path

    try:
        from piper.download_voices import download_voice
    except ImportError:
        return None

    try:
        print(f"  [TTS] Downloading Piper voice '{voice_name}' → {voice_dir}")
        download_voice(voice_name, voice_dir)
        if onnx_path.is_file() and config_path.is_file():
            return onnx_path
    except Exception as e:
        print(f"  [TTS] Failed to download piper voice '{voice_name}': {e}")
    return None


def _piper_tts(text: str, wav_path: str, voice_list: list) -> bool:
    """
    Try each Piper voice in *voice_list* in turn.
    Returns True if any succeeds. Writes a PCM-16 mono WAV to *wav_path*.
    """
    try:
        from piper import PiperVoice
    except ImportError:
        print("  [TTS] piper-tts not installed.  Run: pip install piper-tts")
        return False

    raw_path = wav_path.replace(".wav", "_piper_raw.wav")

    for voice_name in voice_list:
        try:
            voice = _PIPER_VOICE_CACHE.get(voice_name)
            if voice is None:
                onnx = _piper_ensure_voice(voice_name)
                if onnx is None:
                    continue
                voice = PiperVoice.load(str(onnx))
                _PIPER_VOICE_CACHE[voice_name] = voice

            with wave.open(raw_path, "wb") as wf:
                voice.synthesize_wav(text, wf)

            if not (os.path.isfile(raw_path) and os.path.getsize(raw_path) > 0):
                continue

            # Normalise to 44.1 kHz mono PCM so downstream lip-sync / muxing
            # steps get a consistent format regardless of the voice's native
            # sample rate (Piper medium voices are 22050 Hz, high are 22050 Hz
            # or 44100 Hz depending on the model).
            if _to_wav(raw_path, wav_path):
                _cleanup(raw_path)
                print(f"  [TTS] piper ✓  voice={voice_name}")
                return True

            # If ffmpeg is unavailable, fall back to using the raw WAV directly.
            os.replace(raw_path, wav_path)
            print(f"  [TTS] piper ✓  voice={voice_name} (un-resampled)")
            return True
        except Exception as e:
            print(f"  [TTS] piper voice '{voice_name}' failed: {e}")
            _cleanup(raw_path)
            continue

    _cleanup(raw_path)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. edge-tts  (Microsoft neural TTS — secondary fallback, requires network)
# ─────────────────────────────────────────────────────────────────────────────

EDGE_MALE_VOICES = [
    "en-US-GuyNeural",
    "en-US-ChristopherNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
]

EDGE_FEMALE_VOICES = [
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
]


async def _edge_tts_async(text: str, path: str, voice: str) -> bool:
    """Async helper: generate TTS via edge-tts and save to *path* (any format)."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception as e:
        print(f"  [TTS] edge-tts voice '{voice}' failed: {e}")
        return False


def _edge_tts(text: str, wav_path: str, voice_list: list) -> bool:
    """Try each Microsoft neural voice; convert MP3 → 44.1 kHz mono WAV."""
    try:
        import edge_tts  # noqa: F401 — presence check
    except ImportError:
        print("  [TTS] edge-tts not installed.  Run: pip install edge-tts")
        return False

    tmp_mp3 = wav_path.replace(".wav", "_edge_raw.mp3")

    for voice in voice_list:
        try:
            ok = asyncio.run(_edge_tts_async(text, tmp_mp3, voice))
        except RuntimeError:
            # If there's already a running event loop (e.g. inside Jupyter)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _edge_tts_async(text, tmp_mp3, voice))
                ok = future.result(timeout=30)

        if ok:
            if _to_wav(tmp_mp3, wav_path):
                _cleanup(tmp_mp3)
                print(f"  [TTS] edge-tts ✓  voice={voice}")
                return True

    _cleanup(tmp_mp3)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. pyttsx3  (offline system-voice fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _pyttsx3_wav(text: str, wav_path: str, want_male: bool = True) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 130)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        male_kws = ("male", "david", "mark", "james", "daniel", "alex", "guy",
                    "christopher", "ryan", "william")
        female_kws = ("female", "zira", "hazel", "susan", "samantha", "amy",
                      "jenny", "aria", "sonia", "natasha")
        kws = male_kws if want_male else female_kws
        picked = next(
            (v.id for v in voices
             if any(k in (v.name + v.id).lower() for k in kws)),
            voices[0].id if voices else None,
        )
        if picked:
            engine.setProperty("voice", picked)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        ok = os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0
        if ok:
            print(f"  [TTS] pyttsx3 {'male' if want_male else 'female'} voice ✓")
        return ok
    except Exception as e:
        print(f"  [TTS] pyttsx3 failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Silent WAV  (last resort)
# ─────────────────────────────────────────────────────────────────────────────

def _silent_wav(duration_s: float, path: str) -> bool:
    try:
        rate      = 44100
        n_samples = int(rate * duration_s)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
        print(f"  [TTS] Silent placeholder ({duration_s:.1f} s)")
        return True
    except Exception as e:
        print(f"  [TTS] silent fallback failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_wav(src: str, dst: str) -> bool:
    """Convert any audio to 44.1 kHz mono PCM WAV via ffmpeg."""
    if os.path.abspath(src) == os.path.abspath(dst):
        # Avoid "output file same as input" — round-trip via a temp file.
        tmp = dst + ".resample.wav"
        if not _to_wav(src, tmp):
            return False
        os.replace(tmp, dst)
        return True
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", dst],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def _cleanup(*paths: str):
    for p in paths:
        try:
            if p and os.path.isfile(p):
                os.unlink(p)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def text_to_speech(
    text: str,
    output_path: str | None = None,
    lang: str = "en",
    deep_voice: bool = True,        # kept for API compat; unused
    pitch_rate: float = 0.92,       # kept for API compat; unused
    speaker_gender: str = "male",   # "female" → female voices
) -> str:
    """
    Convert *text* to a natural-sounding WAV audio file.

    Always returns a path to a WAV file.

    Engine priority:
      1. piper-tts  (open-source, offline after first download — **primary**)
      2. edge-tts   (Microsoft neural — needs internet)
      3. pyttsx3    (offline system voices)
      4. Silent WAV (so the pipeline never crashes)
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    # Always work with .wav extension
    base    = os.path.splitext(output_path)[0]
    wav_out = base + ".wav"

    want_female = speaker_gender.strip().lower() in ("female", "f", "woman", "girl")

    # 1. piper-tts (open-source, offline)
    piper_voices = PIPER_FEMALE_VOICES if want_female else PIPER_MALE_VOICES
    if _piper_tts(text, wav_out, piper_voices):
        return wav_out

    # 2. edge-tts (Microsoft neural — fallback)
    print("  [TTS] piper unavailable, trying edge-tts …")
    edge_voices = EDGE_FEMALE_VOICES if want_female else EDGE_MALE_VOICES
    if _edge_tts(text, wav_out, edge_voices):
        return wav_out

    # 3. pyttsx3 (offline system)
    print("  [TTS] edge-tts unavailable, trying pyttsx3 …")
    if _pyttsx3_wav(text, wav_out, want_male=not want_female):
        return wav_out

    # 4. silent — pipeline won't crash but voice will be missing
    print(f"  [TTS] ⚠ ALL engines failed — producing silent audio for: \"{text[:50]}\"")
    duration = max(1.5, len(text.split()) / 120 * 60)
    _silent_wav(duration, wav_out)
    return wav_out


def estimate_duration(text: str, wpm: int = 120) -> float:
    """Rough spoken duration estimate in seconds."""
    return max(1.0, len(text.split()) / wpm * 60)
