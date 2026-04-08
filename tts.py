"""
tts.py  —  Text-to-speech with natural male voices

Engine priority
---------------
1. edge-tts   (Microsoft neural TTS — sounds like a real person, free)
               Voices tried in order:
                 en-US-GuyNeural, en-US-ChristopherNeural, en-GB-RyanNeural,
                 en-AU-WilliamNeural
2. pyttsx3     (offline fallback — system voices, variable quality)
3. Silent WAV  (last resort so the pipeline never crashes)

Why edge-tts?
-------------
Microsoft's neural voices are trained on real human speech and sound natural
out of the box — no pitch manipulation needed, no speed artefacts.
gTTS has been removed completely: it sounds robotic regardless of pitch
processing, and it requires internet access just like edge-tts anyway.

Install
-------
  pip install edge-tts
"""

from __future__ import annotations
import asyncio
import os
import struct
import subprocess
import tempfile
import wave


# ─────────────────────────────────────────────────────────────────────────────
# 1. edge-tts  (Microsoft neural TTS)
# ─────────────────────────────────────────────────────────────────────────────

# Neural voices to try, in preference order — by gender
EDGE_MALE_VOICES = [
    "en-US-GuyNeural",           # confident American broadcaster
    "en-US-ChristopherNeural",   # authoritative American
    "en-GB-RyanNeural",          # British male
    "en-AU-WilliamNeural",       # Australian male
]

EDGE_FEMALE_VOICES = [
    "en-US-JennyNeural",         # warm American female
    "en-US-AriaNeural",          # expressive American female
    "en-GB-SoniaNeural",         # British female
    "en-AU-NatashaNeural",       # Australian female
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
    """
    Try each Microsoft neural voice in *voice_list* in turn.
    Returns True if any succeeds.
    Saves to *wav_path* (edge-tts saves as MP3; we convert to WAV via ffmpeg).
    """
    try:
        import edge_tts  # noqa: F401 — just to check it's installed
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
            # Convert to WAV so the rest of the pipeline always gets a WAV
            if _to_wav(tmp_mp3, wav_path):
                _cleanup(tmp_mp3)
                print(f"  [TTS] edge-tts ✓  voice={voice}")
                return True

    _cleanup(tmp_mp3)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. pyttsx3  (offline fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _pyttsx3_wav(text: str, wav_path: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 130)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        male_kws = ("male", "david", "mark", "james", "daniel", "alex", "guy",
                    "christopher", "ryan", "william")
        male_voice = next(
            (v.id for v in voices
             if any(k in (v.name + v.id).lower() for k in male_kws)),
            voices[0].id if voices else None,
        )
        if male_voice:
            engine.setProperty("voice", male_voice)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        ok = os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0
        if ok:
            print("  [TTS] pyttsx3 male voice ✓")
        return ok
    except Exception as e:
        print(f"  [TTS] pyttsx3 failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Silent WAV  (last resort)
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
    speaker_gender: str = "male",   # "female" → female neural voices
) -> str:
    """
    Convert *text* to a natural WAV audio file.

    Always returns a path to a WAV file.

    Engine priority:
      1. edge-tts  (Microsoft neural voices  — best quality, needs internet)
         Uses EDGE_FEMALE_VOICES when speaker_gender="female",
         EDGE_MALE_VOICES otherwise.
      2. pyttsx3   (system voices — offline, quality varies)
      3. Silent WAV (so the pipeline never crashes)
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    # Always work with .wav extension
    base    = os.path.splitext(output_path)[0]
    wav_out = base + ".wav"

    # Pick voice list based on gender
    voice_list = (
        EDGE_FEMALE_VOICES
        if speaker_gender.strip().lower() in ("female", "f", "woman", "girl")
        else EDGE_MALE_VOICES
    )

    # 1. edge-tts  (Microsoft neural — best quality)
    if _edge_tts(text, wav_out, voice_list):
        return wav_out

    # 2. pyttsx3 (offline)
    print("  [TTS] edge-tts unavailable, trying pyttsx3 …")
    if _pyttsx3_wav(text, wav_out):
        return wav_out

    # 3. silent — pipeline won't crash but voice will be missing
    print(f"  [TTS] ⚠ ALL engines failed — producing silent audio for: \"{text[:50]}\"")
    duration = max(1.5, len(text.split()) / 120 * 60)
    _silent_wav(duration, wav_out)
    return wav_out


def estimate_duration(text: str, wpm: int = 120) -> float:
    """Rough spoken duration estimate in seconds."""
    return max(1.0, len(text.split()) / wpm * 60)
