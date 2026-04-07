"""
tts.py  —  Text-to-speech with natural male voices

Engine priority
---------------
1. edge-tts   (Microsoft neural TTS — sounds like a real person, free)
               Voices tried in order:
                 en-US-GuyNeural, en-US-ChristopherNeural, en-GB-RyanNeural
2. gTTS        (Google TTS fallback — decent quality, needs internet)
               Slight pitch-down to 44.1 kHz WAV via ffmpeg (no atempo)
3. pyttsx3     (offline fallback — system voices, variable quality)
4. Silent WAV  (last resort so the pipeline never crashes)

Why edge-tts?
-------------
gTTS is a robot.  No amount of pitch-shifting fixes its monotone cadence.
Microsoft's neural voices (edge-tts) are trained on real human speech and
sound natural out of the box — no pitch manipulation needed, no speed
artefacts.

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

# Male neural voices to try, in preference order
EDGE_MALE_VOICES = [
    "en-US-GuyNeural",           # confident American broadcaster
    "en-US-ChristopherNeural",   # authoritative American
    "en-GB-RyanNeural",          # British male
    "en-AU-WilliamNeural",       # Australian male
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


def _edge_tts(text: str, wav_path: str) -> bool:
    """
    Try each Microsoft neural voice in turn.  Returns True if any succeeds.
    Saves to *wav_path* (edge-tts saves as MP3; we convert to WAV via ffmpeg).
    """
    try:
        import edge_tts  # noqa: F401 — just to check it's installed
    except ImportError:
        print("  [TTS] edge-tts not installed.  Run: pip install edge-tts")
        return False

    tmp_mp3 = wav_path.replace(".wav", "_edge_raw.mp3")

    for voice in EDGE_MALE_VOICES:
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
# 2. gTTS + optional mild pitch-down  (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _gtts_wav(text: str, wav_path: str, lang: str = "en",
              pitch_rate: float = 0.92) -> bool:
    """
    gTTS → raw MP3 → optional pitch-down → WAV.

    pitch_rate=0.92 ≈ 1.5 semitones lower (just enough to sound male without
    introducing speed artefacts).  Uses asetrate+aresample only — NO atempo.
    """
    try:
        from gtts import gTTS
    except ImportError:
        return False

    tmp_raw = wav_path.replace(".wav", "_gtts_raw.mp3")
    try:
        tts = gTTS(text=text, lang=lang, slow=True)
        tts.save(tmp_raw)
        if not (os.path.isfile(tmp_raw) and os.path.getsize(tmp_raw) > 0):
            return False
        print(f"  [TTS] gTTS raw ✓")
    except Exception as e:
        print(f"  [TTS] gTTS failed: {e}")
        _cleanup(tmp_raw)
        return False

    # Pitch-down via asetrate+aresample, output straight to WAV
    # (WAV has no sample-rate restrictions, unlike MP3)
    af = f"asetrate=44100*{pitch_rate:.6f},aresample=44100"
    cmd = [
        "ffmpeg", "-y", "-i", tmp_raw,
        "-af", af,
        "-ar", "44100", "-ac", "1",
        "-c:a", "pcm_s16le",
        wav_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        _cleanup(tmp_raw)
        if r.returncode == 0 and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            print(f"  [TTS] gTTS + pitch-down ✓  rate={pitch_rate}")
            return True
        # ffmpeg failed — try plain conversion without pitch shift
        print("  [TTS] pitch-down failed, converting gTTS directly to WAV …")
        from gtts import gTTS as _gTTS
        tts2 = _gTTS(text=text, lang=lang, slow=True)
        tmp2 = wav_path.replace(".wav", "_gtts2_raw.mp3")
        tts2.save(tmp2)
        ok = _to_wav(tmp2, wav_path)
        _cleanup(tmp2)
        return ok
    except Exception as e:
        print(f"  [TTS] gTTS WAV conversion failed: {e}")
        _cleanup(tmp_raw)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. pyttsx3  (offline fallback)
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
    deep_voice: bool = True,        # kept for API compat; edge-tts ignores it
    pitch_rate: float = 0.92,       # only used by gTTS fallback
) -> str:
    """
    Convert *text* to a natural male WAV audio file.

    Always returns a path to a WAV file.

    Engine priority:
      1. edge-tts  (Microsoft neural voices  — best quality, needs internet)
      2. gTTS      (Google TTS + mild pitch-down — decent, needs internet)
      3. pyttsx3   (system voices — offline, quality varies)
      4. Silent WAV (so the pipeline never crashes)
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    # Always work with .wav extension
    base    = os.path.splitext(output_path)[0]
    wav_out = base + ".wav"

    # 1. edge-tts  (Microsoft neural — best quality)
    if _edge_tts(text, wav_out):
        return wav_out

    # 2. gTTS + pitch-down
    print("  [TTS] edge-tts unavailable, trying gTTS …")
    if _gtts_wav(text, wav_out, lang=lang, pitch_rate=pitch_rate):
        return wav_out

    # 3. pyttsx3
    print("  [TTS] gTTS unavailable, trying pyttsx3 …")
    if _pyttsx3_wav(text, wav_out):
        return wav_out

    # 4. silent — pipeline won't crash but voice will be missing
    print(f"  [TTS] ⚠ ALL engines failed — producing silent audio for: \"{text[:50]}\"")
    duration = max(1.5, len(text.split()) / 120 * 60)
    _silent_wav(duration, wav_out)
    return wav_out


def estimate_duration(text: str, wpm: int = 120) -> float:
    """Rough spoken duration estimate in seconds."""
    return max(1.0, len(text.split()) / wpm * 60)
