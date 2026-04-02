"""
tts.py
Text-to-speech engine.

Primary  : gTTS  (Google TTS – requires internet)
Fallback : pyttsx3 (offline, system voices)
"""

from __future__ import annotations
import os
import tempfile


def _gtts(text: str, path: str, lang: str = "en") -> bool:
    """Try to generate audio with gTTS. Returns True on success."""
    try:
        from gtts import gTTS  # type: ignore
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(path)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        print(f"  [TTS] gTTS failed: {e}")
        return False


def _pyttsx3(text: str, path: str) -> bool:
    """Try to generate audio with pyttsx3. Returns True on success."""
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.save_to_file(text, path)
        engine.runAndWait()
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        print(f"  [TTS] pyttsx3 failed: {e}")
        return False


def _silent_wav(duration_s: float, path: str) -> bool:
    """
    Last-resort fallback: write a silent WAV file of the given duration
    so the pipeline can continue even with no TTS engine.
    """
    try:
        import wave, struct, math
        sample_rate = 22050
        n_samples   = int(sample_rate * duration_s)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Write silence (zeros)
            wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
        print(f"  [TTS] Wrote silent placeholder ({duration_s:.1f}s)")
        return True
    except Exception as e:
        print(f"  [TTS] Silent fallback failed: {e}")
        return False


# ── public API ────────────────────────────────────────────────────────────────

def text_to_speech(
    text: str,
    output_path: str | None = None,
    lang: str = "en",
) -> str:
    """
    Convert *text* to an audio file.

    Returns the path to the generated audio file (mp3 or wav).
    Falls back through: gTTS → pyttsx3 → silent wav.

    Parameters
    ----------
    text        : dialogue line to synthesise
    output_path : where to save the file; a temp file is created if None
    lang        : BCP-47 language code (gTTS only)
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp3")

    wav_path = output_path.replace(".mp3", ".wav")

    if _gtts(text, output_path, lang):
        print(f"  [TTS] gTTS ✓")
        return output_path

    if _pyttsx3(text, wav_path):
        print(f"  [TTS] pyttsx3 ✓")
        return wav_path

    # estimate ~130 wpm
    duration = max(1.5, len(text.split()) / 130 * 60)
    _silent_wav(duration, wav_path)
    return wav_path


def estimate_duration(text: str, wpm: int = 130) -> float:
    """Rough spoken duration in seconds (used when TTS is unavailable)."""
    words = len(text.split())
    return max(1.0, words / wpm * 60)
