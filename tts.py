"""
tts.py
Text-to-speech engine — deep, masculine voice output.

Primary  : gTTS  (Google TTS) + ffmpeg pitch-down for a manly voice
Fallback : pyttsx3 with male voice selection
Last     : silent WAV placeholder
"""

from __future__ import annotations
import os
import subprocess
import tempfile


# ── pitch shift ───────────────────────────────────────────────────────────────

def _pitch_shift_deep(src: str, dst: str, rate: float = 0.82) -> bool:
    """
    Use ffmpeg to lower the pitch of *src* audio and write to *dst*.
    rate < 1.0  →  deeper / slower pitch.  0.82 ≈ a full tone lower.
    Returns True on success.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-af", (
                f"asetrate=44100*{rate},"     # lower sample rate → deeper pitch
                f"aresample=44100,"           # restore to 44.1 kHz
                f"atempo={1/rate:.4f}"        # compensate speed so duration stays same
            ),
            "-q:a", "3",
            dst,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception as e:
        print(f"  [TTS] pitch-shift failed: {e}")
        return False


# ── engines ───────────────────────────────────────────────────────────────────

def _gtts(text: str, path: str, lang: str = "en") -> bool:
    """Generate audio with Google TTS. Returns True on success."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(path)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        print(f"  [TTS] gTTS failed: {e}")
        return False


def _pyttsx3_male(text: str, path: str) -> bool:
    """Generate audio with pyttsx3, forcing a male voice. Returns True on success."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 148)      # slightly slower = more gravitas
        engine.setProperty("volume", 1.0)

        # Pick male voice
        voices = engine.getProperty("voices")
        male_voice = None
        for v in voices:
            name_id = (v.name + v.id).lower()
            if any(w in name_id for w in ("male", "david", "mark", "james", "daniel", "alex")):
                male_voice = v.id
                break
        if male_voice:
            engine.setProperty("voice", male_voice)
        elif voices:
            engine.setProperty("voice", voices[0].id)   # first available

        engine.save_to_file(text, path)
        engine.runAndWait()
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        print(f"  [TTS] pyttsx3 failed: {e}")
        return False


def _silent_wav(duration_s: float, path: str) -> bool:
    """Write a silent WAV so the pipeline can continue."""
    try:
        import wave, struct
        sample_rate = 22050
        n_samples   = int(sample_rate * duration_s)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))
        print(f"  [TTS] Silent placeholder ({duration_s:.1f}s)")
        return True
    except Exception as e:
        print(f"  [TTS] Silent fallback failed: {e}")
        return False


# ── public API ────────────────────────────────────────────────────────────────

def text_to_speech(
    text: str,
    output_path: str | None = None,
    lang: str = "en",
    deep_voice: bool = True,
    pitch_rate: float = 0.82,
) -> str:
    """
    Convert *text* to a deep, masculine audio file.

    Strategy
    --------
    1. gTTS  →  generate normal audio
    2. ffmpeg pitch-shift down by (1 - pitch_rate) to make it deeper
    3. Fallback: pyttsx3 with male voice  (no pitch shift needed)
    4. Fallback: silent WAV

    Parameters
    ----------
    text        : dialogue line to synthesise
    output_path : where to save; a temp file is created if None
    lang        : BCP-47 language code (gTTS)
    deep_voice  : apply ffmpeg pitch-shift for deeper/manly voice
    pitch_rate  : 0.82 ≈ one tone lower; 0.75 for very deep
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp3")

    # Derive wav path alongside the mp3
    base     = output_path.rsplit(".", 1)[0]
    wav_path = base + ".wav"
    tmp_raw  = base + "_raw.mp3"

    # ── 1. gTTS ───────────────────────────────────────────────────────────────
    if _gtts(text, tmp_raw, lang):
        if deep_voice:
            if _pitch_shift_deep(tmp_raw, output_path, rate=pitch_rate):
                print(f"  [TTS] gTTS + deep voice ✓")
                # clean up raw
                try: os.unlink(tmp_raw)
                except: pass
                return output_path
        # If pitch shift failed or disabled, use raw gTTS
        try:
            os.replace(tmp_raw, output_path)
        except Exception:
            pass
        print(f"  [TTS] gTTS ✓")
        return output_path

    # ── 2. pyttsx3 (male voice) ───────────────────────────────────────────────
    if _pyttsx3_male(text, wav_path):
        print(f"  [TTS] pyttsx3 male voice ✓")
        return wav_path

    # ── 3. Silent WAV ─────────────────────────────────────────────────────────
    duration = max(1.5, len(text.split()) / 130 * 60)
    _silent_wav(duration, wav_path)
    return wav_path


def estimate_duration(text: str, wpm: int = 130) -> float:
    """Rough spoken duration in seconds."""
    return max(1.0, len(text.split()) / wpm * 60)
