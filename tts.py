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

def _pitch_shift_deep(src: str, dst: str, rate: float = 0.85) -> bool:
    """
    Lower the pitch of *src* audio and save to *dst* using ffmpeg.

    Technique
    ---------
    asetrate=44100*rate
        Re-labels the stream as a lower sample rate WITHOUT resampling the
        PCM data.  The audio sounds deeper (fewer Hz perceived per second)
        and naturally a little slower — just like a record played at a lower
        RPM.  No atempo / time-stretch is applied so there is zero speed
        distortion.

    aresample=44100
        Resamples the re-labelled stream back to the standard 44.1 kHz output
        rate so media players handle it correctly.

    Why NO atempo?
    --------------
    Adding `atempo=1/rate` to "restore" the original duration makes ffmpeg
    time-stretch the audio, which is what caused the "voice going 2x fast"
    problem.  A deep voice that is 15–20 % slower sounds completely natural
    (think movie trailers, documentary narrators).  Forcing the duration back
    to match the original creates a rushed, robotic artefact.

    Rate guide
    ----------
    rate=0.92 → ~1.5 semitones lower → warm, authoritative male voice
    rate=0.88 → ~2.5 semitones lower → confident broadcaster         ← default
    rate=0.82 → ~3.5 semitones lower → deep / cinematic narration
    rate=0.75 → ~5   semitones lower → very deep / baritone
    """
    try:
        af = f"asetrate=44100*{rate:.4f},aresample=44100"
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-af", af,
            "-ar", "44100",
            "-q:a", "2",
            dst,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            err = result.stderr[-400:].decode(errors="replace")
            print(f"  [TTS] ffmpeg pitch-shift failed: {err}")
            return False
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except FileNotFoundError:
        print("  [TTS] ffmpeg not found — skipping pitch shift")
        return False
    except Exception as e:
        print(f"  [TTS] pitch-shift error: {e}")
        return False


# ── engines ───────────────────────────────────────────────────────────────────

def _gtts(text: str, path: str, lang: str = "en") -> bool:
    """Generate audio with Google TTS. Returns True on success."""
    try:
        from gtts import gTTS
        # slow=True gives a more deliberate, measured pace which sounds
        # natural when combined with the pitch-down step.
        tts = gTTS(text=text, lang=lang, slow=True)
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
    pitch_rate: float = 0.88,   # 0.88 ≈ 2.5 semitones lower → broadcaster voice
) -> str:
    """
    Convert *text* to a deep, masculine audio file.

    Strategy
    --------
    1. gTTS  →  generate normal audio  (slow=True for measured pace)
    2. ffmpeg  →  pitch-down via asetrate+aresample (NO atempo — avoids
                  the "fast voice" artefact that time-stretching causes)
    3. Fallback: pyttsx3 with male voice
    4. Fallback: silent WAV

    Parameters
    ----------
    text        : dialogue line to synthesise
    output_path : where to save; a temp file is created if None
    lang        : BCP-47 language code (gTTS)
    deep_voice  : apply ffmpeg pitch-shift for deeper voice
    pitch_rate  : 0.92 = warm, 0.88 = broadcaster (default),
                  0.82 = cinematic, 0.75 = very deep/baritone
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
