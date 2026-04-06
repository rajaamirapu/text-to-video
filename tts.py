"""
tts.py
Text-to-speech engine — deep, masculine voice output.

Primary  : gTTS  (Google TTS, slow=True) + ffmpeg pitch-down
Fallback : pyttsx3 with male voice selection
Last     : silent WAV placeholder

Why WAV for pitch-shifting?
---------------------------
MP3 only supports a handful of sample rates (8k / 11k / 16k / 22k / 44.1k Hz).
When we use `asetrate` to reinterpret the stream at a non-standard rate (e.g.
38808 Hz = 44100 × 0.88) and then encode directly to MP3, ffmpeg silently
quantises the rate back to 44100 Hz, which cancels the pitch-down effect and
makes the audio play back at the wrong speed.

Fix: always write the pitch-shifted audio to a PCM WAV file (any sample rate
is valid) and use that WAV through the rest of the pipeline.
"""

from __future__ import annotations
import os
import struct
import subprocess
import tempfile
import wave


# ── pitch shift ───────────────────────────────────────────────────────────────

def _pitch_shift_deep(src: str, dst_wav: str, rate: float = 0.88) -> bool:
    """
    Lower the pitch of *src* audio and write a PCM-WAV to *dst_wav*.

    Technique
    ---------
    Step 1 — asetrate=44100*rate
        Re-labels the stream as a lower sample rate WITHOUT resampling the
        PCM data.  The same audio samples now represent fewer Hz per second,
        so the voice sounds deeper and a little slower (natural for deep voices).

    Step 2 — aresample=44100
        Resamples back to standard 44.1 kHz so media players handle it
        correctly.  The pitch-down is preserved; only the output clock is
        normalised.

    Output format — WAV (pcm_s16le)
        WAV accepts any integer sample rate, so there is no quantisation
        artefact.  MP3 would silently snap the rate back to a permitted value,
        undoing the pitch-down and causing playback speed errors.

    Rate guide
    ----------
    rate=0.92 → ~1.5 semitones lower → warm, authoritative
    rate=0.88 → ~2.5 semitones lower → confident broadcaster (default)
    rate=0.82 → ~3.5 semitones lower → deep / cinematic
    rate=0.75 → ~5   semitones lower → very deep / baritone
    """
    try:
        af = f"asetrate=44100*{rate:.6f},aresample=44100"
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-af", af,
            "-ar", "44100",
            "-c:a", "pcm_s16le",   # uncompressed WAV — no sample-rate restriction
            dst_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            err = result.stderr[-500:].decode(errors="replace")
            print(f"  [TTS] ffmpeg pitch-shift failed:\n{err}")
            return False
        ok = os.path.isfile(dst_wav) and os.path.getsize(dst_wav) > 0
        if ok:
            print(f"  [TTS] pitch-shift ✓  rate={rate}  → {os.path.basename(dst_wav)}")
        return ok
    except FileNotFoundError:
        print("  [TTS] ffmpeg not found — skipping pitch shift")
        return False
    except Exception as e:
        print(f"  [TTS] pitch-shift error: {e}")
        return False


# ── engines ───────────────────────────────────────────────────────────────────

def _gtts(text: str, path: str, lang: str = "en") -> bool:
    """
    Generate TTS audio with Google TTS.

    slow=True gives a measured, deliberate pace that sounds natural after
    the pitch-down step.  The combined effect (slow pace + lower pitch) is
    a confident, masculine broadcaster voice.
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=True)
        tts.save(path)
        ok = os.path.isfile(path) and os.path.getsize(path) > 0
        if ok:
            print(f"  [TTS] gTTS generated → {os.path.basename(path)}")
        return ok
    except Exception as e:
        print(f"  [TTS] gTTS failed: {e}")
        return False


def _pyttsx3_male(text: str, path: str) -> bool:
    """Generate audio with pyttsx3, forcing a male voice."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 135)    # words per minute — slower = deeper feel
        engine.setProperty("volume", 1.0)

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
            engine.setProperty("voice", voices[0].id)

        engine.save_to_file(text, path)
        engine.runAndWait()
        ok = os.path.isfile(path) and os.path.getsize(path) > 0
        if ok:
            print(f"  [TTS] pyttsx3 male voice ✓")
        return ok
    except Exception as e:
        print(f"  [TTS] pyttsx3 failed: {e}")
        return False


def _silent_wav(duration_s: float, path: str) -> bool:
    """Write a silent WAV so the pipeline can continue."""
    try:
        sample_rate = 44100
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
    pitch_rate: float = 0.88,
) -> str:
    """
    Convert *text* to a deep, masculine WAV audio file.

    Parameters
    ----------
    text        : dialogue line to synthesise
    output_path : desired output path (extension ignored — always returns WAV)
    lang        : BCP-47 language code (gTTS)
    deep_voice  : apply ffmpeg pitch-shift for a deeper voice
    pitch_rate  : 0.92 = warm/natural, 0.88 = broadcaster (default),
                  0.82 = cinematic deep, 0.75 = baritone

    Returns
    -------
    Path to the generated WAV file (always .wav regardless of output_path).

    Pipeline
    --------
    1. gTTS (slow=True) → raw MP3
    2. ffmpeg asetrate+aresample → pitch-shifted PCM WAV  ← no speed artefact
    3. Fallback: pyttsx3 male voice → WAV
    4. Fallback: silent WAV
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    # Always work in WAV to avoid MP3 sample-rate quantisation
    base    = os.path.splitext(output_path)[0]
    wav_out = base + ".wav"
    tmp_raw = base + "_raw_gtts.mp3"   # intermediate gTTS MP3

    # ── 1. gTTS → raw MP3 → pitch-shifted WAV ────────────────────────────────
    if _gtts(text, tmp_raw, lang):
        if deep_voice:
            if _pitch_shift_deep(tmp_raw, wav_out, rate=pitch_rate):
                _cleanup(tmp_raw)
                return wav_out
        # Pitch shift failed (e.g. no ffmpeg) — convert gTTS MP3 → WAV as-is
        if _convert_to_wav(tmp_raw, wav_out):
            _cleanup(tmp_raw)
            return wav_out
        # Last resort: return the MP3 directly
        _cleanup(wav_out)
        return tmp_raw

    # ── 2. pyttsx3 male voice (writes WAV natively) ───────────────────────────
    if _pyttsx3_male(text, wav_out):
        return wav_out

    # ── 3. Silent WAV ─────────────────────────────────────────────────────────
    duration = max(1.5, len(text.split()) / 120 * 60)
    _silent_wav(duration, wav_out)
    return wav_out


# ── internal helpers ──────────────────────────────────────────────────────────

def _convert_to_wav(src: str, dst: str) -> bool:
    """Convert any audio file to 44.1kHz mono PCM WAV using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-ar", "44100", "-ac", "1",
            "-c:a", "pcm_s16le", dst,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
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


def estimate_duration(text: str, wpm: int = 120) -> float:
    """Rough spoken duration in seconds (accounting for slow=True pace)."""
    return max(1.0, len(text.split()) / wpm * 60)
