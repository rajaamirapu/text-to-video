"""
video_composer.py

Side-by-side video composition for real-human talking-head videos.

For each dialogue line we receive:
  • a Wav2Lip output video  (speaker's face, lips in sync with TTS audio)
  • a static face image     (listener's portrait, mouth closed)

These are composited into a single frame:
  ┌────────────────────────┬────────────────────────┐
  │  left character panel  │  right character panel │  85 % of height
  ├────────────────────────┴────────────────────────┤
  │  Subtitle bar: "Speaker: dialogue text …"       │  15 % of height
  └─────────────────────────────────────────────────┘

Active speaker gets a subtle glowing border.
"""

from __future__ import annotations
import os
import textwrap
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw


# ── MoviePy import (v1 / v2 compatible) ─────────────────────────────────────

def _import_moviepy():
    try:
        from moviepy.editor import (       # type: ignore  # v1
            VideoFileClip, ImageClip, CompositeVideoClip,
            AudioFileClip, concatenate_videoclips, ColorClip,
        )
    except ImportError:
        from moviepy import (              # type: ignore  # v2
            VideoFileClip, ImageClip, CompositeVideoClip,
            AudioFileClip, concatenate_videoclips, ColorClip,
        )
    return (
        VideoFileClip, ImageClip, CompositeVideoClip,
        AudioFileClip, concatenate_videoclips, ColorClip,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _resize_image(image_path: str, width: int, height: int) -> np.ndarray:
    """Load and resize a face image to the given size, returning an RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    # Add a subtle vignette border for a 'photo' feel
    draw = ImageDraw.Draw(img)
    for i in range(8):
        alpha = max(0, 80 - i * 10)
        draw.rectangle([i, i, width - 1 - i, height - 1 - i],
                       outline=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)


def _active_glow(img: np.ndarray) -> np.ndarray:
    """Add a blue glow border to indicate the active (speaking) character."""
    out = img.copy()
    for i in range(5):
        brightness = max(0, 160 - i * 30)
        r, g, b = 30, 100, brightness
        out[i, :] = [r, g, b]
        out[-1 - i, :] = [r, g, b]
        out[:, i] = [r, g, b]
        out[:, -1 - i] = [r, g, b]
    return out


def _make_subtitle_image(
    width: int,
    height: int,
    speaker_name: str,
    text: str,
) -> np.ndarray:
    """Render the subtitle bar as a numpy array."""
    img = Image.new("RGB", (width, height), (10, 13, 22))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 0), (width, 0)], fill=(45, 75, 130), width=2)

    # Speaker name
    draw.text((28, 10), f"{speaker_name}:", fill=(85, 170, 255))

    # Wrapped dialogue
    wrapped = textwrap.wrap(text, width=int(width / 8.5))
    ty = 32
    for line in wrapped[:3]:
        draw.text((28, ty), line, fill=(218, 218, 218))
        ty += 21

    return np.asarray(img, dtype=np.uint8)


# ── main class ────────────────────────────────────────────────────────────────

class VideoComposer:
    """
    Parameters
    ----------
    width, height : output video resolution
    fps           : output frame rate (should match Wav2Lip's fps)
    """

    SUBTITLE_RATIO = 0.15

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 25):
        self.width   = width
        self.height  = height
        self.fps     = fps
        self.char_w  = width  // 2
        self.char_h  = int(height * (1 - self.SUBTITLE_RATIO))
        self.sub_h   = height - self.char_h

    # ── public API ────────────────────────────────────────────────────────────

    def create_segment(
        self,
        speaker_idx: int,
        dialogue_text: str,
        speaker_name: str,
        face_image_paths: list[str],   # [left_face.png, right_face.png]
        wav2lip_video_path: str,       # talking face video from Wav2Lip
        audio_path: str,
    ):
        """
        Compose one side-by-side dialogue segment.

        Returns a MoviePy clip (video + audio).
        """
        (VideoFileClip, ImageClip, CompositeVideoClip,
         AudioFileClip, _, ColorClip) = _import_moviepy()

        # ── load Wav2Lip talking video ────────────────────────────────────────
        talking = VideoFileClip(wav2lip_video_path)
        try:
            talking = talking.resized((self.char_w, self.char_h))
        except AttributeError:
            talking = talking.resize((self.char_w, self.char_h))
        duration = talking.duration

        # ── build per-position clips ──────────────────────────────────────────
        clips = []

        for i, face_path in enumerate(face_image_paths[:2]):
            x_pos = i * self.char_w
            is_speaker = (i == speaker_idx)

            if is_speaker:
                # Place the Wav2Lip talking video
                c = talking.with_position((x_pos, 0)) if hasattr(talking, "with_position") \
                    else talking.set_position((x_pos, 0))
            else:
                # Static listening face with optional glow suppressed
                face_arr = _resize_image(face_path, self.char_w, self.char_h)
                c = ImageClip(face_arr, duration=duration)
                c = c.with_position((x_pos, 0)) if hasattr(c, "with_position") \
                    else c.set_position((x_pos, 0))

            clips.append(c)

        # ── glow overlay on speaker panel ─────────────────────────────────────
        # We draw a coloured border frame for the active speaker
        glow_arr = self._make_glow_overlay(speaker_idx, duration)
        glow_clip = ImageClip(glow_arr, duration=duration)
        glow_clip = glow_clip.with_position((0, 0)) if hasattr(glow_clip, "with_position") \
            else glow_clip.set_position((0, 0))
        clips.append(glow_clip)

        # ── subtitle bar ──────────────────────────────────────────────────────
        sub_arr  = _make_subtitle_image(self.width, self.sub_h, speaker_name, dialogue_text)
        sub_clip = ImageClip(sub_arr, duration=duration)
        sub_clip = sub_clip.with_position((0, self.char_h)) if hasattr(sub_clip, "with_position") \
            else sub_clip.set_position((0, self.char_h))
        clips.append(sub_clip)

        # ── dark background ───────────────────────────────────────────────────
        bg = ColorClip(size=(self.width, self.height), color=(14, 17, 26), duration=duration)

        final = CompositeVideoClip([bg] + clips, size=(self.width, self.height))

        # ── attach audio ──────────────────────────────────────────────────────
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio = AudioFileClip(audio_path)
            if audio.duration > duration:
                try:
                    audio = audio.subclipped(0, duration)
                except AttributeError:
                    audio = audio.subclip(0, duration)
            try:
                final = final.with_audio(audio)
            except AttributeError:
                final = final.set_audio(audio)

        return final

    def _make_glow_overlay(self, speaker_idx: int, duration: float) -> np.ndarray:
        """RGBA overlay that draws a glowing border around the speaker's panel."""
        arr = np.zeros((self.char_h, self.width, 4), dtype=np.uint8)
        x0 = speaker_idx * self.char_w
        x1 = x0 + self.char_w

        for i in range(5):
            alpha = max(0, 200 - i * 40)
            # top
            arr[i, x0:x1, :] = [30, 100, 220, alpha]
            # bottom
            arr[self.char_h - 1 - i, x0:x1, :] = [30, 100, 220, alpha]
            # left / right
            arr[:, x0 + i, :]   = [30, 100, 220, alpha]
            arr[:, x1 - 1 - i, :] = [30, 100, 220, alpha]

        return arr

    # ── write helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def concat_and_write(clips: list, output_path: str, fps: int = 25):
        """Concatenate all segment clips and export to *output_path*."""
        (_, _, _, _, concatenate_videoclips, _) = _import_moviepy()
        import inspect, tempfile as _tf

        final  = concatenate_videoclips(clips, method="compose")
        tmp_a  = os.path.join(_tf.gettempdir(), "ttv_final_audio.m4a")

        kw = dict(
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=tmp_a,
            remove_temp=False,
            logger="bar",
        )
        sig = inspect.signature(final.write_videofile)
        if "verbose" in sig.parameters:
            kw["verbose"] = False

        try:
            final.write_videofile(output_path, **kw)
        finally:
            try:
                if os.path.exists(tmp_a):
                    os.unlink(tmp_a)
            except Exception:
                pass

        return final
