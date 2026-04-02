"""
video_composer.py

Builds the final MP4 from rendered portrait frames + TTS audio.

Layout (1280 × 720 default)
────────────────────────────────────────────────────────────────
 ┌──────────────────────┬──────────────────────┐
 │  Character A (left)  │  Character B (right) │  85 % of height
 │  640 × 612 px        │  640 × 612 px        │
 ├──────────────────────┴──────────────────────┤
 │  Subtitle bar: "Speaker: dialogue text"     │  15 % of height
 └─────────────────────────────────────────────┘

Active speaker panel has a glowing border; the other character is in a
"listening" pose (mouth closed, slightly dimmer background).
"""

from __future__ import annotations
import os
import textwrap
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from character_renderer import CharacterRenderer


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp_idx(idx: int, length: int) -> int:
    return max(0, min(idx, length - 1))


# ── main class ────────────────────────────────────────────────────────────────

class VideoComposer:
    """
    Parameters
    ----------
    width, height : output video resolution (pixels)
    fps           : output frame rate
    """

    SUBTITLE_RATIO = 0.15   # fraction of height reserved for the subtitle bar

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 24):
        self.width      = width
        self.height     = height
        self.fps        = fps
        self.char_w     = width  // 2
        self.char_h     = int(height * (1 - self.SUBTITLE_RATIO))
        self.sub_y      = self.char_h
        self.sub_h      = height - self.char_h

    # ── public API ────────────────────────────────────────────────────────────

    def create_segment(
        self,
        speaker_idx: int,
        dialogue_text: str,
        speaker_name: str,
        renderers: Sequence[CharacterRenderer],
        mouth_openings: Sequence[float],
        audio_path: str,
    ):
        """
        Create a MoviePy VideoFileClip for one dialogue line.

        Parameters
        ----------
        speaker_idx    : index into *renderers* for the active speaker
        dialogue_text  : the spoken line (shown as subtitle)
        speaker_name   : display name used in the subtitle
        renderers      : list of CharacterRenderer objects (one per character)
        mouth_openings : per-frame mouth openings [0.0 – 1.0] (lip-sync)
        audio_path     : path to the TTS audio file for this line
        """
        try:
            from moviepy.editor import VideoClip, AudioFileClip  # type: ignore  # v1
        except ImportError:
            from moviepy import VideoClip, AudioFileClip  # type: ignore  # v2

        n_frames = len(mouth_openings)
        duration = n_frames / self.fps

        # Pre-render all frames (PIL Images → numpy arrays)
        frames_np: list[np.ndarray] = []
        for fi in range(n_frames):
            mo = float(mouth_openings[fi])
            pil_frame = self._render_frame(
                speaker_idx=speaker_idx,
                renderers=renderers,
                mouth_opening=mo,
                text=dialogue_text,
                speaker_name=speaker_name,
            )
            frames_np.append(np.asarray(pil_frame, dtype=np.uint8))

        def make_frame(t: float) -> np.ndarray:
            fi = _clamp_idx(int(t * self.fps), n_frames)
            return frames_np[fi]

        clip = VideoClip(make_frame, duration=duration)  # type: ignore[arg-type]

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio = AudioFileClip(audio_path)
            # Trim audio to match video duration (MoviePy v1 / v2 compat)
            if audio.duration > duration:
                try:
                    audio = audio.subclipped(0, duration)   # v2
                except AttributeError:
                    audio = audio.subclip(0, duration)      # v1
            # set_audio → with_audio in v2
            try:
                clip = clip.with_audio(audio)               # v2
            except AttributeError:
                clip = clip.set_audio(audio)                # v1

        return clip

    # ── frame rendering ───────────────────────────────────────────────────────

    def _render_frame(
        self,
        speaker_idx: int,
        renderers: Sequence[CharacterRenderer],
        mouth_opening: float,
        text: str,
        speaker_name: str,
    ) -> Image.Image:
        """Compose one full video frame as a PIL Image."""
        frame = Image.new("RGB", (self.width, self.height), (18, 22, 32))

        # ── character panels ──────────────────────────────────────────────────
        for i, renderer in enumerate(renderers):
            is_active   = i == speaker_idx
            char_mouth  = mouth_opening if is_active else 0.0
            char_img    = renderer.render(
                width=self.char_w,
                height=self.char_h,
                mouth_opening=char_mouth,
                is_active=is_active,
            )
            frame.paste(char_img, (i * self.char_w, 0))

        # ── vertical divider ─────────────────────────────────────────────────
        draw = ImageDraw.Draw(frame)
        draw.line(
            [(self.char_w, 0), (self.char_w, self.char_h)],
            fill=(50, 60, 80),
            width=2,
        )

        # ── subtitle bar ──────────────────────────────────────────────────────
        self._draw_subtitle(draw, speaker_name, text)

        return frame

    def _draw_subtitle(
        self, draw: ImageDraw.ImageDraw, speaker_name: str, text: str
    ):
        """Draw the speaker name + wrapped dialogue text in the subtitle strip."""
        y0 = self.sub_y
        # dark background for readability
        draw.rectangle([0, y0, self.width, self.height], fill=(12, 15, 24))
        draw.line([0, y0, self.width, y0], fill=(55, 80, 130), width=2)

        # Speaker name (highlighted colour)
        name_x = 30
        name_y = y0 + 10
        draw.text((name_x, name_y), f"{speaker_name}:", fill=(90, 175, 255))

        # Dialogue (wrapped, white)
        wrap_width = 95           # approximate chars per line at default font size
        wrapped    = textwrap.wrap(text, width=wrap_width)
        text_x = name_x
        text_y = name_y + 22
        line_h = 20

        for line in wrapped[:3]:  # max 3 subtitle lines
            draw.text((text_x, text_y), line, fill=(225, 225, 225))
            text_y += line_h
