"""
video_composer.py

Composes a two-person conversation inside a room scene.

Layout  (1280×720 default)
──────────────────────────
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   ROOM BACKGROUND  (warm office / studio)           │
  │                                                      │
  │   [left portrait]              [right portrait]      │  85 % height
  │   speaker = Wav2Lip video      listener = still img  │
  │                                                      │
  ├──────────────────────────────────────────────────────┤
  │   Subtitle bar  "Speaker: dialogue text …"           │  15 % height
  └──────────────────────────────────────────────────────┘

Everything is composited as numpy ImageClips — no ColorClip, so it
works with every MoviePy version (1.x and 2.x).
"""

from __future__ import annotations
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ── MoviePy import (v1 / v2 compatible) ─────────────────────────────────────

def _mp():
    try:
        from moviepy.editor import (
            VideoFileClip, ImageClip, CompositeVideoClip,
            AudioFileClip, concatenate_videoclips,
        )
    except ImportError:
        from moviepy import (
            VideoFileClip, ImageClip, CompositeVideoClip,
            AudioFileClip, concatenate_videoclips,
        )
    return VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips


def _pos(clip, x, y):
    """Set clip position — compatible with MoviePy v1 and v2."""
    if hasattr(clip, "with_position"):
        return clip.with_position((x, y))
    return clip.set_position((x, y))


# ─────────────────────────────────────────────────────────────────────────────
# Room background
# ─────────────────────────────────────────────────────────────────────────────

def _make_room_background(width: int, height: int) -> Image.Image:
    """Render a warm office room background with PIL."""
    img  = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    wall_h = int(height * 0.62)

    # Wall — cream-beige gradient
    for y in range(wall_h):
        t = y / max(1, wall_h)
        draw.line([(0, y), (width, y)], fill=(
            int(215 - t * 18),
            int(205 - t * 14),
            int(190 - t * 12),
        ))

    # Window + light bloom
    win_cx, win_cy = width // 2, int(wall_h * 0.38)
    win_w, win_h   = 260, 190
    wx0, wy0 = win_cx - win_w // 2, win_cy - win_h // 2
    wx1, wy1 = win_cx + win_w // 2, win_cy + win_h // 2

    for r in range(200, 0, -20):
        draw.ellipse([win_cx - r * 2, win_cy - r, win_cx + r * 2, win_cy + r],
                     fill=(min(255, 240 + r // 12), min(255, 245 + r // 14), 255))

    for y in range(wy0, wy1):
        t = (y - wy0) / max(1, wy1 - wy0)
        draw.line([(wx0, y), (wx1, y)], fill=(
            int(175 + t * 35), int(205 + t * 20), 240))

    frame_col = (155, 138, 110)
    draw.rectangle([wx0, wy0, wx1, wy1], outline=frame_col, width=7)
    draw.line([(win_cx, wy0), (win_cx, wy1)], fill=frame_col, width=5)
    draw.line([(wx0, win_cy), (wx1, win_cy)], fill=frame_col, width=5)

    # Floor — hardwood gradient
    for y in range(wall_h, height):
        t = (y - wall_h) / max(1, height - wall_h)
        draw.line([(0, y), (width, y)], fill=(
            int(148 - t * 30), int(108 - t * 22), int(68 - t * 16)))

    for i in range(7):
        y = wall_h + i * max(1, (height - wall_h) // 6)
        if y < height:
            draw.line([(0, y), (width, y)], fill=(120, 88, 55), width=1)

    # Baseboard
    draw.rectangle([0, wall_h - 12, width, wall_h + 4], fill=(185, 173, 152))

    # Conference table
    table_top = int(height * 0.70)
    tw = int(width * 0.72)
    tcx = width // 2
    draw.polygon([
        (tcx - tw // 2,      table_top),
        (tcx + tw // 2,      table_top),
        (tcx + tw // 2 + 80, height + 2),
        (tcx - tw // 2 - 80, height + 2),
    ], fill=(125, 88, 52))
    draw.line([(tcx - tw // 2, table_top), (tcx + tw // 2, table_top)],
              fill=(165, 125, 82), width=5)

    # Vignette
    vig = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vd  = ImageDraw.Draw(vig)
    for i in range(30):
        alpha = int(90 * (1 - i / 30) ** 2)
        pad   = int(i * min(width, height) * 0.018)
        vd.rectangle([pad, pad, width - pad, height - pad],
                     outline=(0, 0, 0, alpha), width=2)
    base = img.convert("RGBA")
    base.alpha_composite(vig)
    return base.convert("RGB")


_BG_CACHE: dict = {}


def _get_room_bg(width: int, height: int, room_bg_path: str | None) -> np.ndarray:
    key = (width, height, room_bg_path)
    if key not in _BG_CACHE:
        if room_bg_path and os.path.isfile(room_bg_path):
            img = Image.open(room_bg_path).convert("RGB").resize((width, height), Image.LANCZOS)
        else:
            img = _make_room_background(width, height)
        _BG_CACHE[key] = np.asarray(img, dtype=np.uint8)
    return _BG_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# Face helpers
# ─────────────────────────────────────────────────────────────────────────────

def _oval_crop(img: Image.Image) -> Image.Image:
    """Soft oval mask so the portrait blends into the room."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    md   = ImageDraw.Draw(mask)
    px, py = int(w * 0.06), int(h * 0.04)
    md.ellipse([px, py, w - px, h - py], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, int(w * 0.025))))
    out  = img.convert("RGBA")
    out.putalpha(mask)
    return out


def _load_face(path: str, fw: int, fh: int) -> Image.Image:
    img = Image.open(path).convert("RGB").resize((fw, fh), Image.LANCZOS)
    return _oval_crop(img)


def _glow_overlay(fw: int, fh: int) -> np.ndarray:
    """Golden speaker glow ring as RGBA numpy array."""
    glow = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    for i in range(10):
        alpha = max(0, 180 - i * 18)
        p     = i * 2
        gd.ellipse([p, p, fw - p, fh - p], outline=(255, 200, 80, alpha), width=3)
    glow = glow.filter(ImageFilter.GaussianBlur(4))
    return np.asarray(glow, dtype=np.uint8)


def _subtitle_img(width: int, height: int, speaker: str, text: str) -> np.ndarray:
    img  = Image.new("RGB", (width, height), (12, 10, 18))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 3], fill=(200, 160, 40))
    draw.text((30, 8), f"{speaker}:", fill=(255, 200, 60))
    ty = 30
    for line in textwrap.wrap(text, width=int(width / 8.5))[:3]:
        draw.text((30, ty), line, fill=(230, 228, 225))
        ty += 22
    return np.asarray(img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# VideoComposer
# ─────────────────────────────────────────────────────────────────────────────

class VideoComposer:

    SUBTITLE_RATIO = 0.15

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 25,
        room_bg_path: str | None = None,
    ):
        self.width        = width
        self.height       = height
        self.fps          = fps
        self.room_bg_path = room_bg_path
        self.char_h       = int(height * (1 - self.SUBTITLE_RATIO))
        self.sub_h        = height - self.char_h

        # Portrait dimensions (~38 % of width, 4:5 aspect)
        self.face_w = int(width * 0.38)
        self.face_h = int(self.face_w * 1.20)

        # Face centre positions in the room area
        half = width // 2
        self.face_centres = [
            (half // 2,        int(self.char_h * 0.54)),   # left
            (half + half // 2, int(self.char_h * 0.54)),   # right
        ]

    def create_segment(
        self,
        speaker_idx: int,
        dialogue_text: str,
        speaker_name: str,
        face_image_paths: list[str],
        wav2lip_video_path: str,
        audio_path: str,
    ):
        VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip, _ = _mp()

        # Load Wav2Lip talking video
        talking  = VideoFileClip(wav2lip_video_path)
        duration = talking.duration

        listener_idx = 1 - speaker_idx

        # ── Build full background frame as numpy (room + listener face) ────────
        # Room background for the top char_h rows
        room_arr = _get_room_bg(self.width, self.char_h, self.room_bg_path)
        room_img = Image.fromarray(room_arr).convert("RGBA")

        # Paste listener (static, oval-cropped)
        l_cx, l_cy = self.face_centres[listener_idx]
        l_face     = _load_face(face_image_paths[listener_idx], self.face_w, self.face_h)
        room_img.alpha_composite(l_face, (l_cx - self.face_w // 2, l_cy - self.face_h // 2))

        # Full frame: room area on top, dark subtitle area below
        full = Image.new("RGB", (self.width, self.height), (12, 10, 18))
        full.paste(room_img.convert("RGB"), (0, 0))

        # ── Background ImageClip (static, whole frame) ─────────────────────────
        bg_clip = ImageClip(np.asarray(full, dtype=np.uint8), duration=duration)
        bg_clip = _pos(bg_clip, 0, 0)

        # ── Speaker Wav2Lip clip (positioned in room) ─────────────────────────
        s_cx, s_cy = self.face_centres[speaker_idx]
        sx = s_cx - self.face_w // 2
        sy = s_cy - self.face_h // 2

        try:
            spk_clip = talking.resized((self.face_w, self.face_h))
        except AttributeError:
            spk_clip = talking.resize((self.face_w, self.face_h))
        spk_clip = _pos(spk_clip, sx, sy)

        # ── Speaker glow ring ─────────────────────────────────────────────────
        glow_clip = ImageClip(_glow_overlay(self.face_w, self.face_h), duration=duration)
        glow_clip = _pos(glow_clip, sx, sy)

        # ── Subtitle bar ──────────────────────────────────────────────────────
        sub_clip = ImageClip(
            _subtitle_img(self.width, self.sub_h, speaker_name, dialogue_text),
            duration=duration,
        )
        sub_clip = _pos(sub_clip, 0, self.char_h)

        # ── Composite ─────────────────────────────────────────────────────────
        final = CompositeVideoClip(
            [bg_clip, spk_clip, glow_clip, sub_clip],
            size=(self.width, self.height),
        )

        # ── Audio ─────────────────────────────────────────────────────────────
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

    @staticmethod
    def concat_and_write(clips: list, output_path: str, fps: int = 25):
        _, _, _, _, concatenate_videoclips = _mp()
        import inspect, tempfile as _tf

        final = concatenate_videoclips(clips, method="compose")
        tmp_a = os.path.join(_tf.gettempdir(), "ttv_final_audio.m4a")
        kw    = dict(fps=fps, codec="libx264", audio_codec="aac",
                     temp_audiofile=tmp_a, remove_temp=False, logger="bar")
        if "verbose" in inspect.signature(final.write_videofile).parameters:
            kw["verbose"] = False
        try:
            final.write_videofile(output_path, **kw)
        finally:
            try:
                os.path.exists(tmp_a) and os.unlink(tmp_a)
            except Exception:
                pass
        return final
