"""
video_composer.py

Composes a two-person conversation scene inside a room.

Layout (1280×720 default)
--------------------------
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  ROOM BACKGROUND  (office / studio setting)        │
  │                                                     │
  │  [left portrait]              [right portrait]      │  ~85 % height
  │  speaker = Wav2Lip video      listener = still img  │
  │                                                     │
  ├─────────────────────────────────────────────────────┤
  │   Subtitle bar  "Speaker: dialogue text …"          │  ~15 % height
  └─────────────────────────────────────────────────────┘

The room background is generated once with PIL (warm office look) and
cached.  Face portraits are placed as seated busts inside the room.
Active speaker gets a soft warm rim-light overlay to indicate they're
talking.
"""

from __future__ import annotations
import os
import textwrap
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ── MoviePy import (v1 / v2 compatible) ─────────────────────────────────────

def _import_moviepy():
    try:
        from moviepy.editor import (       # v1
            VideoFileClip, ImageClip, CompositeVideoClip,
            AudioFileClip, concatenate_videoclips, ColorClip,
        )
    except ImportError:
        from moviepy import (              # v2
            VideoFileClip, ImageClip, CompositeVideoClip,
            AudioFileClip, concatenate_videoclips, ColorClip,
        )
    return VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips, ColorClip


# ─────────────────────────────────────────────────────────────────────────────
# Room background (PIL — no internet required)
# ─────────────────────────────────────────────────────────────────────────────

def _make_room_background(width: int, height: int) -> Image.Image:
    """
    Render a warm office/study room background using PIL.
    Returned image is (width × height) RGB.
    """
    img  = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    wall_h = int(height * 0.62)

    # ── wall: warm cream-beige gradient ──────────────────────────────────────
    for y in range(wall_h):
        t   = y / wall_h
        col = (
            int(215 - t * 18),
            int(205 - t * 14),
            int(190 - t * 12),
        )
        draw.line([(0, y), (width, y)], fill=col)

    # ── window / light bloom on back wall ────────────────────────────────────
    win_cx = width // 2
    win_cy = int(wall_h * 0.38)
    win_w, win_h2 = 260, 190
    wx0, wy0 = win_cx - win_w // 2, win_cy - win_h2 // 2
    wx1, wy1 = win_cx + win_w // 2, win_cy + win_h2 // 2

    # ambient glow behind window
    for r in range(220, 0, -20):
        alpha = max(0, 40 - r // 8)
        col   = (min(255, 235 + r // 10), min(255, 240 + r // 12), 255)
        draw.ellipse([win_cx - r * 2, win_cy - r, win_cx + r * 2, win_cy + r],
                     fill=col)

    # glass pane
    draw.rectangle([wx0, wy0, wx1, wy1], fill=(210, 228, 248))
    # sky visible through glass
    for y in range(wy0, wy1):
        t   = (y - wy0) / max(1, wy1 - wy0)
        col = (int(175 + t * 35), int(205 + t * 20), int(240))
        draw.line([(wx0, y), (wx1, y)], fill=col)

    # window frame
    frame_col = (155, 138, 110)
    draw.rectangle([wx0, wy0, wx1, wy1], outline=frame_col, width=7)
    draw.line([(win_cx, wy0), (win_cx, wy1)], fill=frame_col, width=5)
    draw.line([(wx0, win_cy), (wx1, win_cy)], fill=frame_col, width=5)

    # ── floor: hardwood gradient ──────────────────────────────────────────────
    for y in range(wall_h, height):
        t   = (y - wall_h) / max(1, height - wall_h)
        col = (
            int(148 - t * 30),
            int(108 - t * 22),
            int(68  - t * 16),
        )
        draw.line([(0, y), (width, y)], fill=col)

    # floor plank lines
    plank_col = (120, 88, 55)
    plank_spacing = int((height - wall_h) / 6)
    for i in range(7):
        y = wall_h + i * plank_spacing
        if y < height:
            draw.line([(0, y), (width, y)], fill=plank_col, width=1)

    # ── baseboard ─────────────────────────────────────────────────────────────
    draw.rectangle([0, wall_h - 12, width, wall_h + 4], fill=(185, 173, 152))
    draw.line([(0, wall_h - 12), (width, wall_h - 12)],
              fill=(200, 190, 170), width=2)

    # ── conference table ──────────────────────────────────────────────────────
    table_top_y = int(height * 0.70)
    table_w     = int(width * 0.72)
    table_cx    = width // 2
    table_pts   = [
        (table_cx - table_w // 2,      table_top_y),
        (table_cx + table_w // 2,      table_top_y),
        (table_cx + table_w // 2 + 80, height + 2),
        (table_cx - table_w // 2 - 80, height + 2),
    ]
    draw.polygon(table_pts, fill=(125, 88, 52))
    # table top edge highlight
    draw.line(
        [(table_cx - table_w // 2, table_top_y),
         (table_cx + table_w // 2, table_top_y)],
        fill=(165, 125, 82), width=5,
    )
    # subtle table reflections
    for i in range(3):
        x  = table_cx - 80 + i * 80
        ry = table_top_y + 8
        draw.line([(x, ry), (x + 40, ry + 50)], fill=(140, 100, 62), width=1)

    # ── ambient room shadow / vignette ────────────────────────────────────────
    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    steps = 30
    for i in range(steps):
        t     = i / steps
        alpha = int(90 * (1 - t) ** 2)
        pad   = int(i * min(width, height) * 0.018)
        vd.rectangle([pad, pad, width - pad, height - pad],
                     outline=(0, 0, 0, alpha), width=2)
    img_rgba = img.convert("RGBA")
    img_rgba.alpha_composite(vignette)
    img = img_rgba.convert("RGB")

    return img


_ROOM_BG_CACHE: Image.Image | None = None


def _get_room_background(width: int, height: int, room_bg_path: str | None = None) -> np.ndarray:
    """Return the room background as an RGB numpy array."""
    global _ROOM_BG_CACHE
    if room_bg_path and os.path.isfile(room_bg_path):
        img = Image.open(room_bg_path).convert("RGB").resize((width, height), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)
    if _ROOM_BG_CACHE is None or _ROOM_BG_CACHE.size != (width, height):
        _ROOM_BG_CACHE = _make_room_background(width, height)
    return np.asarray(_ROOM_BG_CACHE, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Face portrait helpers
# ─────────────────────────────────────────────────────────────────────────────

def _oval_crop(img: Image.Image) -> Image.Image:
    """
    Apply a soft oval mask so portrait blends into the room naturally.
    Returns an RGBA image.
    """
    w, h   = img.size
    mask   = Image.new("L", (w, h), 0)
    md     = ImageDraw.Draw(mask)
    pad_x, pad_y = int(w * 0.06), int(h * 0.04)
    md.ellipse([pad_x, pad_y, w - pad_x, h - pad_y], fill=255)
    mask   = mask.filter(ImageFilter.GaussianBlur(int(w * 0.025)))
    out    = img.convert("RGBA")
    out.putalpha(mask)
    return out


def _load_face(path: str, face_w: int, face_h: int, oval: bool = True) -> Image.Image:
    """Load, resize, and optionally oval-crop a face image."""
    img = Image.open(path).convert("RGB").resize((face_w, face_h), Image.LANCZOS)
    return _oval_crop(img) if oval else img.convert("RGBA")


def _speaker_glow_overlay(face_w: int, face_h: int) -> Image.Image:
    """
    Warm golden rim-light ring that appears around the active speaker's portrait.
    """
    glow = Image.new("RGBA", (face_w, face_h), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    for i in range(10):
        alpha = max(0, 180 - i * 18)
        pad   = i * 2
        gd.ellipse([pad, pad, face_w - pad, face_h - pad],
                   outline=(255, 200, 80, alpha), width=3)
    return glow.filter(ImageFilter.GaussianBlur(4))


def _make_subtitle_image(width: int, height: int, speaker_name: str, text: str) -> np.ndarray:
    """Render the subtitle bar."""
    img  = Image.new("RGB", (width, height), (12, 10, 18))
    draw = ImageDraw.Draw(img)
    # accent line
    draw.rectangle([0, 0, width, 3], fill=(200, 160, 40))
    # speaker label
    draw.text((30, 8), f"{speaker_name}:", fill=(255, 200, 60))
    # dialogue
    wrapped = textwrap.wrap(text, width=int(width / 8.5))
    ty = 30
    for line in wrapped[:3]:
        draw.text((30, ty), line, fill=(230, 228, 225))
        ty += 22
    return np.asarray(img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# VideoComposer
# ─────────────────────────────────────────────────────────────────────────────

class VideoComposer:
    """
    Parameters
    ----------
    width, height  : output resolution (default 1280×720)
    fps            : frame rate (match Wav2Lip)
    room_bg_path   : optional path to a custom room background image
    """

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

        # Portrait size — each face occupies ~38 % of total width, aspect ~4:5
        self.face_w = int(width * 0.38)
        self.face_h = int(self.face_w * 1.20)

        # Face anchor centres (x, y) — lower-centre of each half
        half = width // 2
        self.face_centres = [
            (half // 2,          int(self.char_h * 0.54)),   # left character
            (half + half // 2,   int(self.char_h * 0.54)),   # right character
        ]

    # ── public API ────────────────────────────────────────────────────────────

    def create_segment(
        self,
        speaker_idx: int,
        dialogue_text: str,
        speaker_name: str,
        face_image_paths: list[str],
        wav2lip_video_path: str,
        audio_path: str,
    ):
        """
        Compose one room-scene dialogue segment.
        Returns a MoviePy clip (video + audio).
        """
        (VideoFileClip, ImageClip, CompositeVideoClip,
         AudioFileClip, _, ColorClip) = _import_moviepy()

        # ── load Wav2Lip talking video ────────────────────────────────────────
        talking  = VideoFileClip(wav2lip_video_path)
        duration = talking.duration

        # ── pre-render room background ────────────────────────────────────────
        room_arr = _get_room_background(self.width, self.char_h, self.room_bg_path)
        room_img = Image.fromarray(room_arr)

        # ── build static frame: room + listener face ──────────────────────────
        listener_idx = 1 - speaker_idx

        # Start with room background
        base = room_img.copy().convert("RGBA")

        # Paste listener (static)
        l_cx, l_cy = self.face_centres[listener_idx]
        l_face     = _load_face(face_image_paths[listener_idx], self.face_w, self.face_h)
        lx = l_cx - self.face_w // 2
        ly = l_cy - self.face_h // 2
        base.alpha_composite(l_face, (lx, ly))

        # Convert final static layer to RGB numpy for ImageClip
        static_arr = np.asarray(base.convert("RGB"), dtype=np.uint8)
        static_clip = ImageClip(static_arr, duration=duration)
        static_clip = (static_clip.with_position((0, 0))
                       if hasattr(static_clip, "with_position")
                       else static_clip.set_position((0, 0)))

        # ── position Wav2Lip speaker video in scene ───────────────────────────
        s_cx, s_cy = self.face_centres[speaker_idx]
        sx = s_cx - self.face_w // 2
        sy = s_cy - self.face_h // 2

        # Resize Wav2Lip output to face_w × face_h
        try:
            spk_clip = talking.resized((self.face_w, self.face_h))
        except AttributeError:
            spk_clip = talking.resize((self.face_w, self.face_h))

        spk_clip = (spk_clip.with_position((sx, sy))
                    if hasattr(spk_clip, "with_position")
                    else spk_clip.set_position((sx, sy)))

        # ── speaker glow ring ─────────────────────────────────────────────────
        glow_img  = _speaker_glow_overlay(self.face_w, self.face_h)
        glow_arr  = np.asarray(glow_img, dtype=np.uint8)
        glow_clip = ImageClip(glow_arr, duration=duration)
        glow_clip = (glow_clip.with_position((sx, sy))
                     if hasattr(glow_clip, "with_position")
                     else glow_clip.set_position((sx, sy)))

        # ── subtitle bar ──────────────────────────────────────────────────────
        sub_arr  = _make_subtitle_image(self.width, self.sub_h, speaker_name, dialogue_text)
        sub_clip = ImageClip(sub_arr, duration=duration)
        sub_clip = (sub_clip.with_position((0, self.char_h))
                    if hasattr(sub_clip, "with_position")
                    else sub_clip.set_position((0, self.char_h)))

        # ── composite ─────────────────────────────────────────────────────────
        bg = ColorClip(size=(self.width, self.height), color=(12, 10, 18), duration=duration)
        final = CompositeVideoClip(
            [bg, static_clip, spk_clip, glow_clip, sub_clip],
            size=(self.width, self.height),
        )

        # ── audio ─────────────────────────────────────────────────────────────
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

    # ── write helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def concat_and_write(clips: list, output_path: str, fps: int = 25):
        """Concatenate all segments and export to output_path."""
        (_, _, _, _, concatenate_videoclips, _) = _import_moviepy()
        import inspect
        import tempfile as _tf

        final  = concatenate_videoclips(clips, method="compose")
        tmp_a  = os.path.join(_tf.gettempdir(), "ttv_final_audio.m4a")
        kw = dict(fps=fps, codec="libx264", audio_codec="aac",
                  temp_audiofile=tmp_a, remove_temp=False, logger="bar")
        if "verbose" in inspect.signature(final.write_videofile).parameters:
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
