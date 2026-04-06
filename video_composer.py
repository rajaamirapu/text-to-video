"""
video_composer.py

Seamless two-person conversation scene — no rectangular boxes.

Each frame is composited entirely in PIL so characters blend
naturally into the room with soft oval masks, exactly like a
real photo of two people sitting together.

Layout
──────
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │        [person L]      plant/window      [person R]  │  85 %
  │     soft-oval blend    room background   soft blend  │
  │                                                      │
  ├──────────────────────────────────────────────────────┤
  │   Subtitle  "Speaker: text …"                        │  15 %
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ── MoviePy imports ──────────────────────────────────────────────────────────

def _mp():
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip, concatenate_videoclips
    except ImportError:
        from moviepy import VideoFileClip, AudioFileClip, VideoClip, concatenate_videoclips
    return VideoFileClip, AudioFileClip, VideoClip, concatenate_videoclips


# ─────────────────────────────────────────────────────────────────────────────
# Room background — modern lounge (plants + large windows + sofa)
# ─────────────────────────────────────────────────────────────────────────────

def _make_room_background(width: int, height: int) -> Image.Image:
    """
    Renders a modern lounge room similar to the reference image:
    large rear windows, monstera plants, warm ambient light, sofa seats.
    """
    img  = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    wall_h = int(height * 0.68)

    # ── rear wall — warm off-white gradient ──────────────────────────────────
    for y in range(wall_h):
        t   = y / max(1, wall_h)
        col = (int(240 - t * 22), int(238 - t * 18), int(228 - t * 14))
        draw.line([(0, y), (width, y)], fill=col)

    # ── large panoramic window (centre rear) ─────────────────────────────────
    win_x0, win_y0 = int(width * 0.20), int(height * 0.04)
    win_x1, win_y1 = int(width * 0.80), int(wall_h * 0.82)

    # sky gradient inside glass
    for y in range(win_y0, win_y1):
        t   = (y - win_y0) / max(1, win_y1 - win_y0)
        col = (int(185 + t * 40), int(215 + t * 20), int(245))
        draw.line([(win_x0, y), (win_x1, y)], fill=col)

    # foliage blur seen through window
    for i in range(12):
        gx = win_x0 + int((win_x1 - win_x0) * (0.1 + i * 0.07))
        gy = win_y0 + int((win_y1 - win_y0) * (0.35 + (i % 3) * 0.12))
        gr = int(height * 0.10) + (i % 4) * 12
        alpha_col = (int(80 + i * 8), int(130 + i * 5), int(70 + i * 3))
        draw.ellipse([gx - gr, gy - gr // 2, gx + gr, gy + gr], fill=alpha_col)

    # window frame
    frame_col = (90, 95, 90)
    draw.rectangle([win_x0, win_y0, win_x1, win_y1], outline=frame_col, width=6)
    # frame mullions — two vertical
    for frac in (0.33, 0.67):
        fx = int(win_x0 + (win_x1 - win_x0) * frac)
        draw.line([(fx, win_y0), (fx, win_y1)], fill=frame_col, width=5)
    # one horizontal
    fy = int(win_y0 + (win_y1 - win_y0) * 0.5)
    draw.line([(win_x0, fy), (win_x1, fy)], fill=frame_col, width=5)

    # window sill
    draw.rectangle([win_x0 - 6, win_y1, win_x1 + 6, win_y1 + 10],
                   fill=(200, 195, 180))

    # window light bloom on wall
    for r in range(300, 0, -25):
        t   = r / 300
        col = (min(255, int(255 - t * 15)),
               min(255, int(255 - t * 12)),
               min(255, int(250 - t * 8)))
        cx, cy = (win_x0 + win_x1) // 2, (win_y0 + win_y1) // 2
        draw.ellipse([cx - r * 2, cy - r, cx + r * 2, cy + r], fill=col)

    # Re-draw window over bloom
    for y in range(win_y0, win_y1):
        t   = (y - win_y0) / max(1, win_y1 - win_y0)
        col = (int(185 + t * 40), int(215 + t * 20), 245)
        draw.line([(win_x0, y), (win_x1, y)], fill=col)
    draw.rectangle([win_x0, win_y0, win_x1, win_y1], outline=frame_col, width=6)
    for frac in (0.33, 0.67):
        fx = int(win_x0 + (win_x1 - win_x0) * frac)
        draw.line([(fx, win_y0), (fx, win_y1)], fill=frame_col, width=5)
    draw.line([(win_x0, fy), (win_x1, fy)], fill=frame_col, width=5)

    # ── floor — light wood / stone ───────────────────────────────────────────
    for y in range(wall_h, height):
        t   = (y - wall_h) / max(1, height - wall_h)
        col = (int(210 - t * 40), int(200 - t * 35), int(185 - t * 30))
        draw.line([(0, y), (width, y)], fill=col)

    # subtle floor planks
    plank_h = max(1, (height - wall_h) // 8)
    for i in range(9):
        py = wall_h + i * plank_h
        if py < height:
            draw.line([(0, py), (width, py)], fill=(190, 180, 165), width=1)

    # baseboard
    draw.rectangle([0, wall_h - 14, width, wall_h + 5], fill=(220, 215, 200))
    draw.line([(0, wall_h - 14), (width, wall_h - 14)], fill=(230, 225, 215), width=2)

    # ── monstera plants (left and right foreground) ───────────────────────────
    for side, px in ((+1, int(width * 0.06)), (-1, int(width * 0.94))):
        pot_cx = px
        pot_y  = int(height * 0.78)
        pot_w, pot_h2 = 38, 44
        # pot
        draw.ellipse([pot_cx - pot_w, pot_y, pot_cx + pot_w, pot_y + pot_h2],
                     fill=(130, 100, 75))
        draw.ellipse([pot_cx - pot_w + 4, pot_y + 4,
                      pot_cx + pot_w - 4, pot_y + 14], fill=(150, 118, 88))
        # stem
        draw.line([(pot_cx, pot_y), (pot_cx + side * 20, int(height * 0.30))],
                  fill=(60, 90, 55), width=4)
        # leaves (large monstera-ish ovals)
        leaf_col  = (55, 120, 65)
        leaf_col2 = (70, 145, 75)
        leaf_positions = [
            (pot_cx + side * 25, int(height * 0.28), 75, 45),
            (pot_cx + side * 10, int(height * 0.38), 60, 38),
            (pot_cx + side * 40, int(height * 0.35), 65, 40),
            (pot_cx + side * 5,  int(height * 0.22), 55, 35),
            (pot_cx + side * 55, int(height * 0.25), 50, 32),
        ]
        for lx, ly, lw, lh in leaf_positions:
            draw.ellipse([lx - lw, ly - lh, lx + lw, ly + lh], fill=leaf_col)
            # mid-rib
            draw.line([(lx - lw + 10, ly), (lx + lw - 10, ly)],
                      fill=leaf_col2, width=2)

    # ── sofa / couch (lower left and right) ──────────────────────────────────
    sofa_y = int(height * 0.72)
    sofa_h2 = int(height * 0.14)
    sofa_col  = (180, 170, 158)
    sofa_col2 = (165, 155, 143)
    for sx0, sx1 in (
        (int(width * 0.01), int(width * 0.38)),
        (int(width * 0.62), int(width * 0.99)),
    ):
        # seat
        draw.rounded_rectangle([sx0, sofa_y, sx1, sofa_y + sofa_h2],
                                radius=8, fill=sofa_col)
        # back cushion
        draw.rounded_rectangle([sx0, int(height * 0.60), sx1, sofa_y + 6],
                                radius=6, fill=sofa_col2)
        # arm rests
        arm_w = int((sx1 - sx0) * 0.08)
        for ax in (sx0, sx1 - arm_w):
            draw.rounded_rectangle([ax, int(height * 0.62), ax + arm_w, sofa_y + sofa_h2],
                                    radius=5, fill=sofa_col2)
        # seat highlight
        draw.line([(sx0 + 10, sofa_y + 8), (sx1 - 10, sofa_y + 8)],
                  fill=(200, 192, 180), width=2)

    # coffee table (centre)
    ct_y  = int(height * 0.78)
    ct_w  = int(width * 0.18)
    ct_cx = width // 2
    draw.rounded_rectangle([ct_cx - ct_w, ct_y, ct_cx + ct_w, ct_y + 22],
                            radius=6, fill=(160, 140, 110))
    draw.line([(ct_cx - ct_w + 6, ct_y + 3), (ct_cx + ct_w - 6, ct_y + 3)],
              fill=(180, 162, 128), width=2)
    # coffee cups on table
    for cup_x in (ct_cx - 30, ct_cx + 30):
        draw.ellipse([cup_x - 10, ct_y - 8, cup_x + 10, ct_y + 6], fill=(245, 243, 238))
        draw.ellipse([cup_x - 8, ct_y - 6, cup_x + 8, ct_y + 4], fill=(220, 210, 195))

    # ── vignette ─────────────────────────────────────────────────────────────
    vig = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vd  = ImageDraw.Draw(vig)
    for i in range(25):
        alpha = int(70 * (1 - i / 25) ** 2)
        pad   = int(i * min(width, height) * 0.02)
        x1 = width - pad - 1
        y1 = height - pad - 1
        if x1 <= pad or y1 <= pad:
            break
        vd.rectangle([pad, pad, x1, y1], outline=(0, 0, 0, alpha), width=2)

    base = img.convert("RGBA")
    base.alpha_composite(vig)
    return base.convert("RGB")


_BG_CACHE: dict = {}


def _get_room_bg(width: int, height: int, path: str | None = None) -> Image.Image:
    key = (width, height, path)
    if key not in _BG_CACHE:
        if path and os.path.isfile(path):
            img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
        else:
            img = _make_room_background(width, height)
        _BG_CACHE[key] = img
    return _BG_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# Face blending helpers
# ─────────────────────────────────────────────────────────────────────────────

def _soft_oval_mask(w: int, h: int, feather: float = 0.12) -> Image.Image:
    """
    Create a smooth oval alpha mask — hard centre, feathered edges.
    Larger feather = softer blend into background.
    """
    mask = Image.new("L", (w, h), 0)
    md   = ImageDraw.Draw(mask)
    px   = int(w * 0.05)
    py   = int(h * 0.03)
    md.ellipse([px, py, w - px, h - py], fill=255)
    blur_r = max(2, int(min(w, h) * feather))
    return mask.filter(ImageFilter.GaussianBlur(blur_r))


def _blend_face(
    canvas: Image.Image,          # RGBA canvas to paste onto
    face_img: Image.Image,        # RGB face/bust photo
    cx: int, cy: int,             # centre position on canvas
    fw: int, fh: int,             # target face size
    feather: float = 0.14,
) -> None:
    """Paste face_img onto canvas with a soft oval blend, in-place."""
    face = face_img.convert("RGB").resize((fw, fh), Image.LANCZOS)
    mask = _soft_oval_mask(fw, fh, feather)
    face_rgba = face.convert("RGBA")
    face_rgba.putalpha(mask)
    x = cx - fw // 2
    y = cy - fh // 2
    # Clip to canvas bounds
    canvas.alpha_composite(face_rgba, (max(0, x), max(0, y)))


def _blend_frame_array(
    canvas: Image.Image,
    frame_arr: np.ndarray,
    cx: int, cy: int,
    fw: int, fh: int,
    feather: float = 0.14,
) -> None:
    """Same as _blend_face but from a raw numpy frame (Wav2Lip output)."""
    face = Image.fromarray(frame_arr).resize((fw, fh), Image.LANCZOS)
    _blend_face(canvas, face, cx, cy, fw, fh, feather)


def _speaker_glow(fw: int, fh: int) -> Image.Image:
    """Soft warm rim-light halo around the active speaker. Returns RGBA."""
    glow = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    for i in range(14):
        alpha = max(0, 140 - i * 10)
        p     = i * 3
        if fw - p * 2 > 0 and fh - p * 2 > 0:
            gd.ellipse([p, p, fw - p, fh - p], outline=(255, 210, 90, alpha), width=3)
    return glow.filter(ImageFilter.GaussianBlur(6))


def _subtitle_img(width: int, height: int, speaker: str, text: str) -> Image.Image:
    img  = Image.new("RGB", (width, height), (10, 8, 16))
    draw = ImageDraw.Draw(img)
    # gold accent bar
    draw.rectangle([0, 0, width, 4], fill=(210, 168, 42))
    draw.text((32, 10), f"{speaker}:", fill=(255, 205, 60))
    ty = 32
    for line in textwrap.wrap(text, width=int(width / 8.5))[:3]:
        draw.text((32, ty), line, fill=(232, 230, 226))
        ty += 23
    return img


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
        room_bg_path: str | None = None,       # legacy path-based option
        room_bg_image: Image.Image | None = None,   # pre-generated PIL image
        char_body_paths: dict | None = None,   # {name: path_or_None}
    ):
        self.width           = width
        self.height          = height
        self.fps             = fps
        self.room_bg_path    = room_bg_path
        self.room_bg_image   = room_bg_image
        self.char_body_paths = char_body_paths or {}
        self.char_h          = int(height * (1 - self.SUBTITLE_RATIO))
        self.sub_h           = height - self.char_h

        # Face bust size — tall portrait, ~36 % of frame width
        self.fw = int(width * 0.36)
        self.fh = int(self.fw * 1.28)

        # Character centre positions (seated in sofa areas)
        cx_l = int(width * 0.24)
        cx_r = int(width * 0.76)
        cy   = int(self.char_h * 0.52)
        self.centres = [(cx_l, cy), (cx_r, cy)]

    # ── build one composite PIL image (room + both faces) ─────────────────────

    def _build_frame(
        self,
        room_bg: Image.Image,
        listener_face: Image.Image,
        listener_idx: int,
        speaker_frame: np.ndarray,
        speaker_idx: int,
        glow: Image.Image,
        subtitle: Image.Image,
    ) -> np.ndarray:
        """Compose a single RGB frame as numpy array."""

        canvas = room_bg.convert("RGBA")

        # 1. Listener — static soft-oval portrait
        lcx, lcy = self.centres[listener_idx]
        _blend_face(canvas, listener_face, lcx, lcy, self.fw, self.fh)

        # 2. Speaker glow halo (behind face)
        scx, scy = self.centres[speaker_idx]
        gx, gy   = scx - self.fw // 2, scy - self.fh // 2
        canvas.alpha_composite(glow, (max(0, gx), max(0, gy)))

        # 3. Speaker — Wav2Lip frame with soft oval blend
        _blend_frame_array(canvas, speaker_frame, scx, scy, self.fw, self.fh)

        # 4. Composite into full frame (room area + subtitle)
        full = Image.new("RGB", (self.width, self.height), (10, 8, 16))
        full.paste(canvas.convert("RGB"), (0, 0))
        full.paste(subtitle, (0, self.char_h))

        return np.asarray(full, dtype=np.uint8)

    # ── public segment builder ────────────────────────────────────────────────

    def create_segment(
        self,
        speaker_idx: int,
        dialogue_text: str,
        speaker_name: str,
        face_image_paths: list[str],
        wav2lip_video_path: str,
        audio_path: str,
    ):
        VideoFileClip, AudioFileClip, VideoClip, _ = _mp()

        talking      = VideoFileClip(wav2lip_video_path)
        duration     = talking.duration
        listener_idx = 1 - speaker_idx

        # Pre-build static resources
        # Use pre-generated image if available, else fall back to PIL/path
        if self.room_bg_image is not None:
            room_bg = self.room_bg_image.resize((self.width, self.char_h), Image.LANCZOS)
        else:
            room_bg = _get_room_bg(self.width, self.char_h, self.room_bg_path)

        # Use SD-generated body image if available, else face headshot
        listener_name = None  # we don't have name here; use path directly
        listener_img_path = face_image_paths[listener_idx]
        listener_face = Image.open(listener_img_path).convert("RGB")
        glow          = _speaker_glow(self.fw, self.fh)
        subtitle      = _subtitle_img(self.width, self.sub_h, speaker_name, dialogue_text)

        def make_frame(t: float) -> np.ndarray:
            spk_frame = talking.get_frame(t)
            return self._build_frame(
                room_bg, listener_face, listener_idx,
                spk_frame, speaker_idx,
                glow, subtitle,
            )

        clip = VideoClip(make_frame, duration=duration)
        clip = clip.with_fps(self.fps) if hasattr(clip, "with_fps") else clip.set_fps(self.fps)

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio = AudioFileClip(audio_path)
            if audio.duration > duration:
                try:
                    audio = audio.subclipped(0, duration)
                except AttributeError:
                    audio = audio.subclip(0, duration)
            try:
                clip = clip.with_audio(audio)
            except AttributeError:
                clip = clip.set_audio(audio)

        return clip

    # ── export ────────────────────────────────────────────────────────────────

    @staticmethod
    def concat_and_write(clips: list, output_path: str, fps: int = 25):
        _, _, _, concatenate_videoclips = _mp()
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
