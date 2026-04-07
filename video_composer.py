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
import math
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
# Animation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Per-character phase offsets so the two people animate independently
_CHAR_PHASE = [0.0, math.pi * 0.61]   # irrational fraction → never in sync


def _breathing_scale(t: float, char_idx: int, is_speaking: bool) -> float:
    """
    Subtle sinusoidal scale that mimics breathing.
    Speakers breathe slightly faster and with a little more depth.
    """
    phase  = _CHAR_PHASE[char_idx % 2]
    freq   = 0.32 if is_speaking else 0.25          # Hz
    depth  = 0.018 if is_speaking else 0.010         # ± fraction of size
    return 1.0 + depth * math.sin(2 * math.pi * freq * t + phase)


def _sway_offset(t: float, char_idx: int, is_speaking: bool) -> tuple[int, int]:
    """
    Gentle X/Y drift — different frequencies on each axis so it looks organic.
    Speakers sway a little more (leaning in with emphasis).
    """
    phase  = _CHAR_PHASE[char_idx % 2]
    amp_x  = 5 if is_speaking else 3
    amp_y  = 3 if is_speaking else 2
    # Slightly different freq on each axis for a Lissajous feel
    dx = amp_x * math.sin(2 * math.pi * 0.17 * t + phase)
    dy = amp_y * math.sin(2 * math.pi * 0.11 * t + phase + 1.1)
    return int(dx), int(dy)


def _listener_nod(t: float, char_idx: int) -> int:
    """
    Slow occasional head-nod for the listener — y-axis only, long period.
    Returns a pixel offset to add to cy.
    """
    phase = _CHAR_PHASE[char_idx % 2]
    # Very slow nod: period ~4 s, amplitude 4 px
    return int(4 * math.sin(2 * math.pi * 0.25 * t + phase))


def _animate_face(
    face_img: Image.Image,
    fw: int, fh: int,
    t: float,
    char_idx: int,
    is_speaking: bool,
) -> tuple[Image.Image, int, int]:
    """
    Apply breathing scale to *face_img* and compute (dx, dy) sway.
    Returns (scaled_face, dx, dy).
    """
    scale  = _breathing_scale(t, char_idx, is_speaking)
    new_w  = max(1, int(fw * scale))
    new_h  = max(1, int(fh * scale))
    scaled = face_img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    dx, dy = _sway_offset(t, char_idx, is_speaking)
    return scaled, dx, dy


# ─────────────────────────────────────────────────────────────────────────────
# Coffee-cup drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_coffee_cup(
    canvas: Image.Image,
    cx: int, cy: int,
    size: int = 38,
    t: float = 0.0,
    char_idx: int = 0,
) -> None:
    """
    Draw a warm ceramic coffee mug held by the character at (cx, cy).

    Animated elements
    -----------------
    • Cup bobs gently with the breathing rhythm.
    • Steam wisps drift upward and fade in/out.
    • Coffee surface ripples very slightly.
    """
    phase     = _CHAR_PHASE[char_idx % 2]
    bob       = int(3 * math.sin(2 * math.pi * 0.25 * t + phase))  # sync with breathing
    cx_f      = cx
    cy_f      = cy + bob

    mw   = size
    mh   = int(size * 1.15)
    x0   = cx_f - mw // 2
    y0   = cy_f - mh // 2
    x1   = x0 + mw
    y1   = y0 + mh

    draw = ImageDraw.Draw(canvas, "RGBA")

    # ── Mug body (warm white ceramic) ────────────────────────────────────────
    draw.rounded_rectangle(
        [x0, y0, x1, y1], radius=5,
        fill=(245, 240, 230, 230),
        outline=(160, 150, 135, 200), width=2,
    )

    # ── Coffee inside (dark brown liquid) ────────────────────────────────────
    inner_pad = 4
    lip_h     = 7
    draw.ellipse(
        [x0 + inner_pad, y0 + inner_pad,
         x1 - inner_pad, y0 + inner_pad + lip_h],
        fill=(80, 48, 22, 230),
    )
    # Tiny highlight on coffee surface
    hi_x = x0 + inner_pad + 4
    draw.ellipse([hi_x, y0 + inner_pad + 1, hi_x + 6, y0 + inner_pad + 4],
                 fill=(140, 100, 60, 160))

    # ── Handle ────────────────────────────────────────────────────────────────
    hx0, hx1 = x1 - 2, x1 + int(size * 0.38)
    hym      = (y0 + y1) // 2
    draw.arc([hx0, hym - mh // 4, hx1, hym + mh // 4],
             start=-90, end=90, fill=(160, 150, 135, 220), width=3)

    # ── Steam wisps ───────────────────────────────────────────────────────────
    steam_period = 2.4
    steam_t      = (t % steam_period) / steam_period   # 0 → 1 cycle
    n_wisps      = 3
    for i in range(n_wisps):
        wt    = (steam_t + i / n_wisps) % 1.0          # each wisp offset
        alpha = int(180 * math.sin(math.pi * wt))      # fade in/out
        if alpha < 10:
            continue
        rise  = int(wt * 22)                            # how high it's risen
        wx    = cx_f + int(5 * math.sin(2 * math.pi * wt + i)) + (i - 1) * 6
        wy    = y0 - 3 - rise
        ws    = max(1, 4 - int(wt * 3))                # wisp shrinks as it rises
        draw.ellipse(
            [wx - ws, wy - ws, wx + ws, wy + ws],
            fill=(210, 210, 210, alpha),
        )


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

        # Coffee-cup anchor — below & slightly inward from each character centre
        cup_off_x = int(self.fw * 0.22)   # inward horizontal offset
        cup_off_y = int(self.fh * 0.40)   # below face centre
        self.cup_centres = [
            (cx_l + cup_off_x,  cy + cup_off_y),   # left character holds cup to right
            (cx_r - cup_off_x,  cy + cup_off_y),   # right character holds cup to left
        ]
        self.cup_size = max(28, int(self.fw * 0.11))

    # ── build one composite PIL image (room + both faces + animation) ─────────

    def _build_frame(
        self,
        t: float,
        room_bg: Image.Image,
        listener_face: Image.Image,
        listener_idx: int,
        speaker_frame: np.ndarray,
        speaker_idx: int,
        glow: Image.Image,
        subtitle: Image.Image,
    ) -> np.ndarray:
        """Compose a single animated RGB frame as numpy array."""

        canvas = room_bg.convert("RGBA")

        # ── 1. Listener — animated portrait ──────────────────────────────────
        lcx, lcy = self.centres[listener_idx]
        nod_dy   = _listener_nod(t, listener_idx)
        l_face, ldx, ldy = _animate_face(
            listener_face, self.fw, self.fh, t, listener_idx, is_speaking=False
        )
        _blend_face(canvas, l_face,
                    lcx + ldx, lcy + ldy + nod_dy,
                    l_face.width, l_face.height)

        # ── 2. Speaker glow halo ──────────────────────────────────────────────
        scx, scy = self.centres[speaker_idx]
        sdx, sdy = _sway_offset(t, speaker_idx, is_speaking=True)
        gx = scx - self.fw // 2 + sdx
        gy = scy - self.fh // 2 + sdy
        canvas.alpha_composite(glow, (max(0, gx), max(0, gy)))

        # ── 3. Speaker — Wav2Lip frame + breathing scale ──────────────────────
        scale = _breathing_scale(t, speaker_idx, is_speaking=True)
        spk_img = Image.fromarray(speaker_frame)
        sw = max(1, int(self.fw * scale))
        sh = max(1, int(self.fh * scale))
        spk_img = spk_img.resize((sw, sh), Image.LANCZOS)
        _blend_face(canvas, spk_img,
                    scx + sdx, scy + sdy, sw, sh)

        # ── 4. Coffee cups with steam animation ──────────────────────────────
        for char_idx in (listener_idx, speaker_idx):
            cup_cx, cup_cy = self.cup_centres[char_idx]
            _draw_coffee_cup(canvas, cup_cx, cup_cy,
                             size=self.cup_size, t=t, char_idx=char_idx)

        # ── 5. Composite into full frame (room area + subtitle bar) ──────────
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
        wav2lip_dur  = talking.duration
        listener_idx = 1 - speaker_idx

        # Use audio duration as the master clock (Wav2Lip sometimes trims audio)
        audio = None
        audio_dur = wav2lip_dur
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio     = AudioFileClip(audio_path)
            audio_dur = audio.duration

        # Final segment duration = audio duration
        duration = audio_dur

        # Pre-build static resources
        if self.room_bg_image is not None:
            room_bg = self.room_bg_image.resize((self.width, self.char_h), Image.LANCZOS)
        else:
            room_bg = _get_room_bg(self.width, self.char_h, self.room_bg_path)

        listener_img_path = face_image_paths[listener_idx]
        listener_face     = Image.open(listener_img_path).convert("RGB")
        glow              = _speaker_glow(self.fw, self.fh)
        subtitle          = _subtitle_img(self.width, self.sub_h, speaker_name, dialogue_text)

        def make_frame(t: float) -> np.ndarray:
            # Clamp t to Wav2Lip video duration so we never seek past end
            safe_t    = min(t, wav2lip_dur - 1.0 / max(1, self.fps))
            safe_t    = max(0.0, safe_t)
            spk_frame = talking.get_frame(safe_t)
            return self._build_frame(
                t,                                  # ← animation clock
                room_bg, listener_face, listener_idx,
                spk_frame, speaker_idx,
                glow, subtitle,
            )

        clip = VideoClip(make_frame, duration=duration)
        clip = clip.with_fps(self.fps) if hasattr(clip, "with_fps") else clip.set_fps(self.fps)

        if audio is not None:
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
