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

# Sip timing
_SIP_PERIOD_LISTEN  = 5.0   # listener sips every ~5 s
_SIP_PERIOD_SPEAK   = 8.0   # speaker sips every ~8 s (between sentences)
_SIP_DURATION       = 1.8   # total rise-hold-lower time per sip


def _sip_lift(t: float, char_idx: int, is_speaking: bool) -> float:
    """
    Returns 0.0 (cup at hand) → 1.0 (cup at mouth).
    Uses a smooth sine bell so the motion feels natural.
    The listener sips more often than the speaker.
    """
    period = _SIP_PERIOD_SPEAK if is_speaking else _SIP_PERIOD_LISTEN
    # Stagger each character by their phase so they never sip simultaneously
    offset     = _CHAR_PHASE[char_idx % 2] / (2 * math.pi) * period
    t_in_cycle = (t + offset) % period
    if t_in_cycle >= _SIP_DURATION:
        return 0.0
    return math.sin(math.pi * t_in_cycle / _SIP_DURATION)   # smooth 0→1→0


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
    lift: float = 0.0,
) -> None:
    """
    Draw a warm ceramic coffee mug.

    Parameters
    ----------
    cx, cy   : rest position (hand level)
    lift     : 0.0 = cup at hand, 1.0 = cup raised to mouth
               The face is composited AFTER the cup, so at lift≈1 the face
               naturally occludes the upper rim — looks like drinking.

    Animated elements
    -----------------
    • Cup bobs gently with breathing when at rest.
    • Cup tilts forward during the sip (visible at the rim as it tilts).
    • Steam wisps only visible when cup is at rest (0 steam while sipping).
    """
    phase = _CHAR_PHASE[char_idx % 2]
    # Only bob when at rest; suppress bob while lifting
    bob   = int(3 * (1 - lift) * math.sin(2 * math.pi * 0.25 * t + phase))

    cx_f  = cx
    cy_f  = cy + bob    # vertical position already set by caller when sipping

    mw    = size
    mh    = int(size * 1.15)

    # When sipping, compress mh slightly (simulates forward tilt of cup)
    tilt_squeeze = 1.0 - 0.25 * lift   # 1.0 → 0.75 at peak sip
    mh_draw      = max(4, int(mh * tilt_squeeze))

    x0 = cx_f - mw // 2
    y0 = cy_f - mh_draw // 2
    x1 = x0 + mw
    y1 = y0 + mh_draw

    draw = ImageDraw.Draw(canvas, "RGBA")

    # ── Mug body ────────────────────────────────────────────────────────────
    draw.rounded_rectangle(
        [x0, y0, x1, y1], radius=max(2, int(5 * tilt_squeeze)),
        fill=(245, 240, 230, 235),
        outline=(160, 150, 135, 210), width=2,
    )

    # ── Coffee inside — extra visible when tilted toward viewer ─────────────
    inner_pad = 3
    lip_h     = max(4, int((7 + 8 * lift) * tilt_squeeze))  # opens wider when tilted
    draw.ellipse(
        [x0 + inner_pad, y0 + inner_pad,
         x1 - inner_pad, y0 + inner_pad + lip_h],
        fill=(75, 44, 18, 240),
    )
    # Highlight
    hi_x = x0 + inner_pad + 4
    draw.ellipse([hi_x, y0 + inner_pad + 1, hi_x + 5, y0 + inner_pad + 4],
                 fill=(140, 100, 60, 150))

    # ── Handle (hidden during deep tilt) ────────────────────────────────────
    handle_alpha = int(220 * (1 - lift * 0.7))
    if handle_alpha > 20:
        hx0 = x1 - 2
        hx1 = x1 + int(size * 0.38)
        hym = (y0 + y1) // 2
        draw.arc([hx0, hym - mh_draw // 4, hx1, hym + mh_draw // 4],
                 start=-90, end=90,
                 fill=(160, 150, 135, handle_alpha), width=3)

    # ── Steam wisps — only when cup is at rest (not sipping) ─────────────────
    steam_alpha_scale = max(0.0, 1.0 - lift * 2.5)   # fades out as cup lifts
    if steam_alpha_scale > 0.05:
        steam_period = 2.4
        steam_t      = (t % steam_period) / steam_period
        for i in range(3):
            wt    = (steam_t + i / 3) % 1.0
            alpha = int(170 * steam_alpha_scale * math.sin(math.pi * wt))
            if alpha < 10:
                continue
            rise = int(wt * 20)
            wx   = cx_f + int(4 * math.sin(2 * math.pi * wt + i)) + (i - 1) * 6
            wy   = y0 - 3 - rise
            ws   = max(1, 4 - int(wt * 3))
            draw.ellipse([wx - ws, wy - ws, wx + ws, wy + ws],
                         fill=(215, 215, 215, alpha))


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
        cup_rest_y = cy + int(self.fh * 0.40)   # hand/lap level
        cup_mouth_y = cy + int(self.fh * 0.13)  # mouth/chin level
        self.cup_rest = [
            (cx_l + cup_off_x, cup_rest_y),
            (cx_r - cup_off_x, cup_rest_y),
        ]
        self.cup_mouth = [
            (cx_l + int(cup_off_x * 0.4), cup_mouth_y),
            (cx_r - int(cup_off_x * 0.4), cup_mouth_y),
        ]
        self.cup_size = max(30, int(self.fw * 0.115))

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

        # ── 1. Coffee cups FIRST — faces composited on top ────────────────────
        # Drawing order: cup → face ensures the face naturally covers the cup
        # when it's raised to mouth level (looks like drinking).
        for char_idx, is_spk in ((listener_idx, False), (speaker_idx, True)):
            lift        = _sip_lift(t, char_idx, is_spk)
            rx, ry      = self.cup_rest[char_idx]
            mx, my      = self.cup_mouth[char_idx]
            cup_cx      = int(rx + lift * (mx - rx))
            cup_cy      = int(ry + lift * (my - ry))
            _draw_coffee_cup(canvas, cup_cx, cup_cy,
                             size=self.cup_size, t=t,
                             char_idx=char_idx, lift=lift)

        # ── 2. Listener — animated portrait (on top of cup) ──────────────────
        lcx, lcy = self.centres[listener_idx]
        nod_dy   = _listener_nod(t, listener_idx)
        l_face, ldx, ldy = _animate_face(
            listener_face, self.fw, self.fh, t, listener_idx, is_speaking=False
        )
        _blend_face(canvas, l_face,
                    lcx + ldx, lcy + ldy + nod_dy,
                    l_face.width, l_face.height)

        # ── 3. Speaker glow halo ──────────────────────────────────────────────
        scx, scy = self.centres[speaker_idx]
        sdx, sdy = _sway_offset(t, speaker_idx, is_speaking=True)
        gx = scx - self.fw // 2 + sdx
        gy = scy - self.fh // 2 + sdy
        canvas.alpha_composite(glow, (max(0, gx), max(0, gy)))

        # ── 4. Speaker — Wav2Lip frame + breathing scale ──────────────────────
        scale = _breathing_scale(t, speaker_idx, is_speaking=True)
        spk_img = Image.fromarray(speaker_frame)
        sw = max(1, int(self.fw * scale))
        sh = max(1, int(self.fh * scale))
        spk_img = spk_img.resize((sw, sh), Image.LANCZOS)
        _blend_face(canvas, spk_img,
                    scx + sdx, scy + sdy, sw, sh)

        # ── 5. Full frame composite ───────────────────────────────────────────
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
        segment_out_path: str,          # write directly to this MP4 file
    ) -> str:
        """
        Render one dialogue segment and write it as a self-contained MP4
        (video + audio muxed together via ffmpeg).  Returns segment_out_path.

        Writing each segment immediately — rather than collecting MoviePy
        clips and concatenating later — guarantees the audio is always
        present.  MoviePy lazy-loads AudioFileClip readers that can become
        stale by the time the final write happens; bypassing that entirely
        removes the most common cause of silent output.
        """
        import subprocess
        VideoFileClip, AudioFileClip, VideoClip, _ = _mp()

        talking      = VideoFileClip(wav2lip_video_path)
        wav2lip_dur  = talking.duration
        listener_idx = 1 - speaker_idx

        # ── Determine duration from TTS audio (master clock) ─────────────────
        audio_dur = wav2lip_dur
        audio_ok  = os.path.exists(audio_path) and os.path.getsize(audio_path) > 256
        if audio_ok:
            try:
                tmp_a = AudioFileClip(audio_path)
                audio_dur = tmp_a.duration
                tmp_a.close()
                print(f"  [Composer] Audio: {audio_dur:.2f}s  ({os.path.basename(audio_path)})")
            except Exception as e:
                print(f"  [Composer] ⚠ Cannot read audio duration: {e}")
                audio_ok = False
        else:
            print(f"  [Composer] ⚠ Audio file missing/empty: {audio_path}")

        duration = max(audio_dur, 0.5)

        # ── Pre-build static per-segment resources ────────────────────────────
        if self.room_bg_image is not None:
            room_bg = self.room_bg_image.resize((self.width, self.char_h), Image.LANCZOS)
        else:
            room_bg = _get_room_bg(self.width, self.char_h, self.room_bg_path)

        listener_face = Image.open(face_image_paths[listener_idx]).convert("RGB")
        glow          = _speaker_glow(self.fw, self.fh)
        subtitle      = _subtitle_img(self.width, self.sub_h, speaker_name, dialogue_text)

        def make_frame(t: float) -> np.ndarray:
            safe_t    = max(0.0, min(t, wav2lip_dur - 1.0 / max(1, self.fps)))
            spk_frame = talking.get_frame(safe_t)
            return self._build_frame(
                t, room_bg, listener_face, listener_idx,
                spk_frame, speaker_idx, glow, subtitle,
            )

        # ── Write video-only to a temp file ───────────────────────────────────
        tmp_vid = segment_out_path + "_noaudio.mp4"
        clip    = VideoClip(make_frame, duration=duration)
        clip    = clip.with_fps(self.fps) if hasattr(clip, "with_fps") else clip.set_fps(self.fps)
        clip.write_videofile(
            tmp_vid, fps=self.fps,
            codec="libx264", audio=False,
            logger=None,
        )
        talking.close()

        # ── Mux audio in via ffmpeg — 100 % reliable ─────────────────────────
        if audio_ok:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_vid,           # video track (no audio)
                "-i", audio_path,        # TTS audio (WAV/MP3)
                "-c:v", "copy",          # copy video stream (no re-encode)
                "-c:a", "aac",           # encode audio to AAC
                "-ar", "44100",
                "-ac", "1",
                "-shortest",             # trim to shorter of the two streams
                segment_out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.isfile(segment_out_path):
                print(f"  [Composer] ✓ Segment written with audio: {segment_out_path}")
            else:
                print(f"  [Composer] ⚠ ffmpeg mux failed:\n{result.stderr[-400:]}")
                # Fallback: just use video-only file
                import shutil
                shutil.move(tmp_vid, segment_out_path)
        else:
            # No audio available — keep video-only segment so pipeline continues
            import shutil
            shutil.move(tmp_vid, segment_out_path)
            print(f"  [Composer] ⚠ Segment written WITHOUT audio: {segment_out_path}")

        # Clean up temp video-only file
        if os.path.isfile(tmp_vid):
            try:
                os.unlink(tmp_vid)
            except Exception:
                pass

        return segment_out_path

    # ── export ────────────────────────────────────────────────────────────────

    @staticmethod
    def concat_and_write(segment_paths: list[str], output_path: str, fps: int = 25):
        """
        Concatenate pre-written segment MP4 files into the final video using
        ffmpeg's concat demuxer.  This is far more reliable than MoviePy's
        concatenate_videoclips() for preserving audio.
        """
        import subprocess
        import tempfile

        if len(segment_paths) == 0:
            raise ValueError("No segments to concatenate.")

        if len(segment_paths) == 1:
            import shutil
            shutil.copy(segment_paths[0], output_path)
            print(f"  [Composer] Single segment copied → {output_path}")
            return

        # Write ffmpeg concat list
        list_file = tempfile.mktemp(suffix="_concat_list.txt")
        with open(list_file, "w") as f:
            for seg in segment_paths:
                # Escape single quotes in paths
                safe = seg.replace("'", "\\'")
                f.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",        # stream copy — no re-encoding
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            os.unlink(list_file)
        except Exception:
            pass

        if result.returncode != 0:
            print(f"  [Composer] ffmpeg concat stderr:\n{result.stderr[-600:]}")
            raise RuntimeError("ffmpeg concat failed — check output above.")

        size = os.path.getsize(output_path)
        print(f"  [Composer] ✓ Final video: {output_path}  ({size // 1024} KB)")
