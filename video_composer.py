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
# Emotion detection
# ─────────────────────────────────────────────────────────────────────────────

_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "excited":    ["incredible", "amazing", "wow", "fantastic", "revolutionary",
                   "breakthrough", "huge", "awesome", "brilliant", "great",
                   "exciting", "wonderful", "magnificent", "outstanding"],
    "surprised":  ["really", "seriously", "no way", "impossible", "unbelievable",
                   "shocked", "never", "unexpected", "whoa", "wait", "what"],
    "happy":      ["thanks", "love", "happy", "pleased", "glad", "delighted",
                   "perfect", "excellent", "congratulations", "great question",
                   "of course", "hello", "welcome"],
    "curious":    ["how", "why", "when", "explain", "tell me", "interesting",
                   "wonder", "question", "mean", "does", "can you", "what about"],
    "thoughtful": ["think", "perhaps", "maybe", "consider", "believe", "actually",
                   "complex", "future", "plan", "planning", "developing", "analyse"],
}


def _detect_emotion(text: str) -> str:
    """Return the dominant emotion for a line of dialogue using keyword matching."""
    low = text.lower()
    scores = {emo: sum(1 for kw in kws if kw in low)
              for emo, kws in _EMOTION_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Expression overlays  (drawn on the LISTENER in reaction to what's being said)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_expression(
    canvas: Image.Image,
    cx: int, cy: int,
    fw: int, fh: int,
    emotion: str,
    t: float,
    char_idx: int,
) -> None:
    """
    Draw animated expression overlays on the canvas at the listener's face
    position.  All elements are drawn using PIL arcs/lines/ellipses so no
    external assets are needed.

    Expressions
    -----------
    excited    → arched raised brows  +  animated pink blush
    surprised  → wide raised brows    +  sweat drop
    happy      → smile arc            +  soft cheek blush
    curious    → one raised brow (quizzical)
    thoughtful → furrowed / angled brows
    neutral    → no overlay
    """
    if emotion == "neutral":
        return

    draw  = ImageDraw.Draw(canvas)
    phase = _CHAR_PHASE[char_idx % 2]

    # Key face landmarks (approximate, based on oval portrait framing)
    brow_y  = cy - int(fh * 0.22)          # eyebrow arch
    eye_y   = cy - int(fh * 0.12)          # eye line
    cheek_y = cy + int(fh * 0.06)          # cheek / smile region
    eye_sep = int(fw * 0.20)               # half-gap between eye centres
    bw      = int(fw * 0.13)               # half-width of one brow arc
    bh      = int(fh * 0.05)               # height of brow arc box

    if emotion == "excited":
        pulse      = 0.7 + 0.3 * math.sin(2 * math.pi * 1.5 * t + phase)
        brow_lift  = int(10 * pulse)
        # Both brows arched high
        for sign in (-1, 1):
            ecx = cx + sign * eye_sep
            draw.arc(
                [ecx - bw, brow_y - brow_lift - bh,
                 ecx + bw, brow_y - brow_lift + bh],
                start=200, end=340, fill=(30, 20, 12), width=3,
            )
        # Animated blush circles on cheeks
        blush_r = int(fw * 0.09)
        blush_a = int(130 * pulse)
        bl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bl)
        for sign in (-1, 1):
            bcx = cx + sign * int(eye_sep * 1.8)
            bd.ellipse([bcx - blush_r, cheek_y - blush_r,
                        bcx + blush_r, cheek_y + blush_r],
                       fill=(255, 110, 110, blush_a))
        canvas.alpha_composite(bl)

    elif emotion == "surprised":
        pulse      = abs(math.sin(2 * math.pi * 0.9 * t + phase))
        brow_lift  = int(14 * pulse)
        # Both brows raised high with slight arch
        for sign in (-1, 1):
            ecx = cx + sign * eye_sep
            draw.arc(
                [ecx - bw, brow_y - brow_lift - int(bh * 1.4),
                 ecx + bw, brow_y - brow_lift + int(bh * 0.6)],
                start=195, end=345, fill=(20, 14, 8), width=4,
            )
        # Animated sweat drop
        sd_x     = cx + int(fw * 0.38)
        sd_y     = brow_y - brow_lift - int(fh * 0.06)
        drop_sz  = int(fw * 0.045)
        drop_a   = int(220 * pulse)
        dl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        dd = ImageDraw.Draw(dl)
        # teardrop: oval body + triangle tip above
        dd.ellipse([sd_x, sd_y + drop_sz // 2,
                    sd_x + drop_sz, sd_y + drop_sz * 2],
                   fill=(80, 170, 255, drop_a))
        dd.polygon([(sd_x + drop_sz // 2, sd_y),
                    (sd_x,                sd_y + drop_sz // 2 + 2),
                    (sd_x + drop_sz,      sd_y + drop_sz // 2 + 2)],
                   fill=(80, 170, 255, drop_a))
        canvas.alpha_composite(dl)

    elif emotion == "happy":
        pulse     = 0.75 + 0.25 * math.sin(2 * math.pi * 0.55 * t + phase)
        # Upward smile arc near mouth level
        smile_y   = cy + int(fh * 0.14)
        smile_w   = int(fw * 0.26)
        smile_h   = int(fh * 0.07)
        draw.arc(
            [cx - smile_w, smile_y - smile_h,
             cx + smile_w, smile_y + smile_h],
            start=10, end=170, fill=(190, 70, 70), width=3,
        )
        # Soft cheek blush
        blush_r = int(fw * 0.07)
        blush_a = int(85 * pulse)
        bl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bl)
        for sign in (-1, 1):
            bcx = cx + sign * int(eye_sep * 1.7)
            bd.ellipse([bcx - blush_r, cheek_y - blush_r,
                        bcx + blush_r, cheek_y + blush_r],
                       fill=(255, 165, 130, blush_a))
        canvas.alpha_composite(bl)

    elif emotion == "curious":
        pulse     = 0.5 + 0.5 * math.sin(2 * math.pi * 0.4 * t + phase)
        brow_lift = int(8 * pulse)
        # One eyebrow raised (which side depends on char_idx for variety)
        raised_sign = -1 if char_idx % 2 == 0 else 1
        flat_sign   = -raised_sign
        # Raised (arched) brow
        ecx_r = cx + raised_sign * eye_sep
        draw.arc(
            [ecx_r - bw, brow_y - brow_lift - bh,
             ecx_r + bw, brow_y - brow_lift + int(bh * 0.4)],
            start=200, end=340, fill=(25, 17, 8), width=4,
        )
        # Flat brow (minimal arch, stays near normal position)
        ecx_f = cx + flat_sign * eye_sep
        draw.arc(
            [ecx_f - int(bw * 0.9), brow_y - int(bh * 0.3),
             ecx_f + int(bw * 0.9), brow_y + int(bh * 0.5)],
            start=215, end=325, fill=(25, 17, 8), width=3,
        )

    elif emotion == "thoughtful":
        pulse     = 0.5 + 0.5 * math.sin(2 * math.pi * 0.28 * t + phase)
        furrow    = int(5 * pulse)
        # Brows drawn with inner ends angled down — "thinking" look
        for sign in (-1, 1):
            ecx      = cx + sign * eye_sep
            inner_x  = ecx - sign * int(bw * 0.8)   # towards nose
            outer_x  = ecx + sign * int(bw * 0.8)   # towards temple
            inner_y  = brow_y + furrow               # inner end dips down
            outer_y  = brow_y - int(fh * 0.04)       # outer end stays up
            draw.line(
                [inner_x, inner_y, outer_x, outer_y],
                fill=(28, 18, 8), width=3,
            )


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


def _sway_offset(
    t: float, char_idx: int,
    is_speaking: bool,
    emotion: str = "neutral",
) -> tuple[int, int]:
    """
    Gentle X/Y drift — different frequencies on each axis so it looks organic.
    Emotion amplifies or shapes the sway for the listener.
    """
    phase = _CHAR_PHASE[char_idx % 2]

    if is_speaking:
        amp_x, amp_y = 5, 3
    elif emotion == "excited":
        # Lean in eagerly — bigger forward sway
        amp_x, amp_y = 6, 4
    elif emotion == "surprised":
        # Jitter / flinch — small rapid tremor on X
        jitter = int(3 * math.sin(2 * math.pi * 3.5 * t + phase))
        return jitter, int(2 * math.sin(2 * math.pi * 2.8 * t + phase + 0.7))
    elif emotion == "curious":
        # Tilt head to one side — asymmetric dx bias
        tilt = 4                           # constant slight lean
        dx = tilt + int(2 * math.sin(2 * math.pi * 0.15 * t + phase))
        dy = int(3 * math.sin(2 * math.pi * 0.10 * t + phase + 1.1))
        return dx * (1 if char_idx % 2 == 0 else -1), dy
    elif emotion == "thoughtful":
        amp_x, amp_y = 2, 2              # slower, more still
    else:
        amp_x, amp_y = 3, 2

    dx = amp_x * math.sin(2 * math.pi * 0.17 * t + phase)
    dy = amp_y * math.sin(2 * math.pi * 0.11 * t + phase + 1.1)
    return int(dx), int(dy)


def _listening_brightness(img: Image.Image, t: float, char_idx: int,
                          max_boost: float = 0.04) -> Image.Image:
    """
    Gently pulse the brightness of the listener's face image — a 3–4 %
    increase/decrease following a slow heartbeat rhythm.  Imperceptible as
    animation but removes the 'frozen statue' feeling.

    Works on any Image mode; returns an RGB image.
    """
    phase  = _CHAR_PHASE[char_idx % 2]
    # Slow dual-frequency heartbeat: 0.9 Hz carrier, 0.13 Hz envelope
    pulse  = (math.sin(2 * math.pi * 0.9 * t + phase) * 0.6 +
               math.sin(2 * math.pi * 0.13 * t + phase * 0.7) * 0.4)
    boost  = 1.0 + max_boost * pulse          # [0.96 … 1.04]
    arr    = np.clip(np.array(img.convert("RGB"), dtype=np.float32) * boost,
                     0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_attention_glow(canvas: Image.Image,
                         cx: int, cy: int,
                         fw: int, fh: int,
                         t: float,
                         color: tuple = (80, 160, 255),
                         max_alpha: int = 38) -> None:
    """
    Draw a soft colored halo AROUND the listener's head — in the background
    space, not touching face pixels.  The glow pulses slowly and never covers
    the actual face.

    The halo is an outer ellipse minus an inner ellipse (donut shape) so
    it only lights up the area around the head silhouette.
    """
    phase     = _CHAR_PHASE[0]   # constant slow pulse
    pulse     = 0.5 + 0.5 * math.sin(2 * math.pi * 0.30 * t + phase)
    alpha     = int(max_alpha * pulse)
    if alpha < 2:
        return

    r, g, b   = color
    outer_rx  = int(fw * 0.72)
    outer_ry  = int(fh * 0.72)
    inner_rx  = int(fw * 0.52)   # inner edge sits just outside the face
    inner_ry  = int(fh * 0.52)

    gl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(gl)
    # Outer ellipse (solid fill)
    gd.ellipse([cx - outer_rx, cy - outer_ry,
                cx + outer_rx, cy + outer_ry],
               fill=(r, g, b, alpha))
    # Inner ellipse (punch hole — transparent)
    gd.ellipse([cx - inner_rx, cy - inner_ry,
                cx + inner_rx, cy + inner_ry],
               fill=(0, 0, 0, 0))
    # Blur for soft glow feel
    gl = gl.filter(ImageFilter.GaussianBlur(int(min(fw, fh) * 0.12)))
    canvas.alpha_composite(gl)


def _draw_scene_attention_glow(frame: Image.Image,
                                cx: int, cy: int,
                                fw: int, fh: int,
                                t: float,
                                color: tuple = (80, 160, 255)) -> Image.Image:
    """
    Same halo but composited on a plain RGB frame (for SingleSceneComposer).
    """
    canvas = frame.convert("RGBA")
    _draw_attention_glow(canvas, cx, cy, fw, fh, t, color)
    return canvas.convert("RGB")


def _draw_listening_hand(canvas: Image.Image,
                          panel_x0: int, panel_w: int, char_h: int,
                          t: float, char_idx: int,
                          skin: tuple = (210, 170, 120)) -> None:
    """
    Draw a small "open-palm / raised hand" icon in the lower-outer corner of
    the listener's panel to signal listening mode.

    Visual design
    ─────────────
    •  Palm: filled rounded ellipse (skin tone)
    •  4 fingers: slim rounded rectangles fanning slightly outward from palm top
    •  Thumb: shorter angled rectangle on the side
    •  Faint drop-shadow for depth
    •  Whole icon scales in/out with a slow 0.28 Hz breath pulse
    •  Icon sits in the bottom outer corner so it never overlaps the face

    Parameters
    ──────────
    canvas   : RGBA image to draw onto
    panel_x0 : left edge of this character's panel (px)
    panel_w  : width of one panel (px)
    char_h   : height of the character area (px)
    t        : current time (seconds)
    char_idx : 0 or 1 — used for phase offset so the two hands pulse independently
    skin     : base skin RGB tuple
    """
    phase  = _CHAR_PHASE[char_idx % 2]
    # Slow breath pulse 0.28 Hz: scale between 0.88 … 1.12
    pulse  = 0.5 + 0.5 * math.sin(2 * math.pi * 0.28 * t + phase)
    scale  = 0.88 + 0.24 * pulse            # 0.88 → 1.12

    # ── size & position ───────────────────────────────────────────────────────
    # Icon size is proportional to panel width (≈ 11 %)
    base_size = max(28, int(panel_w * 0.11))
    sz  = int(base_size * scale)            # scaled palm half-size (radius)
    hw  = max(1, int(sz * 0.55))            # palm half-width
    hh  = max(1, int(sz * 0.65))            # palm half-height

    # Position: bottom-outer corner, slight inset
    margin = max(8, int(panel_w * 0.06))
    # Outer edge = right side for char 0 (left panel), left side for char 1
    if char_idx == 0:
        cx = panel_x0 + panel_w - margin - hw  # right side of left panel
    else:
        cx = panel_x0 + margin + hw             # left side of right panel
    cy = char_h - margin - hh - int(sz * 1.6)  # above the name tag area

    # ── helper: draw rounded rect on a draw object ────────────────────────────
    def _rrect(draw_obj, x0, y0, x1, y1, r, fill):
        r = max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
        draw_obj.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill)

    # ── alpha (opacity) follows pulse, fully visible at peak ──────────────────
    alpha_base = 200
    alpha = max(60, int(alpha_base * (0.6 + 0.4 * pulse)))

    # ── draw shadow ───────────────────────────────────────────────────────────
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_layer)
    shx, shy = 2, 3
    sd.ellipse([cx - hw + shx, cy - hh + shy,
                cx + hw + shx, cy + hh + shy],
               fill=(0, 0, 0, 60))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(3))
    canvas.alpha_composite(shadow_layer)

    # ── draw on a fresh RGBA layer so we can alpha-composite cleanly ──────────
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d     = ImageDraw.Draw(layer)

    sr, sg, sb = skin
    # Slightly darker shading for side of palm
    shade = (max(0, sr - 25), max(0, sg - 20), max(0, sb - 15), alpha)
    base_fill = (sr, sg, sb, alpha)

    # ── Palm ellipse ──────────────────────────────────────────────────────────
    d.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], fill=base_fill)

    # ── Fingers (4): fan out from top of palm ─────────────────────────────────
    # Finger widths/heights relative to sz
    fw_f = max(3, int(sz * 0.18))  # finger width
    fh_f = max(6, int(sz * 0.85))  # finger height (tall)

    # Each finger has (dx_from_cx, height_multiplier, angle_offset_pixels)
    fingers = [
        (-int(sz * 0.36), 0.75, -int(sz * 0.10)),   # index (leftmost)
        (-int(sz * 0.12), 1.00,  0               ),   # middle (tallest)
        ( int(sz * 0.12), 0.95,  0               ),   # ring
        ( int(sz * 0.36), 0.70,  int(sz * 0.08) ),   # pinky (rightmost)
    ]
    for (fdx, fheight_mul, fan) in fingers:
        fx = cx + fdx
        fh_this = max(4, int(fh_f * fheight_mul))
        # Finger base sits at top of palm; fan shifts the bottom slightly
        fy_top    = cy - hh - fh_this + max(1, int(sz * 0.20))
        fy_bottom = cy - hh + max(1, int(sz * 0.20))
        _rrect(d,
               fx - fw_f // 2 + fan,
               fy_top,
               fx + fw_f // 2 + fan,
               fy_bottom,
               fw_f // 2,
               base_fill)

    # ── Thumb: angled, on the outer edge of the palm ─────────────────────────
    tw = max(3, int(sz * 0.19))
    th = max(5, int(sz * 0.55))
    # Thumb points diagonally outward from the palm side
    if char_idx == 0:
        tx = cx + hw - tw // 2          # right side (away from panel centre)
        t_angle_x = int(sz * 0.18)
    else:
        tx = cx - hw + tw // 2          # left side
        t_angle_x = -int(sz * 0.18)
    ty_top    = cy - int(sz * 0.35)
    ty_bottom = cy + int(sz * 0.30)
    _rrect(d,
           tx - tw // 2 + t_angle_x,
           ty_top,
           tx + tw // 2 + t_angle_x,
           ty_bottom,
           tw // 2,
           shade)

    # ── Knuckle line: subtle arc across the top of the palm ──────────────────
    knuckle_alpha = max(20, int(80 * pulse))
    d.arc([cx - hw + 2, cy - hh + 2, cx + hw - 2, cy + int(sz * 0.10)],
          start=200, end=340,
          fill=(max(0, sr - 40), max(0, sg - 35), max(0, sb - 25), knuckle_alpha),
          width=max(1, int(sz * 0.06)))

    canvas.alpha_composite(layer)


def _draw_scene_listening_hand(frame: "np.ndarray",
                                 lx1: int, ly1: int, lx2: int, ly2: int,
                                 t: float, char_idx: int,
                                 skin: tuple = (210, 170, 120)) -> Image.Image:
    """
    Draw the listening-hand icon onto a real-photo RGB frame for
    SingleSceneComposer.  The hand is placed in the lower-outer corner of
    the listener's bounding box.

    Parameters
    ──────────
    frame    : RGB PIL Image (or numpy array) of the scene
    lx1,ly1  : top-left of listener face bbox
    lx2,ly2  : bottom-right of listener face bbox
    """
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = frame.copy()

    canvas = img.convert("RGBA")
    fw_box = lx2 - lx1
    fh_box = ly2 - ly1

    # Treat the bbox as a mini-panel: place hand in its lower-outer corner
    _draw_listening_hand(
        canvas,
        panel_x0 = lx1,
        panel_w  = fw_box,
        char_h   = ly2,          # bottom of bbox = char_h equivalent
        t        = t,
        char_idx = char_idx,
        skin     = skin,
    )
    return canvas.convert("RGB")


def _nod_curve(p: float) -> float:
    """
    Asymmetric nod shape mapped to normalised cycle phase p ∈ [0, 1].
    Returns value ∈ [-1, 0]:  -1 = chin fully down,  0 = head upright.

    Timing breakdown:
      0.00 – 0.15  pause at neutral   (head resting before nod)
      0.15 – 0.45  quick chin-down    (fast lean, the "nod" motion)
      0.45 – 0.60  hold at bottom     (brief pause at lowest point)
      0.60 – 0.80  smooth return up   (ease-out back to neutral)
      0.80 – 1.00  pause at neutral   (rest before next nod)
    """
    if p < 0.15:
        return 0.0                                        # rest
    if p < 0.45:
        q = (p - 0.15) / 0.30
        return -math.sin(q * math.pi * 0.5)              # ease-in down
    if p < 0.60:
        return -1.0                                       # hold at bottom
    if p < 0.80:
        q = (p - 0.60) / 0.20
        return -(1.0 - math.sin(q * math.pi * 0.5))      # ease-out up
    return 0.0                                            # rest


def _apply_nod_shift(
    scene: Image.Image,
    face_bbox: tuple,       # (lx1, ly1, lx2, ly2) padded face region
    shift_y: int,           # pixels downward (positive = head nods down)
    frame: Image.Image,     # current composited frame
) -> Image.Image:
    """
    Simulate a natural head nod with a PURE VERTICAL shift — no rotation,
    no lateral movement.

    Why shift, not rotate:
    ──────────────────────
    PIL rotate() works in 2D.  Rotating around a below-face pivot sweeps the
    face in an ARC — left or right depending on pivot position.  That is NOT
    what a nod looks like from the front.

    A front-view nod is purely vertical: the face moves slightly downward as
    the chin tips toward the chest.  The correct 2D simulation is:

      1. Sample the original scene at a HIGHER position (by shift_y pixels).
      2. Blend that shifted content over the face area at the original position.
      3. Use an extra-wide, heavily-blurred elliptical mask so the seam between
         the shifted region and the static background is invisible.

    Sampling higher = the face content appears to have moved down = nod.
    No rotation involved → zero lateral drift.
    """
    if abs(shift_y) < 1:
        return frame

    lx1, ly1, lx2, ly2 = face_bbox
    l_fw = lx2 - lx1
    l_fh = ly2 - ly1
    img_w, img_h = scene.size

    # ── Generous padded region around the face ────────────────────────────────
    pad_x = int(l_fw * 0.65)
    pad_y = int(l_fh * 0.55)
    bx1 = max(0, lx1 - pad_x)
    by1 = max(0, ly1 - pad_y)
    bx2 = min(img_w, lx2 + pad_x)
    by2 = min(img_h, ly2 + pad_y)
    bw  = bx2 - bx1
    bh  = by2 - by1

    # ── Build the "nod" frame: paste face crop shifted DOWN by shift_y ─────────
    # Take the original scene content for this region (unshifted face)
    base_crop = scene.crop((bx1, by1, bx2, by2)).convert("RGBA")

    # Shift it down: scroll the image content up by shift_y lines so when
    # painted at the original position the face appears to have moved down.
    shifted = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    # Copy rows [0 .. bh-shift_y] of the original crop into rows [shift_y .. bh]
    keep = base_crop.crop((0, 0, bw, max(1, bh - shift_y)))
    shifted.paste(keep, (0, shift_y))

    # ── Soft face-centred elliptical mask (very heavy blur = invisible seam) ──
    face_cx = (lx1 + lx2) // 2 - bx1
    face_cy = (ly1 + ly2) // 2 - by1
    half_w  = int(l_fw * 0.58)
    half_h  = int(l_fh * 0.60)
    blur_r  = int(min(l_fw, l_fh) * 0.30)   # enough to hide seam, not kill motion

    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).ellipse(
        [face_cx - half_w, face_cy - half_h,
         face_cx + half_w, face_cy + half_h],
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(blur_r))
    shifted.putalpha(mask)

    # ── Composite ─────────────────────────────────────────────────────────────
    out = frame.convert("RGBA")
    out.paste(shifted, (bx1, by1), shifted)
    return out.convert("RGB")


def _listener_nod(t: float, char_idx: int, emotion: str = "neutral") -> int:
    """
    Returns the downward pixel shift for the listener's face at time *t*.
    Positive = face content shifts downward (chin-down nod).

    Uses a discrete nod pulse so each nod is clearly visible.
    Small irregularity prevents the rhythm feeling mechanical.
    """
    phase = _CHAR_PHASE[char_idx % 2]
    irr   = 1.0 + 0.10 * math.sin(2 * math.pi * 0.09 * t + phase * 1.7)

    if emotion == "excited":
        freq, amp = 1.00, 14
    elif emotion == "surprised":
        freq, amp = 0.90, 11
    elif emotion == "thoughtful":
        freq, amp = 0.50, 12
    elif emotion == "happy":
        freq, amp = 0.75, 11
    else:
        freq, amp = 0.65, 10      # neutral — visible, not exaggerated

    p   = (t * freq + phase / (2 * math.pi)) % 1.0
    raw = _nod_curve(p) * amp * irr   # negative = shift upward in scene = face moves down
    return int(-raw)                  # positive int = sample higher = face appears lower


def _blink_alpha(t: float, char_idx: int) -> float:
    """
    High-quality blink model:
    - Randomised interval (3.5 – 6.5 s) so it never feels mechanical.
    - 130 ms total duration: fast-snap close (35%), micro-hold (20%),
      slow ease-open (45%) — matches the actual physiology of a blink.
    - Occasional double-blink: a second quick blink ~350 ms after the first.
    - Each character has a different base period and phase offset so they
      never blink in sync.

    Returns 0.0 (fully open) → 1.0 (fully closed).
    """
    # Per-character deterministic seed for pseudo-random interval jitter
    import hashlib
    seed_base = char_idx * 7919          # large prime keeps per-char distinct

    eye_y   = 0                          # unused here, kept for symmetry
    base    = 4.0 + char_idx * 1.7      # 4.0 s  /  5.7 s
    phase   = _CHAR_PHASE[char_idx % 2]

    # Build a deterministic jitter per blink cycle using integer cycle index
    # so the same t always produces the same result (no hidden state needed).
    # Jitter range: ±1.2 s
    def _jitter(cycle: int) -> float:
        h = int(hashlib.md5(f"{seed_base}_{cycle}".encode()).hexdigest()[:8], 16)
        return ((h % 2400) - 1200) / 1000.0   # ±1.2 s

    # Walk forward in time to find which blink cycle we are in
    t_shifted = t + phase * base / (2 * math.pi)
    cycle     = 0
    t_acc     = 0.0
    SINGLE_DUR = 0.130          # 130 ms primary blink
    DOUBLE_GAP = 0.350          # gap before possible double-blink
    DOUBLE_DUR = 0.095          # 95 ms for the quicker second blink
    DOUBLE_PROB_SEED = 3        # every 3rd cycle is a double-blink

    while True:
        period = max(2.5, base + _jitter(cycle))
        if t_acc + period > t_shifted:
            break
        t_acc += period
        cycle += 1

    t_in_cycle = t_shifted - t_acc      # time within current cycle [0, period)

    def _eyelid(p: float) -> float:
        """Normalised progress p ∈ [0,1] → lid closure 0→1→0."""
        if p < 0.35:  return p / 0.35                        # fast snap close
        if p < 0.55:  return 1.0                             # micro-hold
        return 1.0 - ((p - 0.55) / 0.45) ** 0.6             # slow ease open

    # Primary blink
    if t_in_cycle < SINGLE_DUR:
        return _eyelid(t_in_cycle / SINGLE_DUR)

    # Double-blink (every DOUBLE_PROB_SEED-th cycle)
    if cycle % DOUBLE_PROB_SEED == 0:
        t2 = t_in_cycle - SINGLE_DUR - DOUBLE_GAP
        if 0 <= t2 < DOUBLE_DUR:
            return _eyelid(t2 / DOUBLE_DUR) * 0.85   # slightly softer second blink

    return 0.0


def _sample_skin_colour(canvas: Image.Image, cx: int, cy: int,
                        fw: int, fh: int) -> tuple[int, int, int]:
    """
    Sample the average pixel colour from the forehead/cheek area of the portrait
    so the eyelid overlay automatically matches the character's skin tone.
    Falls back to a neutral value if the sample region is out of bounds.
    """
    # Sample a small patch from the forehead (above the eye region)
    sx = max(0, cx - fw // 6)
    sy = max(0, cy - int(fh * 0.32))
    ex = min(canvas.width,  cx + fw // 6)
    ey = min(canvas.height, cy - int(fh * 0.18))
    if ex <= sx or ey <= sy:
        return (210, 175, 140)
    try:
        patch = canvas.convert("RGB").crop((sx, sy, ex, ey))
        arr   = np.array(patch, dtype=np.float32)
        r, g, b = int(arr[:,:,0].mean()), int(arr[:,:,1].mean()), int(arr[:,:,2].mean())
        # Lighten slightly — eyelid skin is often a little paler than the cheek
        r = min(255, r + 18)
        g = min(255, g + 12)
        b = min(255, b + 8)
        return (r, g, b)
    except Exception:
        return (210, 175, 140)


def _draw_eye_blink(
    canvas: Image.Image,
    cx: int, cy: int,
    fw: int, fh: int,
    alpha: float,
) -> None:
    """
    High-quality eyelid closure overlay:
    - Skin colour sampled from the actual portrait (no hardcoded tone).
    - Upper lid travels 75 % of the eye height; lower lid 25 % — anatomically
      correct (blinks are mostly upper-lid movement).
    - Vertical gradient on the upper lid: darker near the lash line (shadow),
      lighter toward the brow.
    - Thin dark lash-line arc at the bottom of the upper lid when >60 % closed.
    - Soft horizontal feathering on the edges so the overlay blends seamlessly.
    """
    if alpha < 0.04:
        return

    skin        = _sample_skin_colour(canvas, cx, cy, fw, fh)
    eye_y       = cy - int(fh * 0.12)    # vertical centre of the eye row
    eye_sep     = int(fw * 0.20)         # half-distance between eye centres
    eye_rx      = int(fw * 0.115)        # half-width of each eye
    eye_ry      = int(fh * 0.068)        # half-height of each eye opening

    bl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bl)

    for side in (-1, 1):
        ex = cx + side * eye_sep

        # ── Upper eyelid (75 % of alpha travel) ──────────────────────────────
        upper_drop = int(eye_ry * 2 * alpha * 0.75)  # pixels the lid descends
        if upper_drop > 0:
            # Gradient: draw horizontal slices from top (lighter) to bottom (darker)
            for row in range(upper_drop):
                frac     = row / max(1, upper_drop)
                # Darken near the lash line
                shade    = int(255 * (1.0 - frac * 0.30))
                lid_r    = min(255, int(skin[0] * shade / 255))
                lid_g    = min(255, int(skin[1] * shade / 255))
                lid_b    = min(255, int(skin[2] * shade / 255))
                # Horizontal feathering: full opacity in centre, fade at edges
                row_y    = eye_y - eye_ry + row
                # clip to ellipse width at this row
                ry_frac  = abs(row - eye_ry) / max(1, eye_ry)
                row_rx   = max(1, int(eye_rx * math.sqrt(max(0.0, 1 - ry_frac ** 2))))
                a_row    = int(255 * alpha * min(1.0, (upper_drop - row) / max(1, upper_drop * 0.15) ))
                a_row    = min(int(255 * alpha), a_row)
                bd.line(
                    [(ex - row_rx, row_y), (ex + row_rx, row_y)],
                    fill=(lid_r, lid_g, lid_b, a_row),
                )

        # ── Lower eyelid (25 % travel upward) ────────────────────────────────
        lower_rise = int(eye_ry * 2 * alpha * 0.25)
        if lower_rise > 0:
            for row in range(lower_rise):
                frac  = row / max(1, lower_rise)
                shade = int(255 * (0.92 + frac * 0.05))
                lid_r = min(255, int(skin[0] * shade / 255))
                lid_g = min(255, int(skin[1] * shade / 255))
                lid_b = min(255, int(skin[2] * shade / 255))
                row_y = eye_y + eye_ry - row
                ry_frac  = abs(eye_ry - row) / max(1, eye_ry)
                row_rx   = max(1, int(eye_rx * math.sqrt(max(0.0, 1 - ry_frac ** 2))))
                a_row    = int(255 * alpha * 0.85)
                bd.line(
                    [(ex - row_rx, row_y), (ex + row_rx, row_y)],
                    fill=(lid_r, lid_g, lid_b, a_row),
                )

        # ── Lash-line shadow arc (only when mostly closed) ───────────────────
        if alpha > 0.55:
            lash_y   = eye_y - eye_ry + int(eye_ry * 2 * alpha * 0.75)
            lash_a   = int(180 * min(1.0, (alpha - 0.55) / 0.35))
            lash_th  = max(1, int(fh * 0.008))
            bd.arc(
                [ex - eye_rx, lash_y - lash_th * 2,
                 ex + eye_rx, lash_y + lash_th * 2],
                start=190, end=350,
                fill=(30, 20, 15, lash_a),
                width=lash_th,
            )

    canvas.alpha_composite(bl)


def _draw_gaze_toward_speaker(
    canvas: Image.Image,
    cx: int, cy: int,
    fw: int, fh: int,
    listener_idx: int,
    t: float,
) -> None:
    """
    Subtle dark iris shift to make the listener's eyes appear to look toward
    the speaker (inward).  Left-panel listener (idx 0) looks right; right-panel
    listener (idx 1) looks left.
    """
    eye_y   = cy - int(fh * 0.12)
    eye_sep = int(fw * 0.20)
    iris_r  = max(2, int(fw * 0.055))
    # Gentle saccade: small slow drift toward center + tiny oscillation
    phase   = _CHAR_PHASE[listener_idx % 2]
    drift   = int(iris_r * 0.45) * (1 if listener_idx % 2 == 0 else -1)
    jitter  = int(iris_r * 0.15 * math.sin(2 * math.pi * 0.18 * t + phase))
    shift_x = drift + jitter

    gl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(gl)
    for side in (-1, 1):
        ex = cx + side * eye_sep + shift_x
        # Iris
        gd.ellipse([ex - iris_r, eye_y - iris_r,
                    ex + iris_r, eye_y + iris_r],
                   fill=(55, 80, 50, 200))
        # Pupil
        pr = max(1, int(iris_r * 0.55))
        gd.ellipse([ex - pr, eye_y - pr,
                    ex + pr, eye_y + pr],
                   fill=(12, 8, 6, 220))
        # Catchlight
        cl = max(1, int(iris_r * 0.25))
        gd.ellipse([ex - iris_r + cl, eye_y - iris_r + cl,
                    ex - iris_r + cl * 3, eye_y - iris_r + cl * 3],
                   fill=(255, 255, 255, 160))
    canvas.alpha_composite(gl)


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
# Split-screen studio helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_panel_bg(panel_w: int, panel_h: int, char_idx: int) -> Image.Image:
    """
    Studio-quality gradient background for one character panel.
    Each character gets a distinct warm/cool neutral tone so the panels
    are visually separate without being jarring.
    """
    # Char 0 → cool blue-slate studio  |  Char 1 → warm charcoal studio
    if char_idx % 2 == 0:
        dark  = np.array([22,  28,  42], dtype=np.float32)   # deep blue-slate
        mid   = np.array([48,  58,  82], dtype=np.float32)   # lighter centre
        rim   = np.array([80, 100, 140], dtype=np.float32)   # subtle rim-light
    else:
        dark  = np.array([30,  22,  18], dtype=np.float32)   # warm charcoal
        mid   = np.array([68,  52,  44], dtype=np.float32)
        rim   = np.array([120, 90,  70], dtype=np.float32)

    xs = np.linspace(-1.0,  1.0, panel_w, dtype=np.float32)[None, :]   # (1, W)
    ys = np.linspace( 0.0,  1.0, panel_h, dtype=np.float32)[:, None]   # (H, 1)

    # Radial weight centred at upper-middle (face will live here)
    radial = np.sqrt(xs ** 2 + (ys - 0.35) ** 2)          # (H, W)
    t_c    = np.clip(radial / 1.1, 0.0, 1.0)[:, :, None]  # centre falloff
    # Subtle vertical rim-light strip on the outer edge of each panel
    if char_idx % 2 == 0:
        rim_x = np.clip((xs + 1.0) / 0.18, 0.0, 1.0)     # left edge
    else:
        rim_x = np.clip((1.0 - xs) / 0.18, 0.0, 1.0)     # right edge
    rim_w  = rim_x[:, :, None] * 0.35

    colour = (mid[None, None, :] * (1 - t_c) +
              dark[None, None, :] * t_c +
              rim[None, None, :]  * rim_w)
    colour = np.clip(colour, 0, 255).astype(np.uint8)
    return Image.fromarray(colour, mode="RGB")


def _blend_panel_face(
    canvas: Image.Image,
    face_img: Image.Image,
    panel_x0: int,
    cy: int,
    fw: int,
    fh: int,
) -> None:
    """
    Composite a portrait into its panel.

    Unlike the old oval-mask approach, the portrait fills the full panel width
    so the character looks like a real person in frame.  Only the top and
    bottom edges are feathered (natural vignette) — the left/right panel
    boundaries act as the natural crop.
    """
    face = face_img.convert("RGBA").resize((fw, fh), Image.LANCZOS)

    # Vertical alpha ramp: fade top 8 % and bottom 14 % of portrait
    arr   = np.array(face, dtype=np.uint8)
    alpha = arr[:, :, 3].astype(np.float32)

    fade_top = max(1, int(fh * 0.08))
    fade_bot = max(1, int(fh * 0.14))
    for i in range(fade_top):
        alpha[i, :] *= i / fade_top
    for i in range(fade_bot):
        alpha[fh - 1 - i, :] *= i / fade_bot

    arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    face = Image.fromarray(arr)

    # Paste position (clip to canvas bounds)
    px = panel_x0 + (fw == canvas.width // 2 and 0 or (canvas.width // 2 - fw) // 2)
    # Centre the portrait horizontally within the panel
    px = panel_x0 + max(0, (canvas.width // 2 - fw) // 2)
    py = cy - fh // 2

    # Clamp so we never go outside the canvas
    px = max(0, min(px, canvas.width  - 1))
    py = max(0, min(py, canvas.height - 1))
    canvas.alpha_composite(face, (px, py))


def _draw_speaker_border(
    canvas: Image.Image,
    panel_x0: int,
    panel_w: int,
    char_h: int,
    t: float,
    char_idx: int,
) -> None:
    """
    Pulsing coloured border around the active speaker's panel.
    Colour is unique per character so the audience always knows who is talking.
    """
    colours = [
        (80, 160, 255),   # char 0 → cool blue
        (255, 140,  80),  # char 1 → warm orange
    ]
    base_col = colours[char_idx % len(colours)]
    pulse    = 0.65 + 0.35 * math.sin(2 * math.pi * 1.2 * t + _CHAR_PHASE[char_idx % 2])
    alpha    = int(200 * pulse)

    border_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(border_layer)
    bw = 5   # border thickness
    bd.rectangle(
        [panel_x0 + bw, bw, panel_x0 + panel_w - bw, char_h - bw],
        outline=(*base_col, alpha),
        width=bw,
    )
    canvas.alpha_composite(border_layer)


def _draw_name_tag(
    canvas: Image.Image,
    panel_x0: int,
    panel_w: int,
    char_h: int,
    name: str,
    is_speaker: bool,
    char_idx: int,
) -> None:
    """
    Floating name tag at the bottom of each character's panel.
    Active speaker's tag is brighter / coloured; listener is muted.
    """
    tag_h = max(28, int(char_h * 0.07))
    tag_w = min(panel_w - 40, max(120, int(panel_w * 0.55)))
    tag_x = panel_x0 + (panel_w - tag_w) // 2
    tag_y = char_h - tag_h - int(char_h * 0.04)

    tag = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    td  = ImageDraw.Draw(tag)

    if is_speaker:
        bg_col  = (20, 20, 20, 200)
        border  = [(80, 160, 255, 230), (255, 140, 80, 230)][char_idx % 2]
        txt_col = (255, 255, 255, 255)
    else:
        bg_col  = (15, 15, 15, 140)
        border  = (80, 80, 80, 160)
        txt_col = (180, 180, 180, 200)

    td.rounded_rectangle(
        [tag_x, tag_y, tag_x + tag_w, tag_y + tag_h],
        radius=6, fill=bg_col, outline=border, width=2,
    )

    # Name text — attempt to use a font, fall back to default
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  max(12, tag_h - 10))
    except Exception:
        font = None

    td.text(
        (tag_x + tag_w // 2, tag_y + tag_h // 2),
        name.upper(),
        fill=txt_col,
        font=font,
        anchor="mm" if font else None,
    )
    canvas.alpha_composite(tag)


def _draw_divider(canvas: Image.Image, char_h: int) -> None:
    """Thin vertical divider line between the two panels."""
    mid_x  = canvas.width // 2
    div    = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    dd     = ImageDraw.Draw(div)
    dd.line([(mid_x, 0), (mid_x, char_h)], fill=(255, 255, 255, 40), width=2)
    canvas.alpha_composite(div)


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
        room_bg_path: str | None = None,       # kept for API compat (unused)
        room_bg_image: Image.Image | None = None,   # kept for API compat (unused)
        char_body_paths: dict | None = None,   # kept for API compat (unused)
        char_names: list[str] | None = None,   # display names for name tags
    ):
        self.width       = width
        self.height      = height
        self.fps         = fps
        self.char_names  = char_names or ["Alice", "Bob"]
        self.char_h      = int(height * (1 - self.SUBTITLE_RATIO))
        self.sub_h       = height - self.char_h

        # ── Split-screen layout ───────────────────────────────────────────────
        # Each character owns exactly half the horizontal space.
        self.panel_w = width // 2

        # Portrait fills ~88 % of panel width (natural, not tiny)
        # Taller aspect ratio (1.45) gives a head-and-shoulders bust crop
        self.fw = int(self.panel_w * 0.88)
        self.fh = int(self.fw * 1.45)

        # Panel x-offsets (left edge of each panel)
        self.panel_x = [0, self.panel_w]

        # Character centres: horizontal = panel mid, vertical = upper half
        cx_l = self.panel_w // 2
        cx_r = self.panel_w + self.panel_w // 2
        cy   = int(self.char_h * 0.46)   # head sits in upper portion
        self.centres = [(cx_l, cy), (cx_r, cy)]

        # Pre-render studio panel backgrounds (expensive; done once)
        self._panel_bgs = [
            _make_panel_bg(self.panel_w, self.char_h, 0),
            _make_panel_bg(self.panel_w, self.char_h, 1),
        ]

    # ── build one composite PIL image (room + both faces + animation) ─────────

    def _build_frame(
        self,
        t: float,
        room_bg: Image.Image,            # unused in split-screen mode
        listener_face: Image.Image,
        listener_idx: int,
        speaker_frame: np.ndarray,
        speaker_idx: int,
        glow: Image.Image,               # unused in split-screen mode
        subtitle: Image.Image,
        emotion: str = "neutral",
    ) -> np.ndarray:
        """
        Split-screen podcast compositing.

        Layout
        ──────
          ┌──────────────┬──────────────┐
          │  Char 0      │  Char 1      │  85 %  char area
          │  (full panel │  (full panel │
          │   portrait)  │   portrait)  │
          ├──────────────┴──────────────┤
          │      Subtitle bar           │  15 %
          └─────────────────────────────┘

        Each character fills their entire panel (natural talking-head look).
        The active speaker gets a coloured pulsing border.
        Listener shows emotion expressions + animated head movement.
        """
        # ── Base: two studio panel backgrounds side by side ───────────────────
        canvas = Image.new("RGBA", (self.width, self.char_h))
        canvas.paste(self._panel_bgs[0].convert("RGBA"), (0, 0))
        canvas.paste(self._panel_bgs[1].convert("RGBA"), (self.panel_w, 0))

        # ── Per-character face rendering ──────────────────────────────────────
        # No horizontal flipping is done here.  The portrait prompts in
        # generate_faces.py / main.py bake the correct inward-facing direction
        # directly into the SD-generated image:
        #   char 0 (left panel)  → prompted "facing slightly right"
        #   char 1 (right panel) → prompted "facing slightly left"
        # Flipping in the compositor would fight that and reverse the effect.

        for char_idx in (listener_idx, speaker_idx):
            is_spk   = (char_idx == speaker_idx)
            cx, cy   = self.centres[char_idx]
            px0      = self.panel_x[char_idx]

            if is_spk:
                # Speaker: Wav2Lip animated frame
                scale    = _breathing_scale(t, char_idx, is_speaking=True)
                sdx, sdy = _sway_offset(t, char_idx, is_speaking=True, emotion=emotion)
                sw = max(1, int(self.fw * scale))
                sh = max(1, int(self.fh * scale))
                face_img = Image.fromarray(speaker_frame).resize((sw, sh), Image.LANCZOS)
                _blend_panel_face(canvas, face_img, px0, cy + sdy, sw, sh)
            else:
                # Listener: nod (y-position shift) + brightness pulse + glow
                nod_dy           = _listener_nod(t, char_idx, emotion=emotion)
                l_face, ldx, ldy = _animate_face(
                    listener_face, self.fw, self.fh, t, char_idx, is_speaking=False
                )
                ldx2, ldy2 = _sway_offset(t, char_idx, is_speaking=False, emotion=emotion)
                # Subtle brightness pulse — removes the "frozen" look
                l_face = _listening_brightness(l_face, t, char_idx)
                face_cy = cy + ldy2 + nod_dy
                _blend_panel_face(canvas, l_face, px0, face_cy,
                                  l_face.width, l_face.height)
                # Soft attention glow around the listener's head
                _draw_attention_glow(canvas, cx=cx + ldx2, cy=face_cy,
                                     fw=self.fw, fh=self.fh, t=t)
                # Listening-hand icon in the lower-outer corner of the panel
                _draw_listening_hand(canvas,
                                     panel_x0=self.panel_x[char_idx],
                                     panel_w=self.panel_w,
                                     char_h=self.char_h,
                                     t=t, char_idx=char_idx)

        # ── Name tags ─────────────────────────────────────────────────────────
        for char_idx in (listener_idx, speaker_idx):
            is_spk = (char_idx == speaker_idx)
            name   = (self.char_names[char_idx]
                      if char_idx < len(self.char_names)
                      else f"Speaker {char_idx + 1}")
            _draw_name_tag(canvas, self.panel_x[char_idx], self.panel_w,
                           self.char_h, name, is_spk, char_idx)

        # ── Panel divider ─────────────────────────────────────────────────────
        _draw_divider(canvas, self.char_h)

        # ── Assemble final frame (char area + subtitle bar) ───────────────────
        full = Image.new("RGB", (self.width, self.height), (8, 8, 8))
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
        # room_bg and glow are no longer used in split-screen mode;
        # pass None placeholders so _build_frame signature stays compatible.
        room_bg_placeholder = None
        glow_placeholder    = None

        listener_face = Image.open(face_image_paths[listener_idx]).convert("RGB")
        subtitle      = _subtitle_img(self.width, self.sub_h, speaker_name, dialogue_text)

        # Detect the emotional tone of this line so the listener reacts
        emotion = _detect_emotion(dialogue_text)
        print(f"  [Composer] Emotion detected: {emotion!r}  ({dialogue_text[:50]}…)")

        def make_frame(t: float) -> np.ndarray:
            safe_t    = max(0.0, min(t, wav2lip_dur - 1.0 / max(1, self.fps)))
            spk_frame = talking.get_frame(safe_t)
            return self._build_frame(
                t, room_bg_placeholder, listener_face, listener_idx,
                spk_frame, speaker_idx, glow_placeholder, subtitle,
                emotion=emotion,
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


# ─────────────────────────────────────────────────────────────────────────────
# Single-scene composer  (both characters in one photo, facing each other)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_scene_faces(scene: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in the scene image using OpenCV Haar cascade.
    Returns bboxes as (x1, y1, x2, y2) sorted left-to-right.
    Falls back to simple half-split if detection fails.
    """
    try:
        import cv2
        img_bgr = cv2.cvtColor(np.array(scene), cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) >= 2:
            bboxes = [(x, y, x + w, y + h) for x, y, w, h in faces]
            bboxes.sort(key=lambda b: b[0])   # left → right
            print(f"  [Scene] Detected {len(bboxes)} faces — using leftmost two.")
            return bboxes[:2]
    except Exception as e:
        print(f"  [Scene] Face detection warning: {e}")

    # Fallback: assume left person in left third, right person in right third
    W, H = scene.size
    fw, fh = int(W * 0.22), int(H * 0.40)
    cy     = int(H * 0.32)
    print("  [Scene] ⚠ Could not detect faces — using approximate positions.")
    return [
        (int(W * 0.10), cy, int(W * 0.10) + fw, cy + fh),
        (int(W * 0.65), cy, int(W * 0.65) + fw, cy + fh),
    ]


def _padded_bbox(bbox: tuple, pad_frac: float,
                 img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Expand bbox by pad_frac of its size, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    pad = int(max(x2 - x1, y2 - y1) * pad_frac)
    return (max(0, x1 - pad), max(0, y1 - pad),
            min(img_w, x2 + pad), min(img_h, y2 + pad))


def _blend_face_into_scene(
    scene: Image.Image,
    face_frame: np.ndarray,
    pbbox: tuple[int, int, int, int],
) -> Image.Image:
    """
    Paste an animated Wav2Lip face frame back into *scene* at *pbbox*,
    blending the edges with a soft elliptical mask so it looks seamless.
    """
    x1, y1, x2, y2 = pbbox
    tw, th = x2 - x1, y2 - y1
    face   = Image.fromarray(face_frame).resize((tw, th), Image.LANCZOS)

    # Soft elliptical blend mask
    mask   = Image.new("L", (tw, th), 0)
    md     = ImageDraw.Draw(mask)
    m      = max(4, int(min(tw, th) * 0.10))   # margin for feathering
    md.ellipse([m, m, tw - m, th - m], fill=255)
    mask   = mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(tw, th) * 0.07))))

    result = scene.copy()
    result.paste(face, (x1, y1), mask)
    return result


class SingleSceneComposer:
    """
    Uses ONE scene image containing both characters already posed and facing
    each other.  Only the active speaker's face region is replaced each frame
    with the Wav2Lip-animated version; the listener stays still in the photo.

    This gives the most natural 'two people in conversation' look.
    """

    SUBTITLE_RATIO = 0.15

    def __init__(
        self,
        scene_image_path: str,
        char_names: list[str],
        width: int = 1280,
        height: int = 720,
        fps: int = 25,
    ):
        self.width      = width
        self.height     = height
        self.fps        = fps
        self.char_names = char_names
        self.char_h     = int(height * (1 - self.SUBTITLE_RATIO))
        self.sub_h      = height - self.char_h

        # Load + resize scene to video dimensions
        scene_full  = Image.open(scene_image_path).convert("RGB")
        self.scene  = scene_full.resize((width, self.char_h), Image.LANCZOS)

        # Detect face positions (sorted left-to-right = char 0, char 1)
        raw_bboxes        = _detect_scene_faces(self.scene)
        self.face_bboxes  = raw_bboxes[:2]
        self._padded      = [
            _padded_bbox(bb, 0.45, width, self.char_h)
            for bb in self.face_bboxes
        ]
        for i, (bb, pb) in enumerate(zip(self.face_bboxes, self._padded)):
            name = char_names[i] if i < len(char_names) else f"char{i}"
            print(f"  [Scene] {name} face bbox={bb}  padded={pb}")

    def get_face_crop_path(self, char_idx: int, tmp_dir: str) -> str:
        """
        Save the speaker's face region from the scene as a temp PNG.
        This is passed to run_wav2lip() as the face input.
        """
        x1, y1, x2, y2 = self._padded[char_idx]
        crop      = self.scene.crop((x1, y1, x2, y2))
        crop_path = os.path.join(tmp_dir, f"scene_face_{char_idx}.png")
        crop.save(crop_path)
        return crop_path

    def create_segment(
        self,
        speaker_idx: int,
        dialogue_text: str,
        speaker_name: str,
        wav2lip_video_path: str,
        audio_path: str,
        segment_out_path: str,
    ) -> str:
        """
        Render one dialogue segment.
        The scene stays still; only the speaker's face region animates.
        """
        import subprocess
        VideoFileClip, AudioFileClip, VideoClip, _ = _mp()

        talking     = VideoFileClip(wav2lip_video_path)
        wav2lip_dur = talking.duration

        # Audio duration
        audio_dur = wav2lip_dur
        audio_ok  = os.path.exists(audio_path) and os.path.getsize(audio_path) > 256
        if audio_ok:
            try:
                tmp_a     = AudioFileClip(audio_path)
                audio_dur = tmp_a.duration
                tmp_a.close()
            except Exception:
                audio_ok = False

        duration     = max(audio_dur, 0.5)
        pbbox        = self._padded[speaker_idx]
        listener_idx = 1 - speaker_idx
        subtitle     = _subtitle_img(self.width, self.sub_h, speaker_name, dialogue_text)

        # Listener face centre + approximate size for glow positioning
        lx1, ly1, lx2, ly2 = self._padded[listener_idx]
        l_cx  = (lx1 + lx2) // 2
        l_cy  = (ly1 + ly2) // 2
        l_fw  = lx2 - lx1
        l_fh  = ly2 - ly1
        # Per-character glow colour (warm amber for char 0, cool blue for char 1)
        GLOW_COLORS = [(255, 200, 100), (100, 180, 255)]
        glow_color  = GLOW_COLORS[listener_idx % 2]

        print(f"  [Scene] speaker={speaker_name}  bbox={pbbox}")

        def make_frame(t: float) -> np.ndarray:
            safe_t    = max(0.0, min(t, wav2lip_dur - 1.0 / max(1, self.fps)))
            spk_frame = talking.get_frame(safe_t)

            # ── Speaker: Wav2Lip animated lip region ──────────────────────────
            frame = _blend_face_into_scene(self.scene, spk_frame, pbbox)

            # ── Listener: attention glow around the head (no pixel manipulation)
            # A soft pulsing halo in the background around the listener's head
            # indicates engagement without distorting the real photo.
            frame = _draw_scene_attention_glow(
                frame, l_cx, l_cy, l_fw, l_fh, t, color=glow_color
            )

            # ── Listener: raised-hand icon in lower-outer corner of face bbox
            frame = _draw_scene_listening_hand(
                frame, lx1, ly1, lx2, ly2,
                t=t, char_idx=listener_idx
            )

            full = Image.new("RGB", (self.width, self.height), (8, 8, 8))
            full.paste(frame, (0, 0))
            full.paste(subtitle, (0, self.char_h))
            return np.asarray(full, dtype=np.uint8)

        # Write video-only then mux audio
        tmp_vid = segment_out_path + "_noaudio.mp4"
        clip    = VideoClip(make_frame, duration=duration)
        clip    = clip.with_fps(self.fps) if hasattr(clip, "with_fps") else clip.set_fps(self.fps)
        clip.write_videofile(tmp_vid, fps=self.fps, codec="libx264",
                             audio=False, logger=None)
        talking.close()

        if audio_ok:
            cmd = ["ffmpeg", "-y", "-i", tmp_vid, "-i", audio_path,
                   "-c:v", "copy", "-c:a", "aac", "-ar", "44100",
                   "-ac", "1", "-shortest", segment_out_path]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  [Scene] ✓ Segment: {segment_out_path}")
            else:
                import shutil; shutil.move(tmp_vid, segment_out_path)
        else:
            import shutil; shutil.move(tmp_vid, segment_out_path)

        if os.path.isfile(tmp_vid):
            try: os.unlink(tmp_vid)
            except Exception: pass

        return segment_out_path

    @staticmethod
    def concat_and_write(segment_paths: list[str], output_path: str, fps: int = 25):
        VideoComposer.concat_and_write(segment_paths, output_path, fps)
