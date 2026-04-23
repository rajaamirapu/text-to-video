"""
character_renderer.py

Draws a stylised portrait bust (head + shoulders) using Pillow.

Supported states
----------------
- Listening : neutral expression, mouth closed, slight "attention" gaze
- Talking   : mouth opens according to the lip-sync amplitude (0.0 – 1.0)

Active speaker gets a glowing border + animated sound-bar indicator.
"""

from __future__ import annotations
import math
from typing import Sequence

from PIL import Image, ImageDraw


# ── helpers ──────────────────────────────────────────────────────────────────

def _rgb(seq: Sequence[int]) -> tuple[int, int, int]:
    return (int(seq[0]), int(seq[1]), int(seq[2]))


def _darken(rgb: Sequence[int], amount: int = 30) -> tuple[int, int, int]:
    return tuple(max(0, c - amount) for c in rgb)  # type: ignore[return-value]


def _lighten(rgb: Sequence[int], amount: int = 30) -> tuple[int, int, int]:
    return tuple(min(255, c + amount) for c in rgb)  # type: ignore[return-value]


# ── main class ───────────────────────────────────────────────────────────────

class CharacterRenderer:
    """
    Parameters
    ----------
    name       : displayed name label
    appearance : dict produced by OllamaClient.generate_appearance()
    position   : 'left' or 'right' (affects default colours)
    """

    def __init__(
        self,
        name: str,
        appearance: dict | None = None,
        position: str = "left",
    ):
        self.name = name
        self.position = position

        # ── resolve appearance ───────────────────────────────────────────────
        # Position-based defaults that actually differ by gender, so that e.g.
        # Alice (left / female) does not render identically to Bob (right / male).
        if position == "left":
            defaults = {
                "skin_rgb":   [252, 220, 195],
                "hair_rgb":   [120, 60, 30],
                "eye_rgb":    [70, 130, 180],
                "shirt_rgb":  [180, 80, 110],
                "hair_style": "long",
                "gender":     "female",
            }
        else:
            defaults = {
                "skin_rgb":   [220, 180, 140],
                "hair_rgb":   [45, 30, 20],
                "eye_rgb":    [60, 40, 20],
                "shirt_rgb":  [70, 90, 160],
                "hair_style": "short",
                "gender":     "male",
            }
        if appearance:
            # caller-supplied fields override the position defaults
            defaults = {**defaults, **appearance}

        self.skin       = _rgb(defaults["skin_rgb"])
        self.hair       = _rgb(defaults["hair_rgb"])
        self.eye        = _rgb(defaults["eye_rgb"])
        self.shirt      = _rgb(defaults["shirt_rgb"])
        self.hair_style = defaults["hair_style"]
        self.gender     = defaults["gender"]

        # If gender=female but the caller (or default) picked a neutral
        # hairstyle, upgrade to 'long' so the female portrait actually reads
        # as female. Callers can still force 'short' explicitly.
        if self.gender == "female" and self.hair_style == "medium":
            self.hair_style = "long"

        # derived colours
        self.lip_color = (
            min(255, self.skin[0] + 30),
            max(0,   self.skin[1] - 40),
            max(0,   self.skin[2] - 40),
        )

    # ── public API ───────────────────────────────────────────────────────────

    def render(
        self,
        width: int = 560,
        height: int = 600,
        mouth_opening: float = 0.0,
        is_active: bool = False,
    ) -> Image.Image:
        """
        Render one portrait frame and return a PIL RGB Image.

        Parameters
        ----------
        mouth_opening : 0.0 = closed  →  1.0 = fully open (lip-sync)
        is_active     : True when this character is the current speaker
        """
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        # ── background ───────────────────────────────────────────────────────
        bg = (52, 70, 100) if is_active else (32, 38, 52)
        draw.rectangle([0, 0, width, height], fill=bg)

        if is_active:
            # Glowing border (3 nested rectangles, fading out)
            for i, alpha_frac in enumerate([1.0, 0.6, 0.3]):
                col = tuple(int(c * alpha_frac) for c in (100, 170, 255))
                draw.rectangle([i, i, width - 1 - i, height - 1 - i], outline=col)

        # ── character geometry ───────────────────────────────────────────────
        cx       = width  // 2
        head_r   = int(min(width, height) * 0.21)
        head_cy  = int(height * 0.37)
        head_cx  = cx
        body_top = head_cy + int(head_r * 1.05)

        self._draw_body(draw, cx, body_top, width, height, head_r)
        self._draw_neck(draw, cx, head_cy, head_r)
        self._draw_hair_back(draw, head_cx, head_cy, head_r)
        self._draw_head(draw, head_cx, head_cy, head_r)
        self._draw_hair_front(draw, head_cx, head_cy, head_r)
        self._draw_ears(draw, head_cx, head_cy, head_r)
        self._draw_eyebrows(draw, head_cx, head_cy, head_r)
        self._draw_eyes(draw, head_cx, head_cy, head_r)
        self._draw_nose(draw, head_cx, head_cy, head_r)
        self._draw_mouth(draw, head_cx, head_cy, head_r, mouth_opening)

        # ── sound bars (only for active speaker actually talking) ─────────────
        if is_active and mouth_opening > 0.08:
            self._draw_sound_bars(draw, cx, head_cy + int(head_r * 1.7), mouth_opening)

        # ── name label ───────────────────────────────────────────────────────
        self._draw_name_label(draw, cx, height - 40, width, is_active)

        return img

    # ── drawing helpers ──────────────────────────────────────────────────────

    def _draw_body(self, draw, cx, top_y, width, height, head_r):
        """Shoulders + torso (trapezoid shirt)."""
        sw = int(head_r * 2.1)
        pts = [
            (cx - sw,          top_y),
            (cx + sw,          top_y),
            (cx + int(sw * 1.4), height + 10),
            (cx - int(sw * 1.4), height + 10),
        ]
        draw.polygon(pts, fill=self.shirt)
        # Subtle collar lines
        draw.line([(cx - 18, top_y), (cx,  top_y + 38)], fill=(210, 210, 210), width=2)
        draw.line([(cx + 18, top_y), (cx,  top_y + 38)], fill=(210, 210, 210), width=2)

    def _draw_neck(self, draw, cx, head_cy, head_r):
        nw = int(head_r * 0.26)
        y0 = head_cy + int(head_r * 0.85)
        y1 = y0 + int(head_r * 0.5)
        draw.rectangle([cx - nw, y0, cx + nw, y1], fill=self.skin)

    def _draw_hair_back(self, draw, cx, cy, r):
        """Hair mass behind the head."""
        draw.ellipse(
            [cx - int(r * 1.07), cy - int(r * 1.24),
             cx + int(r * 1.07), cy + int(r * 0.28)],
            fill=self.hair,
        )
        if self.hair_style == "long":
            for side in (-1, 1):
                pts = [
                    (cx + side * int(r * 0.92), cy - int(r * 0.15)),
                    (cx + side * int(r * 1.18), cy + int(r * 0.55)),
                    (cx + side * int(r * 1.02), cy + int(r * 1.25)),
                    (cx + side * int(r * 0.70), cy + int(r * 1.05)),
                ]
                draw.polygon(pts, fill=self.hair)

    def _draw_head(self, draw, cx, cy, r):
        """Skin-coloured face ellipse with clipped soft side-shading.

        The previous implementation drew four full-height rectangles on each
        side of the face for "subtle side shading". Because the head is an
        ellipse (narrow at the top and bottom), those rectangles protruded
        past the face boundary and were clearly visible as rectangular blocks
        flanking every character. This version draws the shading into a
        separate RGBA layer and composites it through an ellipse mask so it
        can never bleed outside the face.
        """
        face_box = [cx - r, cy - int(r * 1.12), cx + r, cy + int(r * 0.92)]
        draw.ellipse(face_box, fill=self.skin)

        img = getattr(draw, "_image", None)
        if img is None or img.mode not in ("RGB", "RGBA"):
            return

        shade = _darken(self.skin, 14)
        fw = face_box[2] - face_box[0]
        fh = face_box[3] - face_box[1]
        if fw <= 0 or fh <= 0:
            return

        layer = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        band = max(2, int(fw * 0.22))
        ld.rectangle([0, 0, band, fh], fill=(*shade, 110))
        ld.rectangle([fw - band, 0, fw, fh], fill=(*shade, 110))

        mask = Image.new("L", (fw, fh), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, fw - 1, fh - 1], fill=255)

        shading_img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        shading_img.paste(layer, (0, 0), mask)

        dst_xy = (face_box[0], face_box[1])
        if img.mode == "RGB":
            img.paste(shading_img, dst_xy, shading_img)
        else:
            img.alpha_composite(shading_img, dst_xy)

    def _draw_hair_front(self, draw, cx, cy, r):
        """Hairline arc sitting on top of the face."""
        draw.arc(
            [cx - int(r * 1.05), cy - int(r * 1.22),
             cx + int(r * 1.05), cy - int(r * 0.22)],
            start=202, end=338,
            fill=self.hair,
            width=int(r * 0.30),
        )

    def _draw_ears(self, draw, cx, cy, r):
        ear_cy = cy - int(r * 0.06)
        ew, eh = int(r * 0.14), int(r * 0.21)
        for side in (-1, 1):
            ex = cx + side * int(r * 0.87)
            draw.ellipse([ex - ew, ear_cy - eh, ex + ew, ear_cy + eh], fill=self.skin)
            inner = _darken(self.skin, 22)
            draw.ellipse(
                [ex - int(ew * 0.48), ear_cy - int(eh * 0.48),
                 ex + int(ew * 0.48), ear_cy + int(eh * 0.48)],
                fill=inner,
            )

    def _draw_eyebrows(self, draw, cx, cy, r):
        brow_y  = cy - int(r * 0.30)
        spacing = int(r * 0.36)
        brow_w  = int(r * 0.24)
        color   = _darken(self.hair, 25)
        for side in (-1, 1):
            bx = cx + side * spacing
            draw.arc(
                [bx - brow_w, brow_y - int(r * 0.06),
                 bx + brow_w, brow_y + int(r * 0.04)],
                start=205, end=335,
                fill=color,
                width=3,
            )

    def _draw_eyes(self, draw, cx, cy, r):
        eye_y   = cy - int(r * 0.15)
        spacing = int(r * 0.36)
        ew, eh  = int(r * 0.23), int(r * 0.14)

        for side in (-1, 1):
            ex = cx + side * spacing

            # white sclera
            draw.ellipse([ex - ew, eye_y - eh, ex + ew, eye_y + eh], fill=(255, 255, 255))

            # iris
            ir = int(eh * 0.80)
            draw.ellipse([ex - ir, eye_y - ir, ex + ir, eye_y + ir], fill=self.eye)

            # pupil
            pr = int(ir * 0.48)
            draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr], fill=(14, 14, 14))

            # specular highlight
            draw.ellipse(
                [ex + 2, eye_y - pr + 1, ex + int(pr * 0.7) + 2, eye_y],
                fill=(255, 255, 255),
            )

            # upper eyelid crease
            draw.arc(
                [ex - ew, eye_y - eh, ex + ew, eye_y + eh],
                start=205, end=335,
                fill=_darken(self.skin, 45),
                width=2,
            )

            # Eyelashes — a small visual cue so female portraits read as
            # female even without explicit appearance overrides.
            if self.gender == "female":
                lash_col = (20, 15, 12)
                for k in (-2, -1, 0, 1, 2):
                    base_x = ex + int(k * ew * 0.30)
                    base_y = eye_y - eh
                    tip_x  = ex + int(k * ew * 0.36)
                    tip_y  = base_y - max(2, int(eh * 0.45))
                    draw.line([(base_x, base_y), (tip_x, tip_y)],
                              fill=lash_col, width=2)

    def _draw_nose(self, draw, cx, cy, r):
        nose_cy = cy + int(r * 0.10)
        nw      = int(r * 0.09)
        shadow  = _darken(self.skin, 28)
        for side in (-1, 1):
            nx = cx + side * int(nw * 0.90)
            draw.ellipse(
                [nx - int(r * 0.042), nose_cy - int(r * 0.030),
                 nx + int(r * 0.042), nose_cy + int(r * 0.042)],
                fill=shadow,
            )

    def _draw_mouth(self, draw, cx, cy, r, mouth_opening: float):
        """Draw lips; mouth opens with `mouth_opening` 0.0→1.0."""
        mouth_cy = cy + int(r * 0.40)
        mw       = int(r * 0.37)
        lc       = self.lip_color           # upper-lip colour
        lc_lo    = _darken(lc, 15)         # lower lip a touch darker

        if mouth_opening < 0.04:
            # ── closed (slight smile) ─────────────────────────────────────
            draw.arc(
                [cx - mw, mouth_cy - int(r * 0.06),
                 cx + mw, mouth_cy + int(r * 0.06)],
                start=8, end=172,
                fill=lc,
                width=3,
            )
            return

        mh = max(5, int(r * 0.30 * mouth_opening))

        # ── dark mouth cavity ─────────────────────────────────────────────
        draw.ellipse(
            [cx - mw, mouth_cy - mh, cx + mw, mouth_cy + mh],
            fill=(35, 12, 12),
        )

        # ── upper teeth (kept inside the mouth cavity width) ──────────────
        teeth_inset = int(mw * 0.84)
        teeth_h     = max(3, int(mh * 0.52))
        teeth_left  = cx - teeth_inset
        teeth_right = cx + teeth_inset
        draw.rectangle(
            [teeth_left, mouth_cy - mh, teeth_right, mouth_cy - mh + teeth_h],
            fill=(242, 240, 235),
        )
        # Vertical tooth dividers, evenly spaced and always inside the teeth rect.
        n_dividers  = 5
        band_width  = teeth_right - teeth_left
        for i in range(1, n_dividers + 1):
            tx = teeth_left + int(band_width * i / (n_dividers + 1))
            draw.line(
                [(tx, mouth_cy - mh + 1), (tx, mouth_cy - mh + teeth_h - 1)],
                fill=(200, 198, 193),
                width=1,
            )

        # ── lower teeth (visible when wide open) ──────────────────────────
        if mouth_opening > 0.45:
            lo_h = max(2, int(teeth_h * 0.65))
            draw.rectangle(
                [cx - int(mw * 0.72), mouth_cy + mh - lo_h,
                 cx + int(mw * 0.72), mouth_cy + mh],
                fill=(228, 226, 220),
            )

        # ── upper lip: a single thin curve hugging the top of the cavity ─
        draw.arc(
            [cx - mw - 2, mouth_cy - mh - int(r * 0.04),
             cx + mw + 2, mouth_cy - mh + int(r * 0.04)],
            start=0, end=180,
            fill=lc,
            width=3,
        )

        # ── cupid's bow: a small, symmetric dip centred above the lip ────
        # The previous implementation used a mismatched pair of arc angles
        # (one side ended at 390° / went the wrong way around the ellipse),
        # so the right half of the bow was always drawn incorrectly. This
        # replacement is two mirror-image line segments that can't bleed
        # outside the lip.
        bow_half = max(2, int(mw * 0.22))
        bow_top  = mouth_cy - mh - int(r * 0.07)
        bow_dip  = mouth_cy - mh - int(r * 0.01)
        bow_col  = _darken(lc, 20)
        draw.line([(cx - bow_half, bow_top), (cx, bow_dip)], fill=bow_col, width=2)
        draw.line([(cx, bow_dip), (cx + bow_half, bow_top)], fill=bow_col, width=2)

        # ── lower lip ─────────────────────────────────────────────────────
        draw.arc(
            [cx - mw, mouth_cy + mh - int(r * 0.06),
             cx + mw, mouth_cy + mh + int(r * 0.05)],
            start=180, end=360,
            fill=lc_lo,
            width=4,
        )

    def _draw_sound_bars(self, draw, cx, cy, amplitude: float):
        """Animated equaliser bars beneath the active speaker."""
        n_bars   = 7
        bar_w    = 7
        spacing  = 5
        total_w  = n_bars * (bar_w + spacing) - spacing

        for i in range(n_bars):
            # envelope: taller in the middle
            env = math.sin(i * math.pi / (n_bars - 1))
            bh  = max(3, int(28 * amplitude * env))
            bx  = cx - total_w // 2 + i * (bar_w + spacing)
            draw.rectangle(
                [bx, cy - bh, bx + bar_w, cy + bh],
                fill=(80, 190, 255),
            )

    def _draw_name_label(self, draw, cx, cy, width, is_active: bool):
        """Rounded pill label at the bottom of the panel."""
        lw, lh = 170, 28
        bg_col  = (75, 115, 175) if is_active else (48, 54, 72)
        txt_col = (255, 255, 255) if is_active else (180, 185, 200)

        # Background pill
        draw.rounded_rectangle(
            [cx - lw // 2, cy - lh // 2, cx + lw // 2, cy + lh // 2],
            radius=8,
            fill=bg_col,
        )
        # Name text
        draw.text((cx, cy), self.name, fill=txt_col, anchor="mm")
