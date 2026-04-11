#!/usr/bin/env python3
"""
main.py — Text-to-Video with AI Portraits (Stable Diffusion + Wav2Lip)
=======================================================================

Full pipeline
-------------
  1. Load dialogue JSON (characters + dialogue lines)
  2. Stable Diffusion  →  portrait per character (face_<name>.png)
  3. Stable Diffusion  →  photorealistic room background
  4. For each dialogue line:
       a. edge-tts (Microsoft neural)  →  TTS audio (.wav)
       b. Wav2Lip (GPU)  →  lip-synced talking face video for the speaker
       c. Static face video  →  the listener (mouth closed)
  5. Compose side-by-side + subtitle bar  →  final MP4

Prerequisites
-------------
  python setup_models.py        # clone Wav2Lip, download weights
  pip install -r requirements.txt

Usage
-----
  python main.py example_dialogue.json
  python main.py my_script.json --output film.mp4 --fps 25
  python main.py my_script.json --regen-faces        # force regenerate faces
  python main.py my_script.json --faces-dir my_faces # use pre-supplied face images
  python main.py my_script.json --sd-model "stabilityai/stable-diffusion-2-1"
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile


# ── helpers ───────────────────────────────────────────────────────────────────

def load_script(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "characters" not in data or "dialogue" not in data:
        sys.exit("[Error] JSON must have 'characters' and 'dialogue' keys.")
    chars = list(data["characters"].keys())
    if len(chars) < 2:
        sys.exit("[Error] Need at least 2 characters.")
    return data


def banner(msg: str):
    w = min(72, len(msg) + 4)
    print(f"\n{'─' * w}\n  {msg}\n{'─' * w}")


def _make_placeholder_face(
    name: str,
    gender: str,
    panel_idx: int,
    width: int = 512,
    height: int = 512,
) -> "Image.Image":
    """
    Draw a simple but clean portrait placeholder using PIL when Stable
    Diffusion is unavailable.  Produces a head + shoulders silhouette with
    skin tone, hair, and clothing so Wav2Lip has a usable face region.

    panel_idx 0 (left)  → face turned slightly right
    panel_idx 1 (right) → face turned slightly left
    """
    from PIL import Image, ImageDraw, ImageFilter
    import math

    is_female = gender.strip().lower() in ("female", "f", "woman", "girl")

    # ── Background gradient ───────────────────────────────────────────────────
    img  = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    bg_top    = (38, 42, 52) if panel_idx == 0 else (42, 38, 34)
    bg_bottom = (22, 26, 34) if panel_idx == 0 else (28, 22, 18)
    for y in range(height):
        t = y / height
        r = int(bg_top[0] + t * (bg_bottom[0] - bg_top[0]))
        g = int(bg_top[1] + t * (bg_bottom[1] - bg_top[1]))
        b = int(bg_top[2] + t * (bg_bottom[2] - bg_top[2]))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # ── Dimensions ────────────────────────────────────────────────────────────
    cx      = width  // 2
    # Subtle horizontal shift to convey inward facing
    cx_off  = int(width * 0.04) * (1 if panel_idx == 0 else -1)
    cx     += cx_off

    head_r  = int(width * 0.22)
    head_cy = int(height * 0.36)

    # ── Shoulders / clothing ──────────────────────────────────────────────────
    shoulder_y = int(height * 0.62)
    shirt_col  = (55, 75, 110) if not is_female else (110, 65, 85)
    draw.ellipse(
        [cx - int(width * 0.46), shoulder_y,
         cx + int(width * 0.46), height + int(height * 0.15)],
        fill=shirt_col,
    )

    # Subtle neck
    neck_w = int(width * 0.08)
    neck_h = int(height * 0.10)
    skin   = (210, 170, 130) if not is_female else (235, 195, 160)
    draw.rectangle(
        [cx - neck_w, head_cy + head_r - 4,
         cx + neck_w, head_cy + head_r + neck_h],
        fill=skin,
    )

    # ── Hair (drawn behind head) ──────────────────────────────────────────────
    hair_col = (45, 30, 20) if not is_female else (90, 55, 30)
    hair_r   = int(head_r * 1.08)
    draw.ellipse(
        [cx - hair_r, head_cy - hair_r - int(head_r * 0.18),
         cx + hair_r, head_cy + hair_r],
        fill=hair_col,
    )
    if is_female:
        # Shoulder-length hair sides
        for side in (-1, 1):
            x_a = cx + side * int(head_r * 0.55)
            x_b = cx + side * int(head_r * 1.35)
            draw.ellipse(
                [min(x_a, x_b), head_cy,
                 max(x_a, x_b), head_cy + int(head_r * 1.4)],
                fill=hair_col,
            )

    # ── Head / skin ───────────────────────────────────────────────────────────
    draw.ellipse(
        [cx - head_r, head_cy - head_r,
         cx + head_r, head_cy + head_r],
        fill=skin,
    )

    # ── Eyes ──────────────────────────────────────────────────────────────────
    eye_y   = head_cy - int(head_r * 0.12)
    eye_sep = int(head_r * 0.38)
    eye_rx  = int(head_r * 0.14)
    eye_ry  = int(head_r * 0.09)
    for side in (-1, 1):
        ex = cx + side * eye_sep
        # White
        draw.ellipse([ex - eye_rx, eye_y - eye_ry,
                      ex + eye_rx, eye_y + eye_ry], fill=(245, 242, 238))
        # Iris
        draw.ellipse([ex - eye_ry, eye_y - eye_ry,
                      ex + eye_ry, eye_y + eye_ry], fill=(60, 90, 50))
        # Pupil
        pr = max(2, int(eye_ry * 0.55))
        draw.ellipse([ex - pr, eye_y - pr, ex + pr, eye_y + pr], fill=(15, 10, 8))

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    brow_y = eye_y - int(head_r * 0.16)
    brow_w = int(head_r * 0.22)
    brow_h = max(2, int(head_r * 0.04))
    for side in (-1, 1):
        bx = cx + side * eye_sep
        draw.ellipse([bx - brow_w, brow_y - brow_h,
                      bx + brow_w, brow_y + brow_h],
                     fill=(50, 32, 18))

    # ── Nose ──────────────────────────────────────────────────────────────────
    nose_y = head_cy + int(head_r * 0.15)
    nose_x = cx + cx_off // 3          # subtle inward-facing offset
    draw.ellipse([nose_x - 5, nose_y - 4,
                  nose_x + 5, nose_y + 6],
                 fill=(int(skin[0] * 0.88), int(skin[1] * 0.82), int(skin[2] * 0.78)))

    # ── Mouth ─────────────────────────────────────────────────────────────────
    mouth_y  = head_cy + int(head_r * 0.38)
    mouth_w  = int(head_r * 0.28)
    lip_col  = (185, 105, 95) if is_female else (165, 100, 90)
    draw.arc([cx - mouth_w, mouth_y - int(head_r * 0.08),
              cx + mouth_w, mouth_y + int(head_r * 0.10)],
             start=5, end=175, fill=lip_col, width=max(2, int(head_r * 0.04)))

    # ── Soft vignette ─────────────────────────────────────────────────────────
    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    for i in range(40):
        a = int(120 * (i / 40) ** 2)
        vd.rectangle([i, i, width - 1 - i, height - 1 - i],
                     outline=(0, 0, 0, a))
    img = img.convert("RGBA")
    img.alpha_composite(vignette)

    # ── Name label ────────────────────────────────────────────────────────────
    label_y = int(height * 0.88)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(14, int(height * 0.045)),
        )
    except Exception:
        font = None
    draw2 = ImageDraw.Draw(img)
    draw2.text(
        (cx, label_y), name,
        fill=(220, 220, 220, 210),
        font=font,
        anchor="mm" if font else None,
    )

    return img.convert("RGB")


def _generate_missing_faces(
    char_names: list,
    char_data: dict,
    faces_dir: str,
    sd_model: str,
    regen: bool,
    width: int = 512,
    height: int = 512,
) -> dict:
    """
    For any character whose face_<name>.png doesn't exist (or --regen),
    generate it with Stable Diffusion.  Returns {name: path}.
    """
    from scene_generator import _sd_generate, PORTRAIT_NEGATIVE

    os.makedirs(faces_dir, exist_ok=True)
    face_paths = {}
    extensions = [".png", ".jpg", ".jpeg", ".webp"]

    for name in char_names:
        safe = name.lower().replace(" ", "_")

        # Check if file already exists (any supported extension)
        found = None
        for ext in extensions:
            candidate = os.path.join(faces_dir, f"face_{safe}{ext}")
            if os.path.isfile(candidate):
                found = candidate
                break

        out_path = os.path.join(faces_dir, f"face_{safe}.png")

        if found and not regen:
            print(f"  ✓ {name} → {found}  (existing)")
            face_paths[name] = found
            continue

        # Generate with Stable Diffusion
        info   = char_data.get(name, {})
        role   = info.get("role",   "person")
        gender = info.get("gender", "neutral")

        gw = "man" if gender == "male" else "woman"
        # char index 0 = left panel  → faces right (toward the other person)
        # char index 1 = right panel → faces left  (toward the other person)
        char_idx_local = char_names.index(name)
        facing = "slightly right, looking toward the right" if char_idx_local == 0 \
                 else "slightly left, looking toward the left"
        prompt = (
            f"photorealistic portrait headshot of a {gw}, {role}, "
            f"facing {facing}, natural conversational expression, "
            f"professional business casual attire, "
            f"soft studio lighting, shallow depth of field, sharp focus on face, "
            f"plain neutral dark background, 4K DSLR quality, ultra realistic skin texture"
        )

        print(f"  [SD] Generating face for '{name}' ({gender} {role}) …")
        img = _sd_generate(prompt, PORTRAIT_NEGATIVE, width, height, model_id=sd_model)
        if img is None:
            print(f"  [SD] ⚠ Stable Diffusion unavailable — using PIL placeholder for '{name}'.")
            img = _make_placeholder_face(name, gender, char_idx_local, width, height)
        img.save(out_path)
        print(f"  ✓ {name} → {out_path}")
        face_paths[name] = out_path

    return face_paths


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video from a dialogue script using SD + Wav2Lip."
    )
    parser.add_argument("script",               help="Dialogue JSON file")
    parser.add_argument("--output",  "-o",      default="output.mp4")
    parser.add_argument("--sd-model",           default="runwayml/stable-diffusion-v1-5",
                        help="HuggingFace model ID or local path for Stable Diffusion")
    parser.add_argument("--faces-dir",          default="faces",
                        help="Directory for face images (face_<name>.png). "
                             "Missing faces are auto-generated with SD.")
    parser.add_argument("--regen-faces",        action="store_true",
                        help="Force regenerate all face images with SD")
    parser.add_argument("--regen-room",         action="store_true",
                        help="Force regenerate room background with SD")
    parser.add_argument("--scene-image",         default=None,
                        help="Single photo with BOTH characters already in frame "
                             "(e.g. scene.jpg). When supplied the pipeline animates "
                             "only the speaker's face region inside this photo — the "
                             "most natural looking result. Skips SD face/room generation.")
    parser.add_argument("--room-bg",            default=None,
                        help="Path to a custom room background image (JPG/PNG). "
                             "Uses SD generation if omitted.")
    parser.add_argument("--fps",     type=int,  default=25)
    parser.add_argument("--width",   type=int,  default=1280)
    parser.add_argument("--height",  type=int,  default=720)
    parser.add_argument("--lang",               default="en",
                        help="TTS language code (default: en)")
    parser.add_argument("--pause",   type=float, default=0.3,
                        help="Silence gap between dialogue lines in seconds")
    parser.add_argument("--no-gan",  action="store_true",
                        help="Use wav2lip.pth instead of wav2lip_gan.pth (faster)")
    args = parser.parse_args()

    # ── 1. Load script ────────────────────────────────────────────────────────
    banner("Loading dialogue script")
    data        = load_script(args.script)
    char_names  = list(data["characters"].keys())[:2]
    char_data   = data["characters"]
    dialogue    = data["dialogue"]
    print(f"  Characters : {char_names}")
    print(f"  Lines      : {len(dialogue)}")

    # ── 2. Check Wav2Lip ──────────────────────────────────────────────────────
    from wav2lip_runner import is_wav2lip_ready
    if not is_wav2lip_ready():
        print("\n[Warning] Wav2Lip is not set up. Run:  python setup_models.py")
        print("          Falling back to cartoon portrait mode for preview.\n")
        _run_cartoon_fallback(args, data, char_names, dialogue)
        return

    # ── 3. Scene image mode vs two-portrait mode ──────────────────────────────
    from tts            import text_to_speech
    from wav2lip_runner import run_wav2lip
    from lip_sync       import get_audio_duration

    use_scene = args.scene_image and os.path.isfile(args.scene_image)

    if use_scene:
        # ── Single-scene mode: one photo with both people ─────────────────────
        banner(f"Single-scene mode  →  {args.scene_image}")
        from video_composer import SingleSceneComposer
        composer      = SingleSceneComposer(
            args.scene_image, char_names,
            args.width, args.height, args.fps,
        )
        ordered_faces = None   # not used; face crops come from the scene
    else:
        # ── Two-portrait mode: separate SD faces + split-screen ───────────────
        banner("Generating / loading face images (Stable Diffusion)")
        face_paths    = _generate_missing_faces(
            char_names, char_data, args.faces_dir,
            sd_model=args.sd_model,
            regen=args.regen_faces,
            width=512, height=512,
        )
        ordered_faces = [face_paths[n] for n in char_names]

        from video_composer import VideoComposer
        composer = VideoComposer(
            args.width, args.height, args.fps,
            char_names=char_names,
        )

    # ── 4. Build dialogue segments ────────────────────────────────────────────
    banner(f"Generating {len(dialogue)} dialogue segments")
    segment_paths = []
    tmp_dir       = tempfile.mkdtemp(prefix="ttv_")

    try:
        for line_idx, line in enumerate(dialogue):
            speaker      = line["speaker"]
            text         = line["text"]
            speaker_idx  = char_names.index(speaker) if speaker in char_names else 0

            print(
                f"\n  [{line_idx + 1}/{len(dialogue)}] "
                f"{speaker}: \"{text[:60]}{'…' if len(text) > 60 else ''}\""
            )

            # ── TTS ───────────────────────────────────────────────────────────
            speaker_gender = char_data.get(speaker, {}).get("gender", "male")
            audio_path = os.path.join(tmp_dir, f"line_{line_idx:03d}_audio.wav")
            audio_path = text_to_speech(
                text,
                output_path    = audio_path,
                lang           = args.lang,
                speaker_gender = speaker_gender,
            )

            duration = get_audio_duration(audio_path) + args.pause

            # ── Face image for Wav2Lip ─────────────────────────────────────────
            if use_scene:
                # Crop the speaker's face from the scene photo
                speaker_face = composer.get_face_crop_path(speaker_idx, tmp_dir)
            else:
                speaker_face = ordered_faces[speaker_idx]

            # ── Wav2Lip (talking face) ─────────────────────────────────────────
            wav2lip_out = os.path.join(tmp_dir, f"line_{line_idx:03d}_wav2lip.mp4")
            run_wav2lip(
                face_image_path = speaker_face,
                audio_path      = audio_path,
                output_path     = wav2lip_out,
                use_gan         = not args.no_gan,
                fps             = args.fps,
                pad_bottom      = 12,
            )

            # ── compose segment (writes MP4 with muxed audio immediately) ─────
            seg_out = os.path.join(tmp_dir, f"line_{line_idx:03d}_segment.mp4")
            if use_scene:
                composer.create_segment(
                    speaker_idx        = speaker_idx,
                    dialogue_text      = text,
                    speaker_name       = speaker,
                    wav2lip_video_path = wav2lip_out,
                    audio_path         = audio_path,
                    segment_out_path   = seg_out,
                )
            else:
                composer.create_segment(
                    speaker_idx        = speaker_idx,
                    dialogue_text      = text,
                    speaker_name       = speaker,
                    face_image_paths   = ordered_faces,
                    wav2lip_video_path = wav2lip_out,
                    audio_path         = audio_path,
                    segment_out_path   = seg_out,
                )
            segment_paths.append(seg_out)

        # ── Concatenate & export ──────────────────────────────────────────────
        banner(f"Exporting final video → {args.output}")
        composer.concat_and_write(segment_paths, args.output, fps=args.fps)

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    banner(f"Done!  {os.path.abspath(args.output)}")
    print(f"  Resolution  : {args.width}×{args.height} @ {args.fps} fps")
    print(f"  Characters  : {', '.join(char_names)}")
    print(f"  Face images : {args.faces_dir}/")


# ── cartoon fallback (when Wav2Lip not installed) ─────────────────────────────

def _run_cartoon_fallback(args, data, char_names, dialogue):
    """
    Run the original PIL-based cartoon pipeline as a quick preview
    when Wav2Lip has not been set up yet.
    """
    print("  [Fallback] Running cartoon-portrait preview …")
    from character_renderer import CharacterRenderer
    from tts                import text_to_speech
    from lip_sync           import extract_mouth_openings
    import tempfile, numpy as np

    try:
        from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips  # v1
    except ImportError:
        from moviepy import VideoClip, AudioFileClip, concatenate_videoclips          # v2

    DEFAULT_APPEARANCES = [
        {"skin_rgb": [235, 200, 170], "hair_rgb": [60, 40, 20],  "hair_style": "short"},
        {"skin_rgb": [200, 165, 130], "hair_rgb": [140, 90, 50], "hair_style": "medium"},
    ]

    renderers = []
    for idx, name in enumerate(char_names):
        info = data["characters"].get(name, {})
        app  = DEFAULT_APPEARANCES[idx % len(DEFAULT_APPEARANCES)]
        renderers.append(CharacterRenderer(name, app, ["left", "right"][idx]))

    from video_composer import _subtitle_img  # noqa: F401
    fps    = args.fps
    W, H   = args.width, args.height
    char_w = W // 2
    char_h = int(H * 0.85)
    sub_h  = H - char_h

    clips     = []
    tmp_files = []

    for line in dialogue:
        speaker  = line["speaker"]
        text     = line["text"]
        spk_idx  = char_names.index(speaker) if speaker in char_names else 0

        audio_path     = tempfile.mktemp(suffix=".wav")
        tmp_files.append(audio_path)
        spk_gender     = char_data.get(speaker, {}).get("gender", "male")
        audio_path     = text_to_speech(
            text,
            output_path    = audio_path,
            lang           = args.lang,
            speaker_gender = spk_gender,
        )
        mouth_vals  = list(extract_mouth_openings(audio_path, fps)) + [0.0] * int(fps * 0.3)
        n_frames    = len(mouth_vals)
        duration    = n_frames / fps

        frames_np = []
        for fi in range(n_frames):
            mo    = float(mouth_vals[fi])
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:] = [14, 17, 26]
            for i, r in enumerate(renderers):
                is_active = (i == spk_idx)
                char_img  = r.render(char_w, char_h, mo if is_active else 0.0, is_active)
                frame[:char_h, i * char_w:(i + 1) * char_w] = np.asarray(char_img)
            sub = _subtitle_img(W, sub_h, speaker, text)
            frame[char_h:, :] = sub
            frames_np.append(frame)

        def make_frame(t, _f=frames_np, _fps=fps):
            return _f[min(int(t * _fps), len(_f) - 1)]

        clip  = VideoClip(make_frame, duration=duration)
        audio = AudioFileClip(audio_path)
        try:
            clip = clip.with_audio(audio)
        except AttributeError:
            clip = clip.set_audio(audio)
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    import inspect, tempfile as _tf
    tmp_a = os.path.join(_tf.gettempdir(), "ttv_cartoon_audio.m4a")
    kw    = dict(fps=fps, codec="libx264", audio_codec="aac",
                 temp_audiofile=tmp_a, remove_temp=False, logger="bar")
    if "verbose" in inspect.signature(final.write_videofile).parameters:
        kw["verbose"] = False
    try:
        final.write_videofile(args.output, **kw)
    finally:
        try:
            os.unlink(tmp_a)
        except Exception:
            pass
    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    print(f"\n  Preview saved → {os.path.abspath(args.output)}")
    print("  Run  python setup_models.py  then re-run for photorealistic output.")


if __name__ == "__main__":
    main()
