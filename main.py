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

        gw    = "man" if gender == "male" else "woman"
        facing = "slightly right" if gender == "female" else "slightly left"
        prompt = (
            f"photorealistic portrait headshot of a {gw}, {role}, "
            f"facing {facing}, natural friendly expression, "
            f"professional business casual attire, "
            f"soft studio lighting, shallow depth of field, sharp focus on face, "
            f"plain light grey background, 4K DSLR quality, ultra realistic skin texture"
        )

        print(f"  [SD] Generating face for '{name}' ({gender} {role}) …")
        img = _sd_generate(prompt, PORTRAIT_NEGATIVE, width, height, model_id=sd_model)
        if img is None:
            sys.exit(
                f"\n[Error] Stable Diffusion failed to generate face for '{name}'.\n"
                f"  Make sure diffusers is installed and the model '{sd_model}' is available.\n"
                f"  Or place your own photo at: {out_path}\n"
            )
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

    # ── 3. Generate / load face images ───────────────────────────────────────
    banner("Generating / loading face images (Stable Diffusion)")
    face_paths    = _generate_missing_faces(
        char_names, char_data, args.faces_dir,
        sd_model=args.sd_model,
        regen=args.regen_faces,
        width=512, height=512,
    )
    ordered_faces = [face_paths[n] for n in char_names]

    # ── 4. Generate room background ───────────────────────────────────────────
    banner("Generating room background (Stable Diffusion)")
    from scene_generator import generate_room_background

    # Delete cached file if --regen-room requested
    if args.regen_room:
        cache_path = os.path.join(args.faces_dir, "room_bg.png")
        if os.path.isfile(cache_path):
            os.remove(cache_path)
            print(f"  Removed cached room background for regeneration.")

    room_img = generate_room_background(
        width        = args.width,
        height       = int(args.height * 0.85),
        model_id     = args.sd_model,
        cache_dir    = args.faces_dir,
        room_bg_path = args.room_bg,
    )

    # ── 5. Generate seated body images (optional enhancement) ─────────────────
    banner("Generating character body images (Stable Diffusion)")
    from scene_generator import generate_character_body

    # Build default appearance data for body-image prompts
    DEFAULT_APPEARANCES = [
        {"skin_rgb": [235, 200, 170], "hair_rgb": [60, 40, 20],  "hair_style": "short"},
        {"skin_rgb": [200, 165, 130], "hair_rgb": [140, 90, 50], "hair_style": "medium"},
    ]

    char_body_paths: dict[str, str | None] = {}
    for idx, name in enumerate(char_names):
        info = char_data.get(name, {})
        app  = DEFAULT_APPEARANCES[idx % len(DEFAULT_APPEARANCES)]
        body = generate_character_body(
            name        = name,
            role        = info.get("role",   "person"),
            gender      = info.get("gender", "neutral"),
            appearance  = app,
            width       = int(args.width * 0.38),
            height      = int(args.width * 0.38 * 1.28),
            model_id    = args.sd_model,
            cache_dir   = args.faces_dir,
            idx         = idx,
        )
        safe = name.lower().replace(" ", "_")
        if body:
            body_path = os.path.join(args.faces_dir, f"body_{safe}.png")
            char_body_paths[name] = body_path
        else:
            char_body_paths[name] = None

    # ── 6. Build dialogue segments ────────────────────────────────────────────
    banner(f"Generating {len(dialogue)} dialogue segments")
    from tts             import text_to_speech
    from wav2lip_runner  import run_wav2lip
    from video_composer  import VideoComposer
    from lip_sync        import get_audio_duration

    composer       = VideoComposer(
        args.width, args.height, args.fps,
        room_bg_image   = room_img,
        char_body_paths = char_body_paths,
    )
    segment_paths  = []
    tmp_dir        = tempfile.mkdtemp(prefix="ttv_")

    try:
        for line_idx, line in enumerate(dialogue):
            speaker      = line["speaker"]
            text         = line["text"]
            speaker_idx  = char_names.index(speaker) if speaker in char_names else 0
            speaker_face = ordered_faces[speaker_idx]

            print(
                f"\n  [{line_idx + 1}/{len(dialogue)}] "
                f"{speaker}: \"{text[:60]}{'…' if len(text) > 60 else ''}\""
            )

            # ── TTS ───────────────────────────────────────────────────────────
            # text_to_speech always returns a WAV file
            speaker_gender = char_data.get(speaker, {}).get("gender", "male")
            audio_path = os.path.join(tmp_dir, f"line_{line_idx:03d}_audio.wav")
            audio_path = text_to_speech(
                text,
                output_path    = audio_path,
                lang           = args.lang,
                speaker_gender = speaker_gender,
            )

            duration = get_audio_duration(audio_path) + args.pause

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

        # ── 7. Concatenate & export ───────────────────────────────────────────
        banner(f"Exporting final video → {args.output}")
        VideoComposer.concat_and_write(segment_paths, args.output, fps=args.fps)

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
