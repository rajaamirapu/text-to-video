#!/usr/bin/env python3
"""
main.py — Text-to-Video with AI Portraits (Ollama + Wav2Lip)
=======================================================================

Full pipeline
-------------
  1. Load dialogue JSON (characters + dialogue lines)
  2. Ollama LLM  →  character appearance description (colours, style)
  3. Ollama image model (if available) or enhanced PIL  →  portrait per character
  4. For each dialogue line:
       a. gTTS  →  TTS audio (.mp3)
       b. Wav2Lip (GPU)  →  lip-synced talking face video for the speaker
       c. Static face video  →  the listener (mouth closed)
  5. Compose side-by-side + subtitle bar  →  final MP4

Prerequisites
-------------
  python setup_models.py        # clone Wav2Lip, download weights
  pip install -r requirements.txt
  ollama serve                  # optional but recommended

Usage
-----
  python main.py example_dialogue.json
  python main.py my_script.json --output film.mp4 --fps 25
  python main.py my_script.json --no-ollama          # skip Ollama descriptions
  python main.py my_script.json --regen-faces        # force regenerate faces
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


def _load_face_images(char_names: list, faces_dir: str) -> dict:
    """
    Load user-supplied face images from faces_dir.

    Expected filenames (case-insensitive):
        face_<name>.png / .jpg / .jpeg
        e.g.  faces/face_alice.png   faces/face_bob.jpg

    Aborts with a clear message listing exactly what files are missing.
    """
    face_paths = {}
    missing    = []
    extensions = [".png", ".jpg", ".jpeg", ".webp"]

    os.makedirs(faces_dir, exist_ok=True)

    for name in char_names:
        safe = name.lower().replace(" ", "_")
        found = None
        for ext in extensions:
            candidate = os.path.join(faces_dir, f"face_{safe}{ext}")
            if os.path.isfile(candidate):
                found = candidate
                break
        if found:
            print(f"  ✓ {name} → {found}")
            face_paths[name] = found
        else:
            tried = ", ".join(f"face_{safe}{e}" for e in extensions)
            missing.append(f"  • {faces_dir}/{tried}")

    if missing:
        msg = (
            "\n[Error] Missing face image(s). "
            f"Place your photos in the '{faces_dir}/' folder:\n"
            + "\n".join(missing)
            + f"\n\nExample:\n"
            + "".join(
                f"  cp /path/to/photo.jpg {faces_dir}/face_{n.lower().replace(' ','_')}.jpg\n"
                for n in char_names if n not in face_paths
            )
        )
        sys.exit(msg)

    return face_paths


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a real-human talking-head video from a dialogue script."
    )
    parser.add_argument("script",              help="Dialogue JSON file")
    parser.add_argument("--output",  "-o",     default="output.mp4")
    parser.add_argument("--ollama-url",        default="http://localhost:11434")
    parser.add_argument("--no-ollama",         action="store_true",
                        help="Use built-in default appearances (skip Ollama)")
    parser.add_argument("--faces-dir",         default="faces",
                        help="Directory containing face images named "
                             "face_<name>.png  (e.g. face_alice.png)")
    parser.add_argument("--fps",     type=int, default=25)
    parser.add_argument("--width",   type=int, default=1280)
    parser.add_argument("--height",  type=int, default=720)
    parser.add_argument("--lang",              default="en",
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
    dialogue    = data["dialogue"]
    print(f"  Characters : {char_names}")
    print(f"  Lines      : {len(dialogue)}")

    # ── 2. Check Wav2Lip ──────────────────────────────────────────────────────
    from wav2lip_runner import is_wav2lip_ready, WAV2LIP_DIR, WAV2LIP_CHECKPOINT
    if not is_wav2lip_ready():
        print("\n[Warning] Wav2Lip is not set up. Run:  python setup_models.py")
        print("          Falling back to cartoon portrait mode for preview.\n")
        _run_cartoon_fallback(args, data, char_names, dialogue)
        return

    # ── 3. Generate appearances via Ollama ────────────────────────────────────
    banner("Generating character appearances (Ollama)")
    from ollama_client import OllamaClient, DEFAULT_APPEARANCES
    ollama     = OllamaClient(args.ollama_url)
    use_ollama = (not args.no_ollama) and ollama.is_available()

    if use_ollama:
        print(f"  Ollama → {args.ollama_url}  model: {ollama.get_best_model()}")
    else:
        reason = "--no-ollama flag" if args.no_ollama else "not reachable"
        print(f"  Ollama skipped ({reason}) — using default appearances")

    appearances: dict = {}
    for idx, name in enumerate(char_names):
        info = data["characters"][name]
        if use_ollama:
            app = ollama.generate_appearance(
                name=name,
                role=info.get("role", "person"),
                gender=info.get("gender", "neutral"),
                default_idx=idx,
            )
        else:
            app = DEFAULT_APPEARANCES[idx % len(DEFAULT_APPEARANCES)]
        appearances[name] = app
        print(f"  ✓ {name}: skin={app.get('skin_rgb')} hair={app.get('hair_rgb')}")

    # ── 4. Load face images (user-supplied) ──────────────────────────────────
    banner("Loading face images")
    face_paths = _load_face_images(char_names, args.faces_dir)

    # List of face paths in character order
    ordered_faces = [face_paths[n] for n in char_names]

    # ── 5. Build dialogue segments ────────────────────────────────────────────
    banner(f"Generating {len(dialogue)} dialogue segments")
    from tts           import text_to_speech
    from wav2lip_runner import run_wav2lip, make_listening_video
    from video_composer import VideoComposer
    from lip_sync       import get_audio_duration

    composer   = VideoComposer(args.width, args.height, args.fps)
    clips      = []
    temp_files = []
    tmp_dir    = tempfile.mkdtemp(prefix="ttv_")

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
            audio_path = os.path.join(tmp_dir, f"line_{line_idx:03d}_audio.mp3")
            temp_files.append(audio_path)
            audio_path = text_to_speech(text, output_path=audio_path, lang=args.lang)

            duration = get_audio_duration(audio_path) + args.pause

            # ── Wav2Lip (talking face) ─────────────────────────────────────────
            wav2lip_out = os.path.join(tmp_dir, f"line_{line_idx:03d}_wav2lip.mp4")
            temp_files.append(wav2lip_out)
            run_wav2lip(
                face_image_path=speaker_face,
                audio_path=audio_path,
                output_path=wav2lip_out,
                use_gan=not args.no_gan,
                fps=args.fps,
                pad_bottom=12,   # ensures chin/mouth not clipped
            )

            # ── compose segment ───────────────────────────────────────────────
            clip = composer.create_segment(
                speaker_idx=speaker_idx,
                dialogue_text=text,
                speaker_name=speaker,
                face_image_paths=ordered_faces,
                wav2lip_video_path=wav2lip_out,
                audio_path=audio_path,
            )
            clips.append(clip)

        # ── 6. Concatenate & export ───────────────────────────────────────────
        banner(f"Exporting final video → {args.output}")
        VideoComposer.concat_and_write(clips, args.output, fps=args.fps)

    finally:
        # Clean up temporary files
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
    from ollama_client     import OllamaClient, DEFAULT_APPEARANCES
    from character_renderer import CharacterRenderer
    from tts               import text_to_speech
    from lip_sync          import extract_mouth_openings
    import tempfile, numpy as np

    try:
        from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips  # v1
    except ImportError:
        from moviepy import VideoClip, AudioFileClip, concatenate_videoclips          # v2

    ollama     = OllamaClient(args.ollama_url)
    use_ollama = (not args.no_ollama) and ollama.is_available()

    renderers  = []
    for idx, name in enumerate(char_names):
        info = data["characters"][name]
        app  = (
            ollama.generate_appearance(name, info.get("role", "person"),
                                       info.get("gender", "neutral"), idx)
            if use_ollama
            else DEFAULT_APPEARANCES[idx % len(DEFAULT_APPEARANCES)]
        )
        renderers.append(CharacterRenderer(name, app, ["left", "right"][idx]))

    # Inline version of the old VideoComposer for cartoon frames
    from video_composer import _make_subtitle_image
    fps = args.fps
    W, H = args.width, args.height
    char_w = W // 2
    char_h = int(H * 0.85)
    sub_h  = H - char_h

    clips     = []
    tmp_files = []

    for line in dialogue:
        speaker     = line["speaker"]
        text        = line["text"]
        spk_idx     = char_names.index(speaker) if speaker in char_names else 0

        audio_path = tempfile.mktemp(suffix=".mp3")
        tmp_files.append(audio_path)
        audio_path  = text_to_speech(text, output_path=audio_path, lang=args.lang)
        mouth_vals  = list(extract_mouth_openings(audio_path, fps)) + [0.0] * int(fps * 0.3)
        n_frames    = len(mouth_vals)
        duration    = n_frames / fps

        frames_np = []
        for fi in range(n_frames):
            mo   = float(mouth_vals[fi])
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:] = [14, 17, 26]
            for i, r in enumerate(renderers):
                is_active = (i == spk_idx)
                char_img  = r.render(char_w, char_h, mo if is_active else 0.0, is_active)
                frame[:char_h, i * char_w:(i + 1) * char_w] = np.asarray(char_img)
            sub = _make_subtitle_image(W, sub_h, speaker, text)
            frame[char_h:, :] = sub
            frames_np.append(frame)

        def make_frame(t, _f=frames_np, _fps=fps):
            return _f[min(int(t * _fps), len(_f) - 1)]

        clip = VideoClip(make_frame, duration=duration)
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
