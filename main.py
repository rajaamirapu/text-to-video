#!/usr/bin/env python3
"""
main.py — Text-to-Video with Ollama
=====================================

Usage
-----
  python main.py example_dialogue.json
  python main.py my_script.json --output my_video.mp4 --fps 24
  python main.py my_script.json --no-ollama          # skip Ollama, use default looks
  python main.py my_script.json --ollama-url http://192.168.1.10:11434

Dialogue script format (JSON)
------------------------------
  {
    "characters": {
      "Alice": { "role": "doctor",  "gender": "female" },
      "Bob":   { "role": "patient", "gender": "male"   }
    },
    "dialogue": [
      { "speaker": "Alice", "text": "Good morning! How are you feeling today?" },
      { "speaker": "Bob",   "text": "I've been having some headaches lately."  }
    ]
  }

Pipeline
--------
  1. Load dialogue JSON
  2. For each character, call Ollama LLM to generate appearance (skin / hair / eyes / shirt)
  3. For each dialogue line:
       a. gTTS → mp3 audio file
       b. librosa → per-frame mouth-opening values (lip-sync)
       c. Render frames (PIL portraits with animated mouth)
       d. Create MoviePy clip
  4. Concatenate all clips → final MP4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def load_script(path: str) -> dict:
    """Load and validate the dialogue JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "characters" not in data or "dialogue" not in data:
        sys.exit(
            "[Error] Script must have top-level keys 'characters' and 'dialogue'."
        )

    chars = list(data["characters"].keys())
    if len(chars) < 2:
        sys.exit("[Error] Need at least two characters in 'characters'.")

    for i, line in enumerate(data["dialogue"]):
        if line["speaker"] not in chars:
            print(
                f"[Warning] Line {i}: speaker '{line['speaker']}' "
                f"not in characters list — will be assigned to character 0."
            )

    return data


def banner(msg: str):
    w = min(72, len(msg) + 4)
    print("─" * w)
    print(f"  {msg}")
    print("─" * w)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video from a dialogue script using Ollama."
    )
    parser.add_argument("script",          help="Path to dialogue JSON file")
    parser.add_argument(
        "--output", "-o", default="output.mp4",
        help="Output video file (default: output.mp4)"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--no-ollama", action="store_true",
        help="Skip Ollama; use built-in default character appearances"
    )
    parser.add_argument("--fps",    type=int, default=24, help="Video frame rate")
    parser.add_argument("--width",  type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720,  help="Video height")
    parser.add_argument("--lang",   default="en", help="TTS language code (e.g. en, es, fr)")
    parser.add_argument(
        "--pause", type=float, default=0.4,
        help="Silent pause between lines in seconds (default: 0.4)"
    )
    args = parser.parse_args()

    # ── 1. Load script ────────────────────────────────────────────────────────
    banner("Loading dialogue script")
    data        = load_script(args.script)
    char_names  = list(data["characters"].keys())[:2]   # max 2 for now
    dialogue    = data["dialogue"]
    print(f"  Characters : {char_names}")
    print(f"  Lines      : {len(dialogue)}")

    # ── 2. Initialise modules ─────────────────────────────────────────────────
    from ollama_client   import OllamaClient
    from character_renderer import CharacterRenderer
    from tts             import text_to_speech
    from lip_sync        import extract_mouth_openings
    from video_composer  import VideoComposer

    ollama = OllamaClient(args.ollama_url)

    # ── 3. Generate appearances ───────────────────────────────────────────────
    banner("Generating character appearances")
    renderers: list[CharacterRenderer] = []
    positions = ["left", "right"]

    use_ollama = (not args.no_ollama) and ollama.is_available()
    if use_ollama:
        print(f"  Ollama reachable at {args.ollama_url}  (model: {ollama.get_best_model()})")
    else:
        reason = "disabled by --no-ollama" if args.no_ollama else "not reachable"
        print(f"  Ollama {reason} — using built-in default appearances")

    for idx, name in enumerate(char_names):
        info = data["characters"][name]
        appearance = None

        if use_ollama:
            appearance = ollama.generate_appearance(
                name=name,
                role=info.get("role", "person"),
                gender=info.get("gender", "neutral"),
                default_idx=idx,
            )
        else:
            from ollama_client import DEFAULT_APPEARANCES
            appearance = DEFAULT_APPEARANCES[idx % len(DEFAULT_APPEARANCES)]

        renderer = CharacterRenderer(name, appearance=appearance, position=positions[idx])
        renderers.append(renderer)
        print(f"  ✓ {name} — skin {renderer.skin}, hair {renderer.hair}")

    # ── 4. Build video clips ──────────────────────────────────────────────────
    banner(f"Generating {len(dialogue)} dialogue segments")
    composer   = VideoComposer(args.width, args.height, args.fps)
    clips      = []
    temp_files = []
    pause_frames = max(1, int(args.fps * args.pause))

    for line_idx, line in enumerate(dialogue):
        speaker      = line["speaker"]
        text         = line["text"]
        speaker_idx  = (
            char_names.index(speaker) if speaker in char_names else 0
        )

        print(
            f"\n  [{line_idx + 1}/{len(dialogue)}] "
            f"{speaker}: \"{text[:55]}{'…' if len(text) > 55 else ''}\""
        )

        # ── TTS ──────────────────────────────────────────────────────────────
        audio_path = tempfile.mktemp(suffix=".mp3")
        temp_files.append(audio_path)
        audio_path = text_to_speech(text, output_path=audio_path, lang=args.lang)

        # ── Lip-sync ─────────────────────────────────────────────────────────
        mouth_openings: list[float] = extract_mouth_openings(audio_path, fps=args.fps)
        print(f"  [LipSync] {len(mouth_openings)} frames ({len(mouth_openings)/args.fps:.2f}s)")

        # Append a short closed-mouth pause at the end of each line
        mouth_openings = list(mouth_openings) + [0.0] * pause_frames

        # ── Video segment ────────────────────────────────────────────────────
        clip = composer.create_segment(
            speaker_idx=speaker_idx,
            dialogue_text=text,
            speaker_name=speaker,
            renderers=renderers,
            mouth_openings=mouth_openings,
            audio_path=audio_path,
        )
        clips.append(clip)

    # ── 5. Concatenate & export ───────────────────────────────────────────────
    banner(f"Exporting → {args.output}")
    try:
        from moviepy.editor import concatenate_videoclips  # type: ignore  # v1
    except ImportError:
        from moviepy import concatenate_videoclips  # type: ignore  # v2

    final = concatenate_videoclips(clips, method="compose")
    import inspect, tempfile as _tf
    _tmp_audio = os.path.join(_tf.gettempdir(), "ttv_temp_audio.m4a")
    write_kwargs = dict(
        fps=args.fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=_tmp_audio,
        remove_temp=False,          # we'll clean up ourselves
        logger="bar",
    )
    sig = inspect.signature(final.write_videofile)
    if "verbose" in sig.parameters:
        write_kwargs["verbose"] = False   # moviepy v1
    try:
        final.write_videofile(args.output, **write_kwargs)
    finally:
        try:
            if os.path.exists(_tmp_audio):
                os.unlink(_tmp_audio)
        except Exception:
            pass

    # ── 6. Cleanup temp audio files ───────────────────────────────────────────
    for f in temp_files:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except Exception:
            pass

    banner(f"Done!  Video saved to: {os.path.abspath(args.output)}")
    print(f"  Duration  : {final.duration:.1f} s")
    print(f"  Resolution: {args.width}×{args.height} @ {args.fps} fps")


if __name__ == "__main__":
    main()
