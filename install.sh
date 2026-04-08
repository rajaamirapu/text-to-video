#!/usr/bin/env bash
# install.sh — set up the text-to-video environment
# Run once:  bash install.sh

set -e

echo "──────────────────────────────────────────────"
echo "  Text-to-Video installer"
echo "──────────────────────────────────────────────"

# Check Python
python3 --version || { echo "[Error] Python 3 not found"; exit 1; }

# Install all dependencies
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "Quick start:"
echo "  python main.py example_dialogue.json"
echo ""
echo "Options:"
echo "  --output FILE        output video name (default: output.mp4)"
echo "  --fps N              frame rate (default: 25)"
echo "  --lang CODE          TTS language, e.g. en es fr de (default: en)"
echo "  --sd-model MODEL     Stable Diffusion model (default: runwayml/stable-diffusion-v1-5)"
echo "  --regen-faces        force regenerate face images"
echo "  --regen-room         force regenerate room background"
echo ""
echo "TTS voice quality (best → fallback):"
echo "  1. edge-tts   Microsoft neural voices (en-US-GuyNeural etc.)  ← best"
echo "  2. pyttsx3    System voices (offline)"
echo ""
