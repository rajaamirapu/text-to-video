#!/usr/bin/env bash
# install.sh — set up the text-to-video environment
# Run once:  bash install.sh

set -e

echo "──────────────────────────────────────────────"
echo "  Text-to-Video installer"
echo "──────────────────────────────────────────────"

# Check Python
python3 --version || { echo "[Error] Python 3 not found"; exit 1; }

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "Quick start:"
echo "  python main.py example_dialogue.json"
echo ""
echo "Options:"
echo "  --no-ollama          skip Ollama (use default character looks)"
echo "  --ollama-url URL     custom Ollama server (default: http://localhost:11434)"
echo "  --output FILE        output video name (default: output.mp4)"
echo "  --fps N              frame rate (default: 24)"
echo "  --lang CODE          TTS language, e.g. en es fr de (default: en)"
echo ""
echo "Make sure Ollama is running (ollama serve) for AI-driven character generation."
