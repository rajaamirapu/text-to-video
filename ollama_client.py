"""
ollama_client.py
Ollama API client for:
  - Character appearance generation (LLM → JSON)
  - Optional image generation (if the running model supports it)
"""

import json
import base64
import io
import requests
from PIL import Image


# ─────────────────────────────────────────────
# Sensible fallback appearances (used when
# Ollama is not reachable or returns bad JSON)
# ─────────────────────────────────────────────
DEFAULT_APPEARANCES = [
    {
        "skin_rgb": [255, 220, 185],
        "hair_rgb": [80, 50, 20],
        "eye_rgb": [70, 130, 180],
        "shirt_rgb": [50, 100, 160],
        "hair_style": "medium",
        "gender": "female",
    },
    {
        "skin_rgb": [200, 160, 120],
        "hair_rgb": [30, 20, 10],
        "eye_rgb": [60, 40, 20],
        "shirt_rgb": [90, 55, 20],
        "hair_style": "short",
        "gender": "male",
    },
]


class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._cached_model: str | None = None

    # ── connectivity ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the Ollama daemon is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=4)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[dict]:
        """Return the list of locally-installed Ollama models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.json().get("models", [])
        except Exception:
            return []

    def get_best_model(self) -> str:
        """Pick the best available model (prefer llama3 variants, else first)."""
        if self._cached_model:
            return self._cached_model
        models = self.list_models()
        if not models:
            return "llama3.2"
        names = [m["name"] for m in models]
        for preferred in ("llama3.2", "llama3", "llama2", "mistral", "gemma"):
            for n in names:
                if preferred in n.lower():
                    self._cached_model = n
                    return n
        self._cached_model = names[0]
        return names[0]

    # ── character appearance (LLM-driven) ───────────────────────────────────

    def generate_appearance(
        self,
        name: str,
        role: str = "person",
        gender: str = "neutral",
        default_idx: int = 0,
    ) -> dict:
        """
        Ask the Ollama LLM to describe the character's appearance.
        Returns a dict with keys: skin_rgb, hair_rgb, eye_rgb,
        shirt_rgb, hair_style, gender.
        Falls back to DEFAULT_APPEARANCES on any error.
        """
        prompt = (
            f"You are a visual character designer. Create appearance details for a "
            f"character named {name} who is a {role} (gender: {gender}).\n"
            f"Return ONLY valid JSON (no markdown, no extra text) with these exact fields:\n"
            f'{{\n'
            f'  "skin_rgb": [r, g, b],\n'
            f'  "hair_rgb": [r, g, b],\n'
            f'  "eye_rgb": [r, g, b],\n'
            f'  "shirt_rgb": [r, g, b],\n'
            f'  "hair_style": "short|medium|long",\n'
            f'  "gender": "male|female|neutral"\n'
            f'}}\n'
            f"Use realistic RGB values (0-255). Hair, eyes and clothing must contrast nicely."
        )

        if not self.is_available():
            print(f"  [Ollama] Not reachable – using default appearance for {name}")
            return DEFAULT_APPEARANCES[default_idx % len(DEFAULT_APPEARANCES)]

        model = self.get_best_model()
        print(f"  [Ollama] Generating appearance for '{name}' using model '{model}' …")

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            text = r.json().get("response", "")

            # Extract JSON blob from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                # Validate & clamp RGB values
                for key in ("skin_rgb", "hair_rgb", "eye_rgb", "shirt_rgb"):
                    if key in data and isinstance(data[key], list) and len(data[key]) == 3:
                        data[key] = [max(0, min(255, int(v))) for v in data[key]]
                    else:
                        # Fall back per-key
                        data[key] = DEFAULT_APPEARANCES[default_idx % len(DEFAULT_APPEARANCES)][key]
                print(f"  [Ollama] Appearance generated successfully for '{name}'")
                return data
        except Exception as e:
            print(f"  [Ollama] Appearance generation error: {e}")

        return DEFAULT_APPEARANCES[default_idx % len(DEFAULT_APPEARANCES)]

    # ── image generation (optional / model-dependent) ───────────────────────

    def try_generate_image(self, prompt: str) -> Image.Image | None:
        """
        Attempt to generate an image via Ollama (works only with models that
        support image generation, e.g. a GGUF Stable Diffusion model loaded
        through Ollama). Returns a PIL Image or None if not supported.
        """
        if not self.is_available():
            return None

        model = self.get_best_model()
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            data = r.json()
            images = data.get("images", [])
            if images:
                img_bytes = base64.b64decode(images[0])
                return Image.open(io.BytesIO(img_bytes))
        except Exception:
            pass
        return None
