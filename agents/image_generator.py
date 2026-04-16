"""Image Generator Agent — generates one image per scene.

Backends:
  huggingface  — HuggingFace Inference API, FREE (default)
                 Uses FLUX.1-schnell or any HF-hosted model.
                 Optional: set HUGGINGFACE_API_KEY for higher rate limits.
  dalle        — OpenAI DALL-E 3 (paid, requires OPENAI_API_KEY)
  stability    — Stability AI (paid, requires STABILITY_API_KEY)
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings
from models.production import ImageAsset
from models.storyboard import Storyboard


class ImageGeneratorAgent:
    def __init__(self, settings: Settings) -> None:
        self.backend = settings.image_backend
        self.settings = settings
        self.output_dir = settings.output_subdirs["images"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.backend == "dalle":
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            )
        elif self.backend == "stability":
            self.stability_api_key = (
                settings.stability_api_key.get_secret_value()
                if settings.stability_api_key
                else None
            )
        # huggingface backend uses requests directly (no SDK needed)

    def run(self, storyboard: Storyboard) -> list[ImageAsset]:
        assets: list[ImageAsset] = []
        for scene in storyboard.scenes:
            print(f"  [ImageGen] Generating image for scene {scene.scene_id} ({self.backend})…")
            image_bytes, revised_prompt = self._generate(scene.visual_description)
            file_path = self.output_dir / f"scene_{scene.scene_id:03d}.png"
            file_path.write_bytes(image_bytes)
            assets.append(
                ImageAsset(
                    scene_id=scene.scene_id,
                    file_path=file_path,
                    prompt_used=scene.visual_description,
                    backend=self.backend,
                    revised_prompt=revised_prompt,
                )
            )
            print(f"  [ImageGen] Saved {file_path}")
        return assets

    def _generate(self, prompt: str) -> tuple[bytes, Optional[str]]:
        if self.backend == "dalle":
            return self._generate_dalle(prompt)
        if self.backend == "stability":
            return self._generate_stability(prompt)
        return self._generate_huggingface(prompt)

    # ──────────────────────────────────────────────────────────────────
    # HuggingFace Inference API — FREE
    # Model: FLUX.1-schnell (fast, high quality, open source)
    # Docs: https://huggingface.co/black-forest-labs/FLUX.1-schnell
    # ──────────────────────────────────────────────────────────────────
    def _generate_huggingface(self, prompt: str) -> tuple[bytes, None]:
        import requests

        model = self.settings.huggingface_image_model
        api_url = f"https://api-inference.huggingface.co/models/{model}"

        headers: dict = {"Content-Type": "application/json"}
        if self.settings.huggingface_api_key:
            headers["Authorization"] = f"Bearer {self.settings.huggingface_api_key}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "width": 1344,    # closest FLUX supports to 16:9 HD
                "height": 768,
                "num_inference_steps": 4,  # FLUX.1-schnell is optimised for 4 steps
                "guidance_scale": 0.0,
            },
        }

        # HuggingFace can return 503 while the model warms up — retry up to 3 times
        for attempt in range(1, 4):
            resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
            if resp.status_code == 503:
                wait = 20 * attempt
                print(f"  [ImageGen] HuggingFace model loading, retrying in {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            # Response is raw image bytes (JPEG or PNG)
            return resp.content, None

        raise RuntimeError(f"HuggingFace model {model} unavailable after 3 attempts")

    # ──────────────────────────────────────────────────────────────────
    # DALL-E 3 (paid)
    # ──────────────────────────────────────────────────────────────────
    def _generate_dalle(self, prompt: str) -> tuple[bytes, str]:
        response = self.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="hd",
            n=1,
            response_format="b64_json",
        )
        b64 = response.data[0].b64_json
        revised = response.data[0].revised_prompt or prompt
        return base64.b64decode(b64), revised

    # ──────────────────────────────────────────────────────────────────
    # Stability AI (paid)
    # ──────────────────────────────────────────────────────────────────
    def _generate_stability(self, prompt: str) -> tuple[bytes, None]:
        import requests

        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers={
                "Authorization": f"Bearer {self.stability_api_key}",
                "Accept": "image/*",
            },
            files={"none": ""},
            data={
                "prompt": prompt,
                "output_format": "png",
                "aspect_ratio": "16:9",
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.content, None
