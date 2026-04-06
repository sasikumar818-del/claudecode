"""Image Generator Agent — generates one image per scene via DALL-E 3 or Stability AI."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings
from models.production import ImageAsset
from models.storyboard import Storyboard


class ImageGeneratorAgent:
    def __init__(self, settings: Settings) -> None:
        self.backend = settings.image_backend
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
        return self._generate_stability(prompt)

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
