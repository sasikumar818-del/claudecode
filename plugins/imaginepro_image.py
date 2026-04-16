"""ImaginePro Image Generator — Midjourney-quality images via the ImaginePro API.

Uses the ImaginePro REST API (Midjourney endpoint) to generate one image per
storyboard scene, then saves each as a PNG file.

Set in .env:
    USE_IMAGINEPRO_IMAGE=true
    IMAGINEPRO_API_KEY=your_api_key_here
    IMAGINEPRO_API_BASE=https://api.imaginepro.ai/api/v1
"""
from __future__ import annotations

import time
from typing import Any

from plugins.base import AgentPlugin
from models.production import ImageAsset
from models.storyboard import Storyboard

# Poll configuration
_POLL_INTERVAL = 5       # seconds between status checks
_MAX_POLLS = 24          # 24 × 5s = 120s max wait per image


class ImagineProImageGenerator(AgentPlugin):
    """Replaces the DALL-E / Stable Diffusion image generator with ImaginePro."""

    @property
    def name(self) -> str:
        return "imaginepro_image_generator"

    @property
    def replaces(self) -> str:
        return "Image Generator"

    def setup(self, settings: Any) -> None:
        api_key = settings.imaginepro_api_key
        if api_key is None:
            raise ValueError("IMAGINEPRO_API_KEY must be set when USE_IMAGINEPRO_IMAGE=true")
        self._api_key = api_key.get_secret_value()
        self._base_url = settings.imaginepro_api_base.rstrip("/")
        self._output_dir = settings.output_subdirs["images"]
        self._output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [ImaginePro] Image generator ready (base: {self._base_url})")

    def run(self, input_data: Storyboard) -> list[ImageAsset]:
        assets: list[ImageAsset] = []
        for scene in input_data.scenes:
            print(f"  [ImaginePro] Generating image — scene {scene.scene_id}…")
            image_bytes = self._generate_image(scene.visual_description)
            out = self._output_dir / f"scene_{scene.scene_id:03d}.png"
            out.write_bytes(image_bytes)
            assets.append(
                ImageAsset(
                    scene_id=scene.scene_id,
                    file_path=out,
                    prompt_used=scene.visual_description,
                    backend="imaginepro",
                )
            )
            print(f"  [ImaginePro] Image saved → {out}")
        return assets

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _generate_image(self, prompt: str) -> bytes:
        import requests

        # Submit the imagine task
        r = requests.post(
            f"{self._base_url}/midjourney/imagine",
            json={"prompt": prompt},
            headers=self._headers(),
            timeout=30,
        )
        r.raise_for_status()
        task_id = r.json()["result"]["taskId"]
        print(f"    [ImaginePro] Task submitted: {task_id}")

        # Poll until completed or failed
        for attempt in range(_MAX_POLLS):
            time.sleep(_POLL_INTERVAL)
            r = requests.get(
                f"{self._base_url}/midjourney/task/{task_id}/fetch",
                headers=self._headers(),
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()["result"]
            status = data.get("status", "")

            if status == "completed":
                img_url = data["uri"]
                print(f"    [ImaginePro] Task {task_id} complete, downloading…")
                img_r = requests.get(img_url, timeout=60)
                img_r.raise_for_status()
                return img_r.content

            if status == "failed":
                reason = data.get("failReason", "unknown")
                raise RuntimeError(
                    f"ImaginePro task {task_id} failed: {reason}"
                )

            print(
                f"    [ImaginePro] Waiting… ({attempt + 1}/{_MAX_POLLS}) "
                f"status={status}"
            )

        raise TimeoutError(
            f"ImaginePro task {task_id} did not complete within "
            f"{_MAX_POLLS * _POLL_INTERVAL}s"
        )
