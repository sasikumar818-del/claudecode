"""Image Generator Agent — generates one image per scene.

Backends (in order of preference for zero-cost use):
  huggingface — HuggingFace Inference API, FLUX.1-schnell (free tier)
  pil         — offline gradient card (no internet, no API key)
  dalle       — OpenAI DALL-E 3 (paid)
  stability   — Stability AI (paid)
"""
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Optional

from config.settings import Settings
from models.production import ImageAsset
from models.storyboard import Storyboard

# 7 cinematic color palettes for PIL fallback
_PALETTES = [
    ((25, 25, 112), (255, 165, 0)),    # midnight blue → amber
    ((0, 51, 102), (0, 204, 153)),     # deep navy → teal
    ((80, 0, 120), (255, 100, 180)),   # violet → rose
    ((20, 60, 20), (200, 255, 100)),   # forest → lime
    ((120, 20, 20), (255, 200, 80)),   # burgundy → gold
    ((0, 80, 120), (180, 230, 255)),   # ocean → sky
    ((60, 40, 0), (255, 210, 120)),    # earth → wheat
]


class ImageGeneratorAgent:
    def __init__(self, settings: Settings) -> None:
        self.backend = settings.image_backend
        self.settings = settings
        self.output_dir = settings.output_subdirs["images"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.backend == "dalle":
            if not settings.openai_api_key:
                raise ValueError("openai_api_key is required for image_backend='dalle'")
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            )
        elif self.backend == "stability":
            if not settings.stability_api_key:
                raise ValueError("stability_api_key is required for image_backend='stability'")
            self.stability_api_key = settings.stability_api_key.get_secret_value()
        elif self.backend == "huggingface":
            self.hf_api_key = settings.huggingface_api_key
            self.hf_model = settings.huggingface_image_model

    def run(self, storyboard: Storyboard) -> list[ImageAsset]:
        assets: list[ImageAsset] = []
        for scene in storyboard.scenes:
            print(f"  [ImageGen] Scene {scene.scene_id} → {self.backend}…")
            image_bytes, revised_prompt = self._generate(
                scene.visual_description, scene.scene_id
            )
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
            print(f"  [ImageGen] Saved {file_path.name}")
        return assets

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, scene_id: int = 1) -> tuple[bytes, Optional[str]]:
        if self.backend == "pil":
            return self._generate_pil(prompt, scene_id), None
        if self.backend == "huggingface":
            try:
                return self._generate_huggingface(prompt), None
            except Exception as exc:
                print(f"  [ImageGen] HuggingFace failed ({exc}), falling back to PIL")
                return self._generate_pil(prompt, scene_id), None
        if self.backend == "dalle":
            return self._generate_dalle(prompt)
        if self.backend == "stability":
            return self._generate_stability(prompt), None
        raise ValueError(f"Unknown image backend: {self.backend}")

    # ------------------------------------------------------------------
    # HuggingFace Inference API (free tier)
    # ------------------------------------------------------------------

    def _generate_huggingface(self, prompt: str) -> bytes:
        import requests

        api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"

        resp = requests.post(
            api_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"width": 1024, "height": 576, "num_inference_steps": 4},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # PIL gradient card (offline fallback, zero-cost)
    # ------------------------------------------------------------------

    def _generate_pil(self, prompt: str, scene_id: int) -> bytes:
        from PIL import Image, ImageDraw, ImageFont

        palette_idx = (scene_id - 1) % len(_PALETTES)
        color_start, color_end = _PALETTES[palette_idx]

        W, H = 1920, 1080
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        # Horizontal gradient
        for x in range(W):
            r = int(color_start[0] + (color_end[0] - color_start[0]) * x / W)
            g = int(color_start[1] + (color_end[1] - color_start[1]) * x / W)
            b = int(color_start[2] + (color_end[2] - color_start[2]) * x / W)
            draw.line([(x, 0), (x, H)], fill=(r, g, b))

        # Decorative accent circles
        seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        rng = seed
        for _ in range(5):
            rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF
            cx = (rng >> 16) % W
            rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF
            cy = (rng >> 16) % H
            rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF
            radius = 80 + (rng >> 16) % 200
            alpha = 30 + (rng >> 8) % 40
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(*color_end, alpha),
            )
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

        # Scene label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
            small_font = font

        label = f"Scene {scene_id}"
        draw.text((W // 2, H // 2 - 60), label, font=font, fill="white", anchor="mm")

        # Wrap prompt text
        words = prompt[:120].split()
        lines, line = [], []
        for w in words:
            if len(" ".join(line + [w])) <= 60:
                line.append(w)
            else:
                lines.append(" ".join(line))
                line = [w]
        if line:
            lines.append(" ".join(line))

        for i, ln in enumerate(lines[:3]):
            draw.text(
                (W // 2, H // 2 + 20 + i * 40),
                ln, font=small_font, fill=(220, 220, 220), anchor="mm",
            )

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # DALL-E 3 (paid)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stability AI (paid)
    # ------------------------------------------------------------------

    def _generate_stability(self, prompt: str) -> bytes:
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
        return response.content
