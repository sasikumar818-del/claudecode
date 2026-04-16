"""Local Image Generator — Stable Diffusion via diffusers (GPU/CPU).

Install dependencies:
    pip install diffusers transformers accelerate torch

Without GPU the first run downloads the model (~4 GB) and is slow.
With CUDA/MPS it runs in ~5-15 s per image.

Falls back to a PIL placeholder if diffusers/torch are not installed,
so the pipeline always produces output regardless of hardware.

Set in .env:
    USE_LOCAL_IMAGE=true
    SD_MODEL_ID=runwayml/stable-diffusion-v1-5   # or any HF model ID
    SD_DEVICE=auto                                # auto | cuda | mps | cpu
    SD_STEPS=20
    SD_GUIDANCE_SCALE=7.5
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Optional

from plugins.base import AgentPlugin
from models.production import ImageAsset
from models.storyboard import Storyboard

# Module-level SD pipeline singleton — loaded once, reused across jobs.
_SD_PIPE = None
_SD_DEVICE: Optional[str] = None


def _load_sd(model_id: str, device: str, steps: int, guidance: float):
    """Load (or return cached) Stable Diffusion pipeline."""
    global _SD_PIPE, _SD_DEVICE
    if _SD_PIPE is not None:
        return _SD_PIPE

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.float16 if device != "cpu" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # Optimisations
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    _SD_PIPE = pipe
    _SD_DEVICE = device
    return pipe


class LocalSDImageGenerator(AgentPlugin):
    """Replaces DALL-E / Stability AI with a local Stable Diffusion pipeline."""

    @property
    def name(self) -> str:
        return "local_sd_image_generator"

    @property
    def replaces(self) -> str:
        return "Image Generator"

    def setup(self, settings: Any) -> None:
        self._output_dir = settings.output_subdirs["images"]
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._model_id = getattr(settings, "sd_model_id", "runwayml/stable-diffusion-v1-5")
        self._device = getattr(settings, "sd_device", "auto")
        self._steps = getattr(settings, "sd_steps", 20)
        self._guidance = getattr(settings, "sd_guidance_scale", 7.5)
        self._pipe = None

        try:
            self._pipe = _load_sd(self._model_id, self._device, self._steps, self._guidance)
            print(f"  [LocalImage] Stable Diffusion ready on {_SD_DEVICE}")
        except ImportError:
            print("  [LocalImage] diffusers/torch not installed — using placeholder images")
        except Exception as exc:
            print(f"  [LocalImage] SD load failed ({exc}) — using placeholder images")

    def run(self, input_data: Storyboard) -> list[ImageAsset]:
        assets: list[ImageAsset] = []
        for scene in input_data.scenes:
            print(f"  [LocalImage] Generating scene {scene.scene_id}…")
            image_bytes = (
                self._generate_sd(scene.visual_description)
                if self._pipe
                else _placeholder(scene.visual_description)
            )
            backend = "stable_diffusion" if self._pipe else "placeholder"
            out = self._output_dir / f"scene_{scene.scene_id:03d}.png"
            out.write_bytes(image_bytes)
            assets.append(
                ImageAsset(
                    scene_id=scene.scene_id,
                    file_path=out,
                    prompt_used=scene.visual_description,
                    backend=backend,
                )
            )
            print(f"  [LocalImage] Saved {out} ({backend})")
        return assets

    def _generate_sd(self, prompt: str) -> bytes:
        image = self._pipe(
            prompt=prompt,
            width=1280,
            height=720,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance,
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


def _placeholder(prompt: str) -> bytes:
    """Generate a simple coloured image with the prompt text (PIL only)."""
    from PIL import Image, ImageDraw

    # Derive a deterministic colour from the prompt
    h = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
    r = max(30, (h >> 16) & 0x7F)
    g = max(30, (h >> 8) & 0x7F)
    b = max(60, h & 0x9F)

    img = Image.new("RGB", (1280, 720), color=(r, g, b))
    draw = ImageDraw.Draw(img)

    # Word-wrap the prompt across the centre of the image
    words, lines, line = prompt.split(), [], []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > 55:
            lines.append(" ".join(line[:-1]))
            line = [word]
    if line:
        lines.append(" ".join(line))

    y = 340 - len(lines) * 20
    for text in lines[:8]:
        draw.text((640, y), text, fill=(220, 220, 220), anchor="mm")
        y += 42

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
