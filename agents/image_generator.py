"""Image Generator Agent — generates one cinematic image per scene.

Backends:
  pil         — offline cinematic scene renderer (atmosphere + silhouettes + vignette)
  huggingface — HuggingFace Inference API, FLUX.1-schnell (free, needs internet)
  dalle       — OpenAI DALL-E 3 (paid)
  stability   — Stability AI (paid)
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path
from typing import Optional

from config.settings import Settings
from models.production import ImageAsset
from models.storyboard import Storyboard

# ── Per-scene cinematic themes ────────────────────────────────────────────────
#   sky_top / sky_bot: vertical sky gradient colours (RGB)
#   ground:            ground strip colour
#   sun:               glow source colour (None = no glow)
#   accent:            highlight / prop colour
#   mood:              "sunrise" | "morning" | "midday" | "golden" | "sunset"
_SCENE_THEMES = [
    # Scene 1 — Morning Arrival
    {
        "sky_top":  (18,  12,  55),   "sky_bot":  (255, 140, 50),
        "ground":   (35,  90,  30),   "sun":      (255, 220, 100),
        "accent":   (255, 200, 80),   "mood":     "sunrise",
        "silhouette": "arrival",
    },
    # Scene 2 — Playground Gates
    {
        "sky_top":  (60, 140, 210),   "sky_bot":  (180, 225, 255),
        "ground":   (210, 195, 135),  "sun":      (255, 240, 180),
        "accent":   (230,  60,  60),  "mood":     "morning",
        "silhouette": "gates",
    },
    # Scene 3 — Slides & Swings
    {
        "sky_top":  (240, 140,  30),  "sky_bot":  (255, 220, 100),
        "ground":   (60, 160,  60),   "sun":      None,
        "accent":   (60,  80, 220),   "mood":     "midday",
        "silhouette": "playground",
    },
    # Scene 4 — Team Games (aerial)
    {
        "sky_top":  (30, 160,  50),   "sky_bot":  (80, 200, 80),
        "ground":   (40, 140,  45),   "sun":      None,
        "accent":   (255, 255, 255),  "mood":     "aerial",
        "silhouette": "field",
    },
    # Scene 5 — Snack Break
    {
        "sky_top":  (180, 140,  40),  "sky_bot":  (255, 200, 100),
        "ground":   (80, 130,  50),   "sun":      (255, 210,  80),
        "accent":   (200,  80,  60),  "mood":     "golden",
        "silhouette": "tree",
    },
    # Scene 6 — Creative Corner
    {
        "sky_top":  (250, 230, 180),  "sky_bot":  (255, 255, 220),
        "ground":   (230, 200, 150),  "sun":      None,
        "accent":   (80, 160, 220),   "mood":     "warm",
        "silhouette": "crafts",
    },
    # Scene 7 — Grand Finale
    {
        "sky_top":  (40,  10,  80),   "sky_bot":  (220, 100,  30),
        "ground":   (20,  20,  40),   "sun":      (255, 180,  60),
        "accent":   (255, 220,  50),  "mood":     "sunset",
        "silhouette": "finale",
    },
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
            self.openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        elif self.backend == "stability":
            if not settings.stability_api_key:
                raise ValueError("stability_api_key is required for image_backend='stability'")
            self.stability_api_key = settings.stability_api_key.get_secret_value()
        elif self.backend == "huggingface":
            self.hf_api_key = settings.huggingface_api_key
            self.hf_model   = settings.huggingface_image_model

    def run(self, storyboard: Storyboard) -> list[ImageAsset]:
        assets: list[ImageAsset] = []
        for scene in storyboard.scenes:
            print(f"  [ImageGen] Scene {scene.scene_id} → {self.backend}…")
            image_bytes, revised = self._generate(scene.visual_description, scene.scene_id)
            file_path = self.output_dir / f"scene_{scene.scene_id:03d}.png"
            file_path.write_bytes(image_bytes)
            assets.append(ImageAsset(
                scene_id=scene.scene_id, file_path=file_path,
                prompt_used=scene.visual_description, backend=self.backend,
                revised_prompt=revised,
            ))
            print(f"  [ImageGen] Saved {file_path.name}")
        return assets

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _generate(self, prompt: str, scene_id: int = 1) -> tuple[bytes, Optional[str]]:
        if self.backend == "pil":
            return self._render_cinematic(prompt, scene_id), None
        if self.backend == "huggingface":
            try:
                return self._generate_huggingface(prompt), None
            except Exception as exc:
                print(f"  [ImageGen] HuggingFace failed ({exc}), using cinematic PIL")
                return self._render_cinematic(prompt, scene_id), None
        if self.backend == "dalle":
            return self._generate_dalle(prompt)
        if self.backend == "stability":
            return self._generate_stability(prompt), None
        raise ValueError(f"Unknown image backend: {self.backend}")

    # ── Cinematic PIL renderer ────────────────────────────────────────────────

    def _render_cinematic(self, prompt: str, scene_id: int) -> bytes:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
        import numpy as np

        W, H = 1920, 1080
        theme = _SCENE_THEMES[(scene_id - 1) % len(_SCENE_THEMES)]
        stype = theme["silhouette"]
        rng   = self._rng(prompt)

        # ── 1. Sky gradient ──────────────────────────────────────────────────
        sky_top = theme["sky_top"]
        sky_bot = theme["sky_bot"]
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for row in range(H):
            t = row / H
            arr[row, :, 0] = int(sky_top[0] + (sky_bot[0] - sky_top[0]) * t)
            arr[row, :, 1] = int(sky_top[1] + (sky_bot[1] - sky_top[1]) * t)
            arr[row, :, 2] = int(sky_top[2] + (sky_bot[2] - sky_top[2]) * t)

        # ── 2. Ground band (bottom 28%) ──────────────────────────────────────
        gr = theme["ground"]
        ground_start = int(H * 0.72)
        for row in range(ground_start, H):
            blend = (row - ground_start) / (H - ground_start)
            arr[row, :, 0] = int(gr[0] * (1 - blend * 0.3))
            arr[row, :, 1] = int(gr[1] * (1 - blend * 0.3))
            arr[row, :, 2] = int(gr[2] * (1 - blend * 0.3))

        img = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        # ── 3. Atmospheric sun/glow ──────────────────────────────────────────
        if theme["sun"]:
            cx = W // 2 + rng(- W // 5, W // 5)
            cy = ground_start - rng(50, 180)
            sr = theme["sun"]
            for radius in range(300, 10, -20):
                alpha = max(0, int(55 - (300 - radius) * 0.18))
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=(*sr, alpha)
                )
            draw.ellipse([cx - 55, cy - 55, cx + 55, cy + 55], fill=(*sr, 240))

        # ── 4. Scene silhouettes ─────────────────────────────────────────────
        ac = theme["accent"]
        self._draw_silhouette(draw, stype, W, H, ground_start, ac, rng)

        # ── 5. Bokeh particles ───────────────────────────────────────────────
        self._draw_bokeh(draw, W, H, ac, rng, n=25)

        # ── 6. Vignette (dark corners) ───────────────────────────────────────
        img = self._add_vignette(img, W, H)

        # ── 7. Cinematic letterbox bars ──────────────────────────────────────
        draw2 = ImageDraw.Draw(img, "RGBA")
        bar_h = int(H * 0.055)
        draw2.rectangle([0, 0, W, bar_h], fill=(0, 0, 0, 240))
        draw2.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 240))

        # ── 8. Text overlay (Tamil title + scene number) ─────────────────────
        self._draw_text(draw2, prompt, scene_id, W, H, bar_h)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    # ── Silhouette renderers ──────────────────────────────────────────────────

    def _draw_silhouette(self, draw, stype: str, W, H, gs, ac, rng) -> None:
        if stype == "arrival":
            # Trees silhouette
            for i in range(7):
                tx = int(W * (0.05 + i * 0.13 + rng(-0.02, 0.02)))
                th = rng(140, 260)
                tw = rng(50, 90)
                draw.rectangle([tx - 6, gs - th, tx + 6, gs], fill=(10, 30, 10, 240))
                draw.ellipse([tx - tw, gs - th - tw//2, tx + tw, gs - th + tw//2],
                             fill=(15, 55, 15, 240))
            # Children running silhouettes
            for i in range(5):
                cx = int(W * (0.35 + i * 0.07))
                cy = gs - rng(30, 60)
                r = rng(8, 18)
                draw.ellipse([cx-r, cy-r*2, cx+r, cy], fill=(*ac, 200))

        elif stype == "gates":
            # Gate arch
            gw = 300; cx = W // 2; gy = gs - 50
            draw.rectangle([cx - gw//2 - 30, gy - 350, cx - gw//2, gy], fill=(60, 40, 20, 240))
            draw.rectangle([cx + gw//2, gy - 350, cx + gw//2 + 30, gy], fill=(60, 40, 20, 240))
            draw.arc([cx - gw//2 - 30, gy - 500, cx + gw//2 + 30, gy - 200],
                     start=180, end=0, fill=(80, 55, 25, 240), width=35)
            # Balloons
            balloon_colors = [(220,50,50), (50,180,50), (50,100,230), (230,180,30)]
            for i, bc in enumerate(balloon_colors):
                bx = cx - gw//2 - 80 + i * (gw + 160) // 3
                by = gy - 300 - rng(0, 80)
                draw.ellipse([bx-25, by-40, bx+25, by+40], fill=(*bc, 230))
                draw.line([bx, by+40, bx+rng(-20,20), by+130], fill=(*bc, 150), width=2)

        elif stype == "playground":
            # Slide shape
            sx = int(W * 0.3); sy_top = gs - 320; sy_bot = gs - 10
            draw.polygon([(sx, sy_top), (sx+60, sy_top), (sx+220, sy_bot), (sx+150, sy_bot)],
                         fill=(220, 60, 40, 230))
            draw.rectangle([sx-20, sy_top-100, sx+20, sy_top], fill=(180, 180, 60, 230))
            # Swings
            for i in range(3):
                swx = int(W * (0.5 + i * 0.13)); swy = gs - 380
                draw.line([swx-15, swy, swx-15, swy+320], fill=(100,100,100,200), width=4)
                draw.line([swx+15, swy, swx+15, swy+320], fill=(100,100,100,200), width=4)
                draw.rectangle([swx-25, swy+300, swx+25, swy+330], fill=(*ac, 220))
                # Child on swing
                draw.ellipse([swx-20, swy+260, swx+20, swy+300], fill=(255,200,160,200))

        elif stype == "field":
            # Aerial field markings
            draw.rectangle([W//6, H//4, W*5//6, H*3//4], fill=(50, 160, 55, 100))
            # Cricket pitch
            cx, cy = W//2, H//2
            draw.rectangle([cx-15, cy-120, cx+15, cy+120], fill=(200,180,120,200))
            # Player dots
            positions = [(0.3,0.4),(0.7,0.4),(0.2,0.6),(0.8,0.6),(0.4,0.3),(0.6,0.3),
                        (0.3,0.7),(0.7,0.7),(0.5,0.5),(0.45,0.55)]
            team_colors = [(220,60,60),(60,100,220)]
            for j,(px,py) in enumerate(positions):
                color = team_colors[j % 2]
                draw.ellipse([int(W*px)-12, int(H*py)-12, int(W*px)+12, int(H*py)+12],
                             fill=(*color, 230))

        elif stype == "tree":
            # Large banyan tree
            tx = W // 2 + rng(-100, 100); ty = gs
            draw.rectangle([tx-25, ty-400, tx+25, ty], fill=(60, 35, 20, 250))
            for i in range(3):
                ox = rng(-80, 80)
                draw.rectangle([tx+ox-8, ty-300+i*60, tx+ox+8, ty], fill=(60, 35, 20, 180))
            draw.ellipse([tx-380, ty-520, tx+380, ty-100], fill=(20, 80, 20, 220))
            draw.ellipse([tx-300, ty-480, tx+350, ty-150], fill=(25, 95, 25, 200))
            # Children sitting
            for i in range(6):
                cx2 = tx - 200 + i * 80; cy2 = ty - rng(10, 30)
                draw.ellipse([cx2-14, cy2-28, cx2+14, cy2], fill=(*ac, 200))

        elif stype == "crafts":
            # Art supplies and paint splashes
            paint_colors = [(220,60,60),(60,180,60),(60,80,220),(220,180,30),(180,60,220)]
            for i, pc in enumerate(paint_colors):
                bx = rng(150, W-150); by = rng(H//4, H*3//4)
                for _ in range(8):
                    dx = rng(-60, 60); dy = rng(-60, 60)
                    r  = rng(10, 40)
                    draw.ellipse([bx+dx-r, by+dy-r, bx+dx+r, by+dy+r], fill=(*pc, 140))
            # Brush shapes
            for i in range(4):
                bx = W//5 + i * W//5; by = gs - 200
                draw.rectangle([bx-5, by-180, bx+5, by], fill=(80, 50, 20, 220))
                draw.ellipse([bx-18, by-200, bx+18, by-160], fill=(*ac, 220))

        elif stype == "finale":
            # Stage podium
            sw = int(W * 0.6); sh = 140
            sx = (W - sw) // 2; sy = gs - sh
            draw.rectangle([sx, sy, sx+sw, gs], fill=(50, 30, 80, 230))
            draw.rectangle([sx-30, sy-20, sx+sw+30, sy+10], fill=(70, 50, 100, 230))
            # Crowd silhouette
            for i in range(40):
                cx2 = int(W * (0.05 + i * 0.023)); cy2 = sy + rng(-20, 10)
                draw.ellipse([cx2-14, cy2-28, cx2+14, cy2+5], fill=(15,10,30,200))
            # Fireworks
            fw_colors = [(255,220,50),(255,100,100),(100,200,255),(200,255,100)]
            for i, fc in enumerate(fw_colors):
                fx = rng(W//8, W*7//8); fy = rng(H//8, H*5//12)
                for angle in range(0, 360, 20):
                    rad = math.radians(angle)
                    length = rng(50, 120)
                    ex = int(fx + length * math.cos(rad))
                    ey = int(fy + length * math.sin(rad))
                    draw.line([fx, fy, ex, ey], fill=(*fc, 200), width=2)
                draw.ellipse([fx-8, fy-8, fx+8, fy+8], fill=(*fc, 255))

    # ── Bokeh particles ───────────────────────────────────────────────────────

    def _draw_bokeh(self, draw, W, H, accent, rng, n=20) -> None:
        for _ in range(n):
            bx = rng(0, W); by = rng(0, H)
            br = rng(4, 22)
            alpha = rng(15, 60)
            draw.ellipse([bx-br, by-br, bx+br, by+br], fill=(*accent, alpha))

    # ── Vignette ─────────────────────────────────────────────────────────────

    def _add_vignette(self, img, W: int, H: int):
        import numpy as np
        from PIL import Image
        arr = np.array(img).astype(float)
        cx, cy = W / 2, H / 2
        max_dist = math.sqrt(cx**2 + cy**2)
        ys, xs = np.mgrid[0:H, 0:W]
        dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        vignette = 1.0 - 0.75 * (dist / max_dist) ** 1.8
        vignette = np.clip(vignette, 0, 1)
        arr *= vignette[:, :, np.newaxis]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    # ── Text overlay ─────────────────────────────────────────────────────────

    def _draw_text(self, draw, prompt: str, scene_id: int, W, H, bar_h) -> None:
        from PIL import ImageFont

        TAMIL_FONT = "/usr/share/fonts/truetype/noto/NotoSansTamilUI-Regular.ttf"
        LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        LATIN_SM   = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        try:
            f_large = ImageFont.truetype(LATIN_FONT, 44)
            f_small = ImageFont.truetype(LATIN_SM, 26)
        except OSError:
            f_large = f_small = ImageFont.load_default()

        # Scene number in top-left letterbox bar
        draw.text((30, bar_h // 2), f"Scene {scene_id:02d}",
                  font=f_large, fill=(255, 255, 255, 230), anchor="lm")

        # Short description in bottom bar (first 72 chars)
        snippet = (prompt[:72] + "…") if len(prompt) > 72 else prompt
        draw.text((W // 2, H - bar_h // 2), snippet,
                  font=f_small, fill=(220, 220, 220, 210), anchor="mm")

    # ── Deterministic RNG helper ──────────────────────────────────────────────

    def _rng(self, seed_str: str):
        """Return a deterministic int-range sampler seeded by the prompt."""
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) & 0xFFFFFFFF
        state = [seed]

        def rand_int(lo: int, hi: int) -> int:
            state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF
            return lo + (state[0] >> 16) % max(1, hi - lo + 1)

        return rand_int

    # ── HuggingFace Inference API ─────────────────────────────────────────────

    def _generate_huggingface(self, prompt: str) -> bytes:
        import requests
        url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.hf_api_key:
            headers["Authorization"] = f"Bearer {self.hf_api_key}"
        resp = requests.post(url, headers=headers,
                             json={"inputs": prompt,
                                   "parameters": {"width": 1024, "height": 576,
                                                  "num_inference_steps": 4}},
                             timeout=120)
        resp.raise_for_status()
        return resp.content

    # ── DALL-E 3 ─────────────────────────────────────────────────────────────

    def _generate_dalle(self, prompt: str) -> tuple[bytes, str]:
        resp = self.openai_client.images.generate(
            model="dall-e-3", prompt=prompt,
            size="1792x1024", quality="hd", n=1, response_format="b64_json",
        )
        return base64.b64decode(resp.data[0].b64_json), resp.data[0].revised_prompt or prompt

    # ── Stability AI ─────────────────────────────────────────────────────────

    def _generate_stability(self, prompt: str) -> bytes:
        import requests
        resp = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers={"Authorization": f"Bearer {self.stability_api_key}", "Accept": "image/*"},
            files={"none": ""},
            data={"prompt": prompt, "output_format": "png", "aspect_ratio": "16:9"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content
