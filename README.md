# AI Video Production Pipeline

A multi-agent Python pipeline that takes a topic and produces a fully assembled, watermarked video ready for review — powered by Claude, ElevenLabs, DALL-E, Runway, and FFmpeg.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         Supervisor Agent         │
                        │  (Claude — controls all agents)  │
                        └────────────┬────────────────────┘
                                     │  validates each step
                    ┌────────────────▼────────────────────┐
                    │           Pipeline Steps             │
                    │                                      │
                    │  1. Script Writer  ──► VideoScript   │
                    │  2. Storyboard     ──► Storyboard    │
                    │  3. Voiceover      ──► AudioAssets   │
                    │  4. Image Gen      ──► ImageAssets   │
                    │  5. Video Assembler──► VideoClips    │
                    │  6. Review         ──► Preview MP4   │
                    └──────────────────────────────────────┘
```

### Agents

| Agent | File | Responsibility | AI Service |
|---|---|---|---|
| **Supervisor** | `agents/supervisor.py` | Controls all agents, validates outputs, retries failures, writes audit log | Claude (Anthropic) |
| **Script Writer** | `agents/script_writer.py` | Generates a structured video script from a topic | Claude (Anthropic) |
| **Storyboard** | `agents/storyboard.py` | Breaks the script into visual scenes with image-gen prompts | Claude (Anthropic) |
| **Voiceover** | `agents/voiceover.py` | Converts narration text to MP3 per scene | ElevenLabs TTS |
| **Image Generator** | `agents/image_generator.py` | Generates one image per scene | DALL-E 3 or Stability AI |
| **Video Assembler** | `agents/video_assembler.py` | Animates images into clips (Runway) and stitches final MP4 (MoviePy) | Runway + FFmpeg |
| **Review** | `agents/review.py` | Adds watermark overlay to produce a preview video for human QA | MoviePy |

---

## How the Supervisor Works

The `SupervisorAgent` sits above all other agents and acts as the pipeline controller:

1. **Calls each agent in order** and captures its output
2. **Asks Claude** after each step: _"Did this succeed? Proceed, retry, skip, or abort?"_
3. **Retries** failed steps automatically (up to 2 times) before aborting
4. **Skips** non-critical steps if they fail and cannot be retried
5. **Aborts** the pipeline and saves the audit log on unrecoverable errors
6. **Writes a structured audit log** (`output/final/<run_id>_supervisor_log.json`) with per-step status, duration, and supervisor decisions

```
Supervisor decision flow per step:

  Run agent
      │
      ▼
  Success? ──Yes──► Ask Claude ──► proceed / retry / skip / abort
      │
     No
      │
      ▼
  Ask Claude ──► retry (up to 2x) / skip / abort
```

---

## Data Flow

Each agent passes a typed Pydantic model to the next:

```
ScriptRequest
    └─► ScriptWriterAgent   ──► VideoScript
            └─► StoryboardAgent     ──► Storyboard
                    ├─► VoiceoverAgent      ──► list[AudioAsset]
                    └─► ImageGeneratorAgent ──► list[ImageAsset]
                              └─► [build ProductionPackage]
                                      └─► VideoAssemblerAgent ──► ProductionPackage
                                                └─► ReviewAgent ──► ProductionPackage
                                                                     (preview_video_path set)
```

### Key models (`models/`)

| Model | Purpose |
|---|---|
| `ScriptRequest` | User input — topic, duration, tone, language |
| `VideoScript` | Script title + list of `ScriptSection` |
| `Storyboard` | List of `Scene` objects with visual descriptions |
| `AudioAsset` | Path to MP3 + duration per scene |
| `ImageAsset` | Path to PNG + prompt used per scene |
| `VideoClip` | Path to MP4 clip per scene |
| `ProductionPackage` | Complete production state carrying all assets through the pipeline |

---

## Project Structure

```
.
├── agents/
│   ├── supervisor.py        # Supervisor — controls all agents
│   ├── script_writer.py     # Script Writer Agent
│   ├── storyboard.py        # Storyboard Agent
│   ├── voiceover.py         # Voiceover Agent
│   ├── image_generator.py   # Image Generator Agent
│   ├── video_assembler.py   # Video Assembler Agent
│   └── review.py            # Review Agent
├── models/
│   ├── script.py            # ScriptRequest, VideoScript, ScriptSection
│   ├── storyboard.py        # Storyboard, Scene, ShotType
│   └── production.py        # ProductionPackage, AudioAsset, ImageAsset, VideoClip
├── pipeline/
│   └── orchestrator.py      # Thin wrapper delegating to SupervisorAgent
├── config/
│   └── settings.py          # Pydantic Settings — reads from .env
├── tests/
│   ├── test_script_writer.py
│   ├── test_storyboard.py
│   ├── test_voiceover.py
│   ├── test_image_generator.py
│   └── test_review.py
├── output/                  # Generated files (git-ignored)
│   ├── scripts/
│   ├── storyboards/
│   ├── audio/
│   ├── images/
│   ├── clips/
│   └── final/               # Final video, preview, audit log, review report
├── main.py                  # CLI entry point
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- FFmpeg installed on your system:
  ```bash
  # Ubuntu / Debian
  sudo apt install ffmpeg

  # macOS
  brew install ffmpeg

  # Windows — download from https://ffmpeg.org/download.html
  ```

### 2. Install Python dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key — [console.anthropic.com](https://console.anthropic.com) |
| `ELEVENLABS_API_KEY` | Yes | ElevenLabs TTS key — [elevenlabs.io](https://elevenlabs.io) |
| `OPENAI_API_KEY` | Yes (DALL-E) | OpenAI key for DALL-E 3 |
| `STABILITY_API_KEY` | Optional | Stability AI key (if `IMAGE_BACKEND=stability`) |
| `RUNWAY_API_KEY` | Optional | Runway key for video animation (if `VIDEO_BACKEND=runway`) |

---

## Usage

### Full pipeline (Runway animation)

```bash
python main.py --topic "The history of space exploration" --duration 90 --tone cinematic
```

### Static images only (faster, no Runway cost)

```bash
python main.py --topic "How black holes form" --duration 60 --no-animate
```

### All options

```
--topic       REQUIRED  Topic or prompt for the video
--duration    OPTIONAL  Target length in seconds (default: 60)
--tone        OPTIONAL  educational | cinematic | promotional (default: educational)
--language    OPTIONAL  ISO language code (default: en)
--no-animate  FLAG      Skip Runway — use static images stitched with MoviePy
```

### Run a single agent in isolation

```bash
python -m agents.script_writer --topic "Quantum computing basics" --duration 45
python -m agents.storyboard --script-file output/scripts/20240101_120000_script.json
```

---

## Output Files

After a successful run, all outputs are in `output/`:

| File | Description |
|---|---|
| `output/scripts/<ts>_script.json` | Generated video script |
| `output/storyboards/<ts>_storyboard.json` | Scene breakdown |
| `output/audio/scene_001.mp3` … | Per-scene voiceover audio |
| `output/images/scene_001.png` … | Per-scene generated images |
| `output/clips/scene_001_static.mp4` … | Per-scene video clips |
| `output/final/<run_id>_final.mp4` | Final assembled video |
| `output/final/<run_id>_preview.mp4` | **Watermarked preview for review** |
| `output/final/<run_id>_review_report.json` | Asset manifest + durations |
| `output/final/<run_id>_supervisor_log.json` | Full supervisor audit log |

> **Note:** Always review `_preview.mp4` before publishing. The pipeline never auto-publishes.

---

## Supervisor Audit Log

Every run produces a `_supervisor_log.json` structured as:

```json
{
  "pipeline_run_id": "a1b2c3d4...",
  "topic": "The history of space exploration",
  "started_at": "2026-04-06T10:00:00",
  "finished_at": "2026-04-06T10:12:34",
  "final_status": "success",
  "steps": [
    {
      "name": "Script Writer",
      "status": "success",
      "attempt": 1,
      "duration_seconds": 4.2,
      "supervisor_decision": "proceed: Script is well-structured with clear sections.",
      "error": null
    },
    ...
  ]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

All external API calls are mocked — no real API keys needed for tests.

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `IMAGE_BACKEND` | `dalle` | `dalle` or `stability` |
| `VIDEO_BACKEND` | `runway` | `runway` or `none` (static images) |
| `CLAUDE_MODEL` | `claude-opus-4-5` | Anthropic model ID |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice (default: Rachel) |
| `OUTPUT_DIR` | `output` | Root directory for all generated files |
| `WATERMARK_TEXT` | `PREVIEW — NOT FOR DISTRIBUTION` | Text overlaid on preview video |
