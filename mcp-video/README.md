# mcp-video

A Model Context Protocol (MCP) server that lets Claude Code analyze video files — without sending them to an expensive API.

**Pipeline:** `ffmpeg` extracts audio → `Whisper` transcribes locally (free) → `Claude Haiku` analyzes frames (minimal cost)

## The problem

Claude Code can read images and PDFs natively, but not video. The naive fix — base64-encode the MP4 and send it to the Anthropic API — works but costs a lot for long videos.

This MCP server solves it differently: do the heavy lifting locally for free, then only send lightweight text and a few frames to the cheapest Claude model.

## Cost comparison

| Approach | 10-min video |
|---|---|
| Full video → Claude API | ~$0.50–2.00 |
| **mcp-video (this repo)** | **~$0.001** |
| Transcription only | **$0.00** |

## Tools

### `transcribe_video`
Transcribes the audio track using local Whisper. **Zero API cost.**

```json
{
  "video_path": "/path/to/video.mp4",
  "model": "base"
}
```

Whisper model options: `tiny` (fastest) · `base` (default) · `small` · `medium` · `large` (most accurate)

### `analyze_video`
Extracts key frames + transcript, then sends both to Claude Haiku for analysis.

```json
{
  "video_path": "/path/to/video.mp4",
  "prompt": "What are the key learnings from this video?",
  "frames": 4
}
```

## Requirements

- [ffmpeg](https://ffmpeg.org/) — `brew install ffmpeg`
- [openai-whisper](https://github.com/openai/whisper) — `pip install openai-whisper`
- Node.js 18+
- Anthropic API key (only needed for `analyze_video`, not `transcribe_video`)

## Installation

```bash
git clone https://github.com/thefranceway/mcp-video
cd mcp-video
npm install
```

Create a wrapper script at `~/bin/mcp-video`:
```bash
#!/bin/zsh
export ANTHROPIC_API_KEY="your-key-here"
exec /path/to/node /path/to/mcp-video/index.js "$@"
```
```bash
chmod +x ~/bin/mcp-video
```

Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "video": {
      "command": "/Users/you/bin/mcp-video",
      "args": []
    }
  }
}
```

Restart Claude Code. The `transcribe_video` and `analyze_video` tools are now available in every session.

## How it works

```
video.mp4
    │
    ├── ffmpeg ──► audio.wav ──► Whisper (local) ──► transcript (free)
    │
    └── ffmpeg ──► frame_0.jpg
                   frame_1.jpg  ──► Claude Haiku ──► analysis
                   frame_2.jpg       + transcript
                   frame_3.jpg
```

Temp files are created in a system tmpdir and cleaned up after every call.

## Usage in Claude Code

Once installed, just reference a video by path:

> "Transcribe /Users/me/Downloads/lecture.mp4"

> "Analyze the frames in this video and give me key takeaways: /Users/me/Downloads/demo.mp4"

Claude will call the appropriate tool automatically.

## License

MIT
