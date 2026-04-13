#!/usr/bin/env node

const { Server } = require("@modelcontextprotocol/sdk/server/index.js");
const { StdioServerTransport } = require("@modelcontextprotocol/sdk/server/stdio.js");
const { CallToolRequestSchema, ListToolsRequestSchema } = require("@modelcontextprotocol/sdk/types.js");
const Anthropic = require("@anthropic-ai/sdk");
const fs = require("fs");
const path = require("path");
const { execSync, spawnSync } = require("child_process");
const os = require("os");

const anthropic = new Anthropic.default({ apiKey: process.env.ANTHROPIC_API_KEY });

const server = new Server(
  { name: "mcp-video", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "analyze_video",
      description: "Analyze a video file using local Whisper transcription + ffmpeg frame extraction, then summarize with Claude Haiku. Minimal API cost.",
      inputSchema: {
        type: "object",
        properties: {
          video_path: { type: "string", description: "Absolute path to the video file" },
          prompt: { type: "string", description: "What to extract. Defaults to full summary + key learnings." },
          frames: { type: "number", description: "Number of key frames to extract (default 4)" }
        },
        required: ["video_path"]
      }
    },
    {
      name: "transcribe_video",
      description: "Transcribe a video's audio locally using Whisper. Free, no API cost.",
      inputSchema: {
        type: "object",
        properties: {
          video_path: { type: "string", description: "Absolute path to the video file" },
          model: { type: "string", description: "Whisper model: tiny, base, small, medium, large (default: base)" }
        },
        required: ["video_path"]
      }
    }
  ]
}));

function extractAudio(videoPath, tmpDir) {
  const audioPath = path.join(tmpDir, "audio.wav");
  execSync(`/opt/homebrew/bin/ffmpeg -i "${videoPath}" -ar 16000 -ac 1 -vn "${audioPath}" -y -loglevel error`);
  return audioPath;
}

function transcribeWithWhisper(audioPath, model = "base") {
  const result = spawnSync(
    "/opt/homebrew/bin/whisper",
    [audioPath, "--model", model, "--output_format", "txt", "--output_dir", path.dirname(audioPath), "--fp16", "False"],
    { encoding: "utf8", timeout: 120000 }
  );
  if (result.status !== 0) throw new Error(`Whisper failed: ${result.stderr}`);
  const txtPath = audioPath.replace(".wav", ".txt");
  return fs.existsSync(txtPath) ? fs.readFileSync(txtPath, "utf8").trim() : "";
}

function extractFrames(videoPath, tmpDir, count = 4) {
  // Get video duration
  const probe = spawnSync(
    "/opt/homebrew/bin/ffprobe",
    ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", videoPath],
    { encoding: "utf8" }
  );
  const duration = parseFloat(probe.stdout.trim()) || 10;

  const frames = [];
  for (let i = 0; i < count; i++) {
    const timestamp = (duration / (count + 1)) * (i + 1);
    const framePath = path.join(tmpDir, `frame_${i}.jpg`);
    execSync(`/opt/homebrew/bin/ffmpeg -i "${videoPath}" -ss ${timestamp.toFixed(2)} -vframes 1 -vf scale=640:-1 "${framePath}" -y -loglevel error`);
    if (fs.existsSync(framePath)) frames.push(framePath);
  }
  return frames;
}

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "transcribe_video") {
    const { video_path, model = "base" } = args;
    if (!fs.existsSync(video_path)) return { content: [{ type: "text", text: `File not found: ${video_path}` }] };

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "mcp-video-"));
    try {
      const audioPath = extractAudio(video_path, tmpDir);
      const transcript = transcribeWithWhisper(audioPath, model);
      return { content: [{ type: "text", text: transcript || "(no speech detected)" }] };
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }

  if (name === "analyze_video") {
    const { video_path, prompt, frames: frameCount = 4 } = args;
    if (!fs.existsSync(video_path)) return { content: [{ type: "text", text: `File not found: ${video_path}` }] };

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "mcp-video-"));
    try {
      // Step 1: Local transcript (free)
      let transcript = "(no audio)";
      try {
        const audioPath = extractAudio(video_path, tmpDir);
        transcript = transcribeWithWhisper(audioPath) || "(no speech detected)";
      } catch (e) {
        transcript = `(transcription failed: ${e.message})`;
      }

      // Step 2: Extract key frames (free)
      const framePaths = extractFrames(video_path, tmpDir, frameCount);

      // Step 3: Send frames + transcript to Haiku (minimal cost)
      const userPrompt = prompt || "Analyze this video. Based on the frames and transcript, provide: 1) What is happening visually, 2) Key spoken content, 3) Key learnings or takeaways.";

      const content = [
        ...framePaths.map(fp => ({
          type: "image",
          source: {
            type: "base64",
            media_type: "image/jpeg",
            data: fs.readFileSync(fp).toString("base64")
          }
        })),
        {
          type: "text",
          text: `Transcript:\n${transcript}\n\n${userPrompt}`
        }
      ];

      const response = await anthropic.messages.create({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 2048,
        messages: [{ role: "user", content }]
      });

      const analysis = response.content[0].text;
      return {
        content: [{
          type: "text",
          text: `**Transcript (local Whisper):**\n${transcript}\n\n---\n\n**Analysis:**\n${analysis}`
        }]
      };

    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }

  throw new Error(`Unknown tool: ${name}`);
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("mcp-video server running (whisper+ffmpeg pipeline)");
}

main().catch(console.error);
