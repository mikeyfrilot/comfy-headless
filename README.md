# Comfy Headless

**Making ComfyUI's power accessible without the complexity.**

A clean, headless interface for ComfyUI with AI-powered prompt enhancement, template-based workflows, and multi-model video generation.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-2.5.1-green.svg)

## Philosophy

ComfyUI is incredibly powerful, but its node-based interface can be overwhelming. Comfy Headless bridges this gap:

- **For Users**: Simple presets and AI-powered prompt enhancement
- **For Developers**: Clean API with template-based workflow compilation
- **For Everyone**: Best settings for your intent, automatically

## Installation

### Modular Installation (v2.5.0+)

Install only what you need:

```bash
# Core only (minimal - ~2MB)
pip install comfy-headless

# With AI prompt enhancement (Ollama)
pip install comfy-headless[ai]

# With WebSocket real-time progress
pip install comfy-headless[websocket]

# Recommended for most users
pip install comfy-headless[standard]

# Everything (UI, health monitoring, observability)
pip install comfy-headless[full]
```

### Available Extras

| Extra | Dependencies | Features |
|-------|--------------|----------|
| `ai` | httpx | Ollama prompt intelligence |
| `websocket` | websockets | Real-time progress updates |
| `health` | psutil | System health monitoring |
| `ui` | gradio | Web interface |
| `validation` | pydantic | Config validation |
| `observability` | opentelemetry | Distributed tracing |
| `standard` | ai + websocket | Recommended bundle |
| `full` | All of the above | Everything |

## Requirements

- Python 3.10+
- ComfyUI running locally (default: `http://localhost:8188`)
- Optional: Ollama for AI prompt enhancement

## Quick Start

### Use as a Library

```python
from comfy_headless import ComfyClient

# Simple image generation
client = ComfyClient()
result = client.generate_image("a beautiful sunset over mountains")
print(f"Generated: {result['images']}")
```

### With AI Enhancement (requires `[ai]` extra)

```python
from comfy_headless import ComfyClient, analyze_prompt, enhance_prompt

# Analyze a prompt
analysis = analyze_prompt("a cyberpunk city at night with neon lights")
print(f"Intent: {analysis.intent}")        # "scene"
print(f"Styles: {analysis.styles}")        # ["scifi", "cinematic"]
print(f"Preset: {analysis.suggested_preset}")  # "cinematic"

# Enhance a prompt
enhanced = enhance_prompt("a cat", style="detailed")
print(enhanced.enhanced)   # "a cat, masterpiece, best quality, highly detailed..."
print(enhanced.negative)   # Style-aware negative prompt
```

### Video Generation

```python
from comfy_headless import ComfyClient, VIDEO_PRESETS, list_video_presets

# See available presets
print(list_video_presets())

# Generate video with preset
client = ComfyClient()
result = client.generate_video(
    prompt="a cat walking through a garden",
    preset="ltx_quality"  # LTX-Video 2, 1280x720, 49 frames
)
```

### Launch the Web UI (requires `[ui]` extra)

```python
from comfy_headless import launch
launch()  # Opens http://localhost:7870
```

Or via command line:
```bash
python -m comfy_headless.ui
```

**UI Features (v2.5.1):**
- **Image Generation** - txt2img with presets, AI prompt enhancement
- **Video Generation** - AnimateDiff, LTX, Hunyuan, Wan support
- **Queue & History** - Real-time queue management, job history
- **Workflows** - Browse, import, and create workflow templates
- **Models Browser** - View checkpoints, LoRAs, motion models
- **Settings** - Connection management, timeouts, system info

**Theme:** Ocean Mist - soft teal accents on warm neutral backgrounds (Gradio 6.0)

## Video Models (v2.5.0)

### Supported Models

| Model | VRAM | Quality | Speed | Best For |
|-------|------|---------|-------|----------|
| **LTX-Video 2** | 12GB+ | Excellent | Fast | General use, RTX 3080+ |
| **Hunyuan 1.5** | 14GB+ | Best | Slow | High quality, RTX 4080+ |
| **Wan 2.1/2.2** | 6-16GB | Great | Medium | Budget GPUs, efficiency |
| **Mochi** | 12GB+ | Excellent | Slow | Text adherence |
| AnimateDiff | 6GB+ | Good | Fast | Quick previews |
| SVD | 8GB+ | Good | Medium | Image-to-video |
| CogVideoX | 10GB+ | Good | Slow | Legacy support |

### Video Presets

```python
from comfy_headless import VIDEO_PRESETS, get_recommended_preset

# Get preset recommendation based on your VRAM
preset = get_recommended_preset(vram_gb=16)  # Returns "hunyuan15_720p"

# Available presets
presets = {
    # LTX-Video 2 (Fast, great quality)
    "ltx_quick": "768x512, 25 frames, 20 steps",
    "ltx_standard": "1280x720, 49 frames, 25 steps",
    "ltx_quality": "1280x720, 97 frames, 30 steps",

    # Hunyuan 1.5 (Best quality)
    "hunyuan15_720p": "1280x720, 121 frames",
    "hunyuan15_quality": "1280x720, 121 frames, cfg=6",
    "hunyuan15_fast": "848x480, 61 frames, 6 steps",
    "hunyuan15_1080p": "1920x1080 with super-resolution",

    # Wan (Efficient)
    "wan_1.3b": "720x480, 49 frames (6GB VRAM)",
    "wan_14b": "1280x720, 81 frames (12GB VRAM)",
    "wan_fast": "720x480, 25 frames, 4 steps",

    # Mochi (Text adherence)
    "mochi": "848x480, 84 frames",
    "mochi_short": "848x480, 37 frames",

    # Legacy AnimateDiff
    "quick": "512x512, 16 frames, 4-step lightning",
    "standard": "512x512, 16 frames",
    "quality": "768x768, 24 frames with RIFE",
}
```

## Feature Flags

Check what features are available:

```python
from comfy_headless import FEATURES, list_available_features, list_missing_features

print(FEATURES)
# {'ai': True, 'websocket': True, 'health': False, ...}

print(list_missing_features())
# {'health': 'pip install comfy-headless[health]', ...}
```

## WebSocket Progress (requires `[websocket]` extra)

```python
import asyncio
from comfy_headless import ComfyWSClient

async def generate_with_progress():
    async with ComfyWSClient() as ws:
        # Queue a prompt
        prompt_id = await ws.queue_prompt(workflow)

        # Wait with progress callbacks
        result = await ws.wait_for_completion(
            prompt_id,
            on_progress=lambda p: print(f"Progress: {p.progress}%")
        )
        return result

asyncio.run(generate_with_progress())
```

## API Reference

### Core Classes

```python
from comfy_headless import (
    # Client
    ComfyClient,           # Main HTTP client
    ComfyWSClient,         # WebSocket client (requires [websocket])

    # Video
    VideoSettings,         # Video generation settings
    VideoModel,            # Model enum (LTXV, HUNYUAN_15, WAN, etc.)
    VIDEO_PRESETS,         # Preset configurations
    get_recommended_preset, # VRAM-based recommendation

    # Workflows
    compile_workflow,      # Compile workflow from preset
    WorkflowCompiler,      # Low-level compiler

    # Intelligence (requires [ai])
    analyze_prompt,        # Analyze prompt intent/style
    enhance_prompt,        # AI-powered enhancement
    PromptAnalysis,        # Analysis result type
)
```

### Configuration

```python
from comfy_headless import settings, get_settings, reload_settings

# Access settings
print(settings.comfy_url)  # http://localhost:8188

# Override via environment
# COMFY_HEADLESS_COMFY_URL=http://remote:8188
```

### Error Handling

```python
from comfy_headless import (
    ComfyHeadlessError,      # Base exception
    ComfyUIConnectionError,  # Can't reach ComfyUI
    ComfyUIOfflineError,     # ComfyUI not responding
    GenerationTimeoutError,  # Generation took too long
    GenerationFailedError,   # Generation failed
    ValidationError,         # Invalid parameters
)

try:
    result = client.generate_image("test")
except ComfyUIOfflineError:
    print("Start ComfyUI first!")
except GenerationTimeoutError:
    print("Generation timed out")
```

## Architecture

```
comfy_headless/
├── __init__.py          # Package exports, lazy loading
├── feature_flags.py     # Optional dependency detection
├── client.py            # ComfyUI HTTP client
├── websocket_client.py  # WebSocket client
├── intelligence.py      # AI prompt analysis (requires [ai])
├── workflows.py         # Template compiler & presets
├── video.py             # Video models & presets
├── ui.py                # Gradio 6.0 interface (requires [ui])
├── theme.py             # Ocean Mist theme (soft teal, warm neutrals)
├── config.py            # Settings management
├── exceptions.py        # Error types with verbosity
├── retry.py             # Circuit breaker, rate limiting
├── health.py            # Health checks (requires [health])
├── secrets_manager.py   # Secrets management utilities
├── help_system.py       # Contextual help system
└── tests/               # Test suite
```

## ComfyUI Node Requirements

### For Video Generation

Install these custom nodes:

**Core:**
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) - Video encoding

**Model-Specific:**
- LTX-Video 2: Built-in ComfyUI support (recent versions)
- Hunyuan 1.5: [ComfyUI-HunyuanVideo](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)
- Wan: [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- AnimateDiff: [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! Please open an issue or pull request.

Areas of interest:
- Additional video model support
- Workflow templates
- Documentation
- Bug fixes
