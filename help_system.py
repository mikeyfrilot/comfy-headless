"""
Comfy Headless - Help System v2.5.1
===================================

Comprehensive help system with multi-level explanations,
error recovery guidance, and interactive discovery.

Features:
- Three expertise levels: ELI5, Casual, Developer
- Contextual error help with recovery actions
- Searchable topic database
- Command reference with examples
- API documentation

Usage:
    from comfy_headless.help_system import get_help, HelpLevel

    # Get help for a topic
    print(get_help("generation"))

    # Get help at specific level
    print(get_help("prompts", level=HelpLevel.DEVELOPER))

    # Search for topics
    topics = search_help("timeout")

    # Get help for an error
    print(get_help_for_error("COMFYUI_OFFLINE"))
"""

import os
from dataclasses import dataclass, field
from enum import Enum

from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "HelpLevel",
    "HelpTopic",
    "HelpRegistry",
    "get_help",
    "get_help_for_error",
    "list_topics",
    "search_help",
    "set_help_level",
    "get_help_level",
    "format_quick_help",
    "format_help_list",
    "get_command_help",
    "get_api_reference",
]


# =============================================================================
# HELP LEVELS
# =============================================================================


class HelpLevel(Enum):
    """
    User expertise levels for help content adaptation.

    ELI5: Simple explanations, no jargon, friendly tone
    CASUAL: Balanced explanations for regular users
    DEVELOPER: Full technical details, code examples, internals
    """

    ELI5 = "eli5"
    CASUAL = "casual"
    DEVELOPER = "developer"


def _get_default_level() -> HelpLevel:
    """Get default help level from environment."""
    level = os.environ.get("COMFY_HEADLESS_HELP_LEVEL", "casual").lower()
    try:
        return HelpLevel(level)
    except ValueError:
        return HelpLevel.CASUAL


_current_level: HelpLevel = _get_default_level()


def set_help_level(level: HelpLevel) -> None:
    """Set the global help verbosity level."""
    global _current_level
    _current_level = level
    logger.debug(f"Help level set to {level.value}")


def get_help_level() -> HelpLevel:
    """Get the current help verbosity level."""
    return _current_level


# =============================================================================
# HELP TOPIC DATA STRUCTURE
# =============================================================================


@dataclass
class HelpTopic:
    """
    A help topic with content adapted for each expertise level.

    Attributes:
        id: Unique identifier (e.g., "generation", "error:TIMEOUT")
        title: Human-readable title
        category: Grouping category (core, feature, error, api)
        eli5: Simple explanation for beginners
        casual: Balanced explanation for regular users
        developer: Technical explanation with code examples
        examples: Code examples (shown for developer level)
        related: Related topic IDs
        keywords: Search keywords
    """

    id: str
    title: str
    category: str
    eli5: str
    casual: str
    developer: str
    examples: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    keywords: set[str] = field(default_factory=set)

    def get_content(self, level: HelpLevel | None = None) -> str:
        """Get content for the specified or current level."""
        level = level or _current_level
        if level == HelpLevel.ELI5:
            return self.eli5
        elif level == HelpLevel.CASUAL:
            return self.casual
        return self.developer

    def format(self, level: HelpLevel | None = None, show_examples: bool = True) -> str:
        """Format topic as displayable help text."""
        level = level or _current_level
        lines = [f"# {self.title}", "", self.get_content(level)]

        if show_examples and level == HelpLevel.DEVELOPER and self.examples:
            lines.extend(["", "## Examples", ""])
            for example in self.examples:
                lines.append(f"```python\n{example}\n```")

        if self.related:
            related_str = ", ".join(self.related)
            lines.extend(["", f"Related: {related_str}"])

        return "\n".join(lines)


# =============================================================================
# HELP REGISTRY
# =============================================================================


class HelpRegistry:
    """
    Central registry for all help topics.

    Provides registration, lookup, and search functionality.
    """

    def __init__(self):
        self._topics: dict[str, HelpTopic] = {}
        self._by_category: dict[str, list[str]] = {}

    def register(self, topic: HelpTopic) -> None:
        """Register a help topic."""
        self._topics[topic.id] = topic

        # Index by category
        if topic.category not in self._by_category:
            self._by_category[topic.category] = []
        if topic.id not in self._by_category[topic.category]:
            self._by_category[topic.category].append(topic.id)

    def get(self, topic_id: str) -> HelpTopic | None:
        """Get a topic by ID."""
        # Normalize ID
        normalized = topic_id.lower().replace(" ", "_").replace("-", "_")
        return self._topics.get(normalized)

    def list_all(self) -> list[str]:
        """List all topic IDs."""
        return sorted(self._topics.keys())

    def list_by_category(self, category: str) -> list[str]:
        """List topic IDs in a category."""
        return self._by_category.get(category, [])

    def categories(self) -> list[str]:
        """List all categories."""
        return sorted(self._by_category.keys())

    def search(self, query: str, level: HelpLevel | None = None) -> list[str]:
        """
        Search topics by query string.

        Searches titles, content, and keywords.
        """
        query_lower = query.lower()
        matches = []

        for topic_id, topic in self._topics.items():
            # Check title
            if query_lower in topic.title.lower():
                matches.append(topic_id)
                continue

            # Check keywords
            if any(query_lower in kw for kw in topic.keywords):
                matches.append(topic_id)
                continue

            # Check content
            content = topic.get_content(level)
            if query_lower in content.lower():
                matches.append(topic_id)

        return matches


# Global registry instance
_registry = HelpRegistry()


# =============================================================================
# BUILT-IN TOPICS: CORE CONCEPTS
# =============================================================================

_registry.register(
    HelpTopic(
        id="generation",
        title="Image Generation",
        category="core",
        eli5="""Making pictures from words!

You write what you want to see (like "a happy cat in a garden"),
and the computer draws it for you. It's like magic drawing!""",
        casual="""Image generation creates images from text descriptions called "prompts".

How it works:
1. You write a description (e.g., "sunset over mountains")
2. The system processes your description
3. An image matching your description is generated

Key settings you can adjust:
- Steps: More steps = better quality but slower (default: 25)
- Size: Image dimensions in pixels (default: 1024x1024)
- Seed: Use the same seed to get the same image again

Quick start:
  client.generate_image("your description here")""",
        developer="""Image generation via Stable Diffusion through ComfyUI.

Pipeline:
1. Prompt → CLIP encoder → text conditioning tensor
2. Random latent noise (or seeded) → shape: (1, 4, H/8, W/8)
3. U-Net iterative denoising (N steps with CFG guidance)
4. VAE decoder → RGB image

Key parameters:
- steps: Denoising iterations (15-50, default 25)
- cfg_scale: Classifier-free guidance (1.0-20.0, default 7.0)
- seed: RNG seed for reproducibility (-1 = random)
- sampler: euler, euler_ancestral, dpmpp_2m_sde, etc.
- scheduler: normal, karras, sgm_uniform

VRAM usage scales with resolution:
- 512x512: ~4GB
- 1024x1024: ~8GB
- 2048x2048: ~16GB+ (uses tiled VAE)""",
        examples=[
            'from comfy_headless import ComfyClient\nclient = ComfyClient()\nresult = client.generate_image("a sunset over mountains")',
            'result = client.generate_image(\n    "cyberpunk city at night",\n    steps=30,\n    cfg_scale=7.5,\n    width=1280,\n    height=720\n)',
            'from comfy_headless import compile_workflow, GENERATION_PRESETS\nworkflow = compile_workflow("portrait of a woman", preset="cinematic")',
        ],
        related=["prompts", "presets", "workflows", "video"],
        keywords={"image", "generate", "create", "make", "picture", "photo"},
    )
)

_registry.register(
    HelpTopic(
        id="prompts",
        title="Writing Prompts",
        category="core",
        eli5="""Tell the computer what picture you want!

Just describe it like you're talking to a friend:
- "a fluffy cat sleeping on a red pillow"
- "a castle in the clouds at sunset"
- "a robot playing guitar"

The better you describe it, the better the picture!""",
        casual="""A prompt is your description of what you want to see in the image.

Writing good prompts:
1. Be specific: "golden retriever puppy" not just "dog"
2. Add style: "oil painting", "photograph", "anime style"
3. Describe lighting: "soft morning light", "dramatic shadows"
4. Include mood: "peaceful", "mysterious", "vibrant"
5. Add quality terms: "detailed", "high resolution", "sharp focus"

Structure: [subject], [description], [style], [lighting], [quality]

Example: "a red fox in an autumn forest, detailed fur, oil painting style,
golden hour lighting, masterpiece quality"

Negative prompts tell the system what to avoid:
"blurry, low quality, watermark, text, deformed" """,
        developer="""Prompts are tokenized via CLIP's BPE tokenizer.

Token limit: 77 tokens per prompt (CLIP architecture constraint).
Use BREAK keyword to extend beyond 77 tokens (requires compatible workflow).

Optimal structure:
[subject] [subject details], [environment], [style], [lighting],
[composition], [quality modifiers], [technical specs]

Prompt weighting (ComfyUI syntax):
- (word) = 1.1x emphasis
- ((word)) = 1.21x emphasis
- (word:1.5) = explicit 1.5x weight
- [word] = 0.9x de-emphasis

Negative prompt best practices:
- Always include: "blurry, low quality, worst quality"
- For people: "deformed, bad anatomy, extra limbs"
- For text: "watermark, text, signature, username"

AI Enhancement (v2.5.1):
- Uses Ollama with qwen2.5:1.5b for fast enhancement
- Concise mode: max 300 chars output
- Styles: minimal, balanced, detailed""",
        examples=[
            'from comfy_headless import enhance_prompt\nenhanced = enhance_prompt("a cat", style="balanced")\nprint(enhanced.enhanced)  # Detailed prompt under 300 chars',
            'from comfy_headless import validate_prompt, sanitize_prompt\nvalid = validate_prompt("my prompt")  # Returns bool\nclean = sanitize_prompt(user_input)  # Removes unsafe chars',
            'from comfy_headless import PromptIntelligence\npi = PromptIntelligence()  # use_few_shot=False by default\nresult = pi.enhance("sunset", style="detailed")',
        ],
        related=["generation", "enhancement"],
        keywords={"prompt", "text", "description", "write", "describe"},
    )
)

_registry.register(
    HelpTopic(
        id="presets",
        title="Generation Presets",
        category="core",
        eli5="""Ready-made recipes for different types of pictures!

Pick a recipe and everything is set up for you:
- "fast" - Quick pictures
- "quality" - Really good pictures (takes longer)
- "portrait" - Pictures of people
- "landscape" - Scenic views""",
        casual="""Presets are pre-configured settings optimized for different use cases.

Available presets:
- fast: Quick generation (~10 seconds)
- balanced: Good quality/speed trade-off (default)
- quality: High quality, slower (~45 seconds)
- cinematic: Movie-like, dramatic lighting
- portrait: Optimized for faces and people
- landscape: Wide scenic images
- anime: Anime/manga style

Usage:
  compile_workflow("your prompt", preset="cinematic")

Or list all presets:
  from comfy_headless import list_presets
  print(list_presets())""",
        developer="""Presets configure: resolution, steps, CFG scale, sampler, scheduler.

Preset definitions in workflows.py:
```
GENERATION_PRESETS = {
    "fast": {
        "steps": 15,
        "cfg_scale": 7.0,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "width": 1024,
        "height": 1024,
    },
    "quality": {
        "steps": 35,
        "cfg_scale": 7.5,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "width": 1024,
        "height": 1024,
    },
    ...
}
```

Custom presets:
- Add to TemplateLibrary for persistence
- Or pass parameters directly to compile_workflow()

VRAM-aware preset selection:
- get_recommended_preset() checks available VRAM
- Automatically suggests smaller resolution for low VRAM""",
        examples=[
            'from comfy_headless import compile_workflow, list_presets\nprint(list_presets())  # See all available presets\nworkflow = compile_workflow("sunset", preset="cinematic")',
            'from comfy_headless import GENERATION_PRESETS\nfor name, config in GENERATION_PRESETS.items():\n    print(f"{name}: {config["steps"]} steps")',
        ],
        related=["generation", "workflows"],
        keywords={"preset", "setting", "config", "recipe", "template"},
    )
)


# =============================================================================
# BUILT-IN TOPICS: FEATURES
# =============================================================================

_registry.register(
    HelpTopic(
        id="video",
        title="Video Generation",
        category="feature",
        eli5="""Making movies from pictures!

You describe what you want to see move, and it creates a short video.
It's like bringing a picture to life!""",
        casual="""Generate short video clips from text or images.

Available models (by quality/speed):
- AnimateDiff: Fast, good for stylized animation (8GB VRAM)
- SVD: Realistic motion, mid-quality (12GB VRAM)
- LTX: Fast, 720p quality (10GB VRAM)
- CogVideoX: High quality, slower (16GB VRAM)
- Wan: Very high quality (16GB VRAM)
- Hunyuan: Best quality, very slow (24GB VRAM)

Quick start:
  from comfy_headless import build_video_workflow
  workflow = build_video_workflow("a cat walking", preset="gentle")

Motion styles: gentle, dynamic, cinematic""",
        developer="""Video generation via multi-frame denoising with temporal attention.

Model comparison:
| Model       | Frames | Resolution | VRAM  | Speed    |
|-------------|--------|------------|-------|----------|
| AnimateDiff | 16     | 512x512    | 8GB   | Fast     |
| SVD         | 25     | 1024x576   | 12GB  | Medium   |
| LTX         | 49     | 1280x720   | 10GB  | Fast     |
| CogVideoX   | 49     | 720x480    | 16GB  | Slow     |
| Wan         | 81     | 832x480    | 16GB  | Medium   |
| Hunyuan     | 129    | 960x544    | 24GB  | V. Slow  |

VideoWorkflowBuilder:
- Selects appropriate workflow template
- Configures frame count, FPS, motion strength
- Handles VRAM optimization (offloading, tiling)

Export formats:
- Default: MP4 (H.264)
- Optional: GIF, WebM, frame sequences""",
        examples=[
            'from comfy_headless import VideoWorkflowBuilder, VideoModel\nbuilder = VideoWorkflowBuilder()\nworkflow = builder.build("ocean waves", model=VideoModel.LTX)',
            'from comfy_headless import build_video_workflow, list_video_presets\nprint(list_video_presets())\nworkflow = build_video_workflow("dancing flames", preset="dynamic")',
            "from comfy_headless import get_recommended_preset\npreset = get_recommended_preset()  # Based on available VRAM",
        ],
        related=["generation", "presets"],
        keywords={"video", "movie", "clip", "animation", "motion"},
    )
)

_registry.register(
    HelpTopic(
        id="enhancement",
        title="AI Prompt Enhancement",
        category="feature",
        eli5="""Making your descriptions better!

You write a simple idea, and the AI helps add more details
to make an even better picture.""",
        casual="""AI enhancement improves your prompts automatically.

How it works:
1. You write a simple prompt: "a cat"
2. AI adds details: lighting, style, quality terms
3. Result: Better, more detailed images

Enhancement styles:
- minimal: Light touch, stays close to original
- balanced: Good mix of additions (default)
- detailed: Maximum detail (use for complex scenes)

Usage:
  from comfy_headless import enhance_prompt
  result = enhance_prompt("sunset", style="balanced")
  print(result.enhanced)

Note: Requires Ollama running locally.""",
        developer="""AI enhancement via Ollama (v2.5.1).

Configuration:
- Model: qwen2.5:1.5b (fast) or qwen2.5:7b (quality)
- Max output: 300 characters (enforced)
- Few-shot: Disabled by default (use_few_shot=False)
- Verbose: Disabled by default (verbose=False)

PromptIntelligence class:
- check_ollama(): Verify connection
- enhance(prompt, style): Quick enhancement
- enhance_with_ai(prompt, style): Full AI enhancement
- analyze(prompt): Get prompt analysis

Caching:
- PromptCache stores results (SQLite backend)
- Cache key: hash(prompt + style + model)
- Default TTL: 24 hours

Settings (config.py):
- ollama.url: http://localhost:11434
- ollama.model: qwen2.5:1.5b
- ollama.quality_model: qwen2.5:7b
- ollama.timeout: 30 seconds""",
        examples=[
            'from comfy_headless import PromptIntelligence\npi = PromptIntelligence()\nif pi.check_ollama():\n    result = pi.enhance("forest", style="balanced")\n    print(f"Enhanced: {result.enhanced}")',
            'from comfy_headless import quick_enhance\nenhanced = quick_enhance("mountain sunset")  # Fast enhancement',
            'from comfy_headless import get_prompt_cache\ncache = get_prompt_cache()\nprint(f"Cache stats: {cache.stats()}")',
        ],
        related=["prompts", "generation"],
        keywords={"enhance", "improve", "ai", "ollama", "smart"},
    )
)

_registry.register(
    HelpTopic(
        id="workflows",
        title="Workflow System",
        category="feature",
        eli5="""Recipes for making pictures!

Each workflow is like a cooking recipe - it has steps
that the computer follows to make your picture.""",
        casual="""Workflows define the step-by-step process for generation.

Workflow types:
- txt2img: Text to image (most common)
- img2img: Modify an existing image
- inpaint: Edit specific parts of an image
- video: Generate video clips

The system uses templates that you customize with parameters.
Presets are pre-configured parameter sets for common use cases.

Usage:
  from comfy_headless import compile_workflow
  workflow = compile_workflow("my prompt", preset="quality")

  # Send to ComfyUI
  client.queue_workflow(workflow)""",
        developer="""Template-based workflow compilation system (v2.5.1).

Architecture:
```
JSON Template + Parameters → WorkflowCompiler → Compiled Workflow → ComfyUI API
```

Key components:
- WorkflowTemplate: Defines node structure and parameters
- TemplateLibrary: Manages and caches templates
- WorkflowCompiler: Injects parameters into templates
- WorkflowCache: Caches compiled workflows (hash-based)

DAG validation:
- validate_workflow_dag() checks for cycles
- Validates node connections before submission
- Reports orphaned nodes

Versioning:
- WorkflowVersion tracks template changes
- WorkflowSnapshot for rollback capability
- SnapshotManager persists snapshots

Custom templates:
1. Create JSON template with {{parameter}} placeholders
2. Register with TemplateLibrary
3. Use template_id in compile_workflow()""",
        examples=[
            'from comfy_headless import get_compiler, get_library\nlib = get_library()\nprint(lib.list_templates())\n\ncompiler = get_compiler()\nworkflow = compiler.compile("txt2img", prompt="sunset", preset="quality")',
            'from comfy_headless import validate_workflow_dag\nis_valid, errors = validate_workflow_dag(workflow)\nif not is_valid:\n    print(f"Validation errors: {errors}")',
            'from comfy_headless import get_snapshot_manager\nmanager = get_snapshot_manager()\nmanager.save("my_workflow", workflow)\n# Later: workflow = manager.load("my_workflow")',
        ],
        related=["generation", "presets"],
        keywords={"workflow", "template", "compile", "dag", "nodes"},
    )
)

_registry.register(
    HelpTopic(
        id="health",
        title="Health Checks",
        category="feature",
        eli5="""Checking if everything is working!

Like a doctor checkup, but for the computer. It makes sure
all the parts are running properly.""",
        casual="""Health checks verify all services are running properly.

What's checked:
- ComfyUI: Is it running and responding?
- Ollama: Is AI enhancement available?
- Disk space: Enough room for images?
- Memory: Enough RAM available?
- Circuit breakers: Any services temporarily disabled?

Quick check:
  from comfy_headless import check_health, is_healthy

  if is_healthy():
      print("All systems go!")
  else:
      report = check_health()
      print(report.status)""",
        developer="""Health monitoring system with auto-recovery (v2.5.1).

Components:
- HealthChecker: Performs individual checks
- HealthMonitor: Background monitoring with callbacks
- HealthReport: Aggregated status report

Check types:
- comfyui: HTTP GET /system_stats (timeout: 5s)
- ollama: HTTP GET /api/tags (timeout: 5s)
- disk: os.statvfs on temp directory (threshold: 1GB)
- memory: psutil.virtual_memory (threshold: 500MB)
- circuits: CircuitBreakerRegistry status

HealthStatus enum: HEALTHY, DEGRADED, UNHEALTHY

Auto-recovery actions:
- temp_files: Cleanup files older than 24h
- circuits: Reset after cooldown period
- memory: Trigger garbage collection

Integration:
- FastAPI endpoint: GET /api/health
- Prometheus metrics: /metrics (if enabled)""",
        examples=[
            'from comfy_headless import full_health_check\nreport = full_health_check()\nprint(report.to_dict())\nfor component, health in report.components.items():\n    print(f"{component}: {health.status}")',
            'from comfy_headless import HealthMonitor\nmonitor = HealthMonitor(interval=60, auto_recover=True)\nmonitor.on_unhealthy(lambda r: print(f"Alert: {r}"))\nmonitor.start()',
            'from comfy_headless import get_health_checker\nchecker = get_health_checker()\ncomfy_health = checker.check_component("comfyui")\nprint(f"ComfyUI: {comfy_health.status}")',
        ],
        related=["error:comfyui_offline", "error:ollama_offline"],
        keywords={"health", "status", "check", "monitor", "diagnose"},
    )
)


# =============================================================================
# BUILT-IN TOPICS: ERRORS
# =============================================================================

_registry.register(
    HelpTopic(
        id="error:comfyui_offline",
        title="ComfyUI Offline Error",
        category="error",
        eli5="""The picture-making program isn't running!

It's like trying to use a printer that's turned off.
We need to start it first.""",
        casual="""ComfyUI isn't running or can't be reached.

Quick fixes:
1. Start ComfyUI if it's not running
2. Check the URL (default: http://localhost:8188)
3. Make sure nothing else is using port 8188
4. Check your firewall settings

To verify it's running, open http://localhost:8188 in your browser.""",
        developer="""ComfyUI server not responding at configured URL.

Debug checklist:
1. Process: `tasklist | findstr python` or `ps aux | grep comfy`
2. Port: `netstat -an | findstr 8188`
3. Test endpoint: `curl http://localhost:8188/system_stats`
4. Firewall: Check Windows Defender / iptables rules
5. Config: `settings.comfyui.url`

Environment variable: COMFY_HEADLESS_COMFYUI__URL

Circuit breaker behavior:
- After 3 consecutive failures, circuit opens
- 30 second cooldown before retry
- Use circuit_registry.get("comfyui").reset() to force reset

Common causes:
- ComfyUI crashed (check for CUDA OOM errors)
- Wrong port configuration
- WSL networking issues (use host IP, not localhost)""",
        examples=[
            'from comfy_headless import check_health\nreport = check_health()\nprint(f"ComfyUI: {report.components["comfyui"]}")',
            'from comfy_headless import settings\nprint(f"ComfyUI URL: {settings.comfyui.url}")',
            'from comfy_headless import circuit_registry\ncircuit = circuit_registry.get("comfyui")\nprint(f"Circuit state: {circuit.state}")',
        ],
        related=["health", "error:timeout"],
        keywords={"offline", "connection", "comfyui", "unreachable", "refused"},
    )
)

_registry.register(
    HelpTopic(
        id="error:ollama_offline",
        title="Ollama Offline Error",
        category="error",
        eli5="""The AI helper isn't running!

Don't worry - you can still make pictures.
The AI just helps make your descriptions better.""",
        casual="""Ollama isn't running or can't be reached.

This only affects AI enhancement features. Image generation still works!

Quick fixes:
1. Start Ollama: `ollama serve`
2. Check it's running: http://localhost:11434
3. Make sure the model is downloaded: `ollama pull qwen2.5:1.5b`

To verify: `curl http://localhost:11434/api/tags`""",
        developer="""Ollama API server not responding.

Debug checklist:
1. Process: `ollama list` (will start server if needed)
2. API test: `curl http://localhost:11434/api/tags`
3. Model check: `ollama list | grep qwen2.5`
4. Config: `settings.ollama.url`

Required model:
- Fast: qwen2.5:1.5b (default)
- Quality: qwen2.5:7b
- Pull: `ollama pull qwen2.5:1.5b`

Graceful degradation:
- PromptIntelligence.check_ollama() returns False
- enhance_prompt() falls back to keyword enhancement
- No exception thrown, just reduced functionality

Environment variables:
- COMFY_HEADLESS_OLLAMA__URL
- COMFY_HEADLESS_OLLAMA__MODEL
- COMFY_HEADLESS_OLLAMA__TIMEOUT""",
        examples=[
            'from comfy_headless import PromptIntelligence\npi = PromptIntelligence()\nif pi.check_ollama():\n    print("Ollama available")\nelse:\n    print("Ollama offline - using keyword enhancement")',
            'from comfy_headless import settings\nprint(f"Ollama URL: {settings.ollama.url}")\nprint(f"Model: {settings.ollama.model}")',
        ],
        related=["enhancement", "health"],
        keywords={"ollama", "offline", "ai", "enhancement", "model"},
    )
)

_registry.register(
    HelpTopic(
        id="error:timeout",
        title="Generation Timeout",
        category="error",
        eli5="""The picture is taking too long to make!

It's like waiting for a really slow drawing.
Let's try making a smaller or simpler picture.""",
        casual="""Generation took longer than allowed.

Quick fixes:
1. Use a smaller image size (512x512 instead of 1024x1024)
2. Reduce the number of steps (15 instead of 25)
3. Use the "fast" preset
4. Close other programs using your GPU

For video:
- Use fewer frames
- Choose a faster model (AnimateDiff, LTX)""",
        developer="""Generation exceeded timeout threshold.

Timeout configuration:
- settings.comfyui.timeout_image: 120s (images)
- settings.comfyui.timeout_video: 600s (video)

Causes:
1. High resolution + many steps (exponential time)
2. Video with many frames
3. VRAM thrashing (using system RAM as fallback)
4. ComfyUI queue backed up
5. Model loading (first run after restart)

Solutions:
- Increase timeout: `settings.comfyui.timeout_image = 300`
- Use WebSocket client for progress updates
- Clear queue: POST /interrupt endpoint
- Use batch_size=1 for large generations

Retry behavior:
- Default: 2 retries with exponential backoff
- Circuit breaker opens after 3 consecutive timeouts
- Configure via RetryConfig in settings""",
        examples=[
            "from comfy_headless import settings\nsettings.comfyui.timeout_image = 300  # 5 minutes",
            'from comfy_headless import ComfyWSClient\nasync with ComfyWSClient() as ws:\n    async for progress in ws.track_progress(prompt_id):\n        print(f"Progress: {progress.percent}%")',
            'from comfy_headless import get_http_client\nclient = get_http_client()\nclient.post(f"{settings.comfyui.url}/interrupt")',
        ],
        related=["generation", "health", "error:vram"],
        keywords={"timeout", "slow", "long", "wait", "stuck"},
    )
)

_registry.register(
    HelpTopic(
        id="error:vram",
        title="Insufficient VRAM",
        category="error",
        eli5="""The computer's picture-making part is too full!

It's like trying to fit too much stuff in a small box.
Let's make a smaller picture so it fits.""",
        casual="""Your GPU doesn't have enough memory.

Quick fixes:
1. Use a smaller image size
2. Close other programs using your GPU
3. Use a smaller model
4. Enable low VRAM mode in ComfyUI

VRAM requirements (approximate):
- 512x512: ~4GB
- 1024x1024: ~8GB
- Video: 10-24GB depending on model""",
        developer="""GPU VRAM exhausted during operation.

VRAM usage breakdown (SD 1.5 / SDXL):
- Model: 2GB / 6GB (fp16)
- VAE: 0.5GB / 1GB
- Latents: scales with resolution
- Attention: scales quadratically with resolution

Mitigation strategies:
1. Resolution: Reduce to 512x512 or 768x768
2. Quantization: Use fp16 or 8-bit models
3. Tiled VAE: Enable for high-res decode
4. CPU offload: --lowvram flag in ComfyUI
5. Attention slicing: Reduce memory spikes
6. Clear cache: torch.cuda.empty_cache()

Health check integration:
- full_health_check() reports VRAM usage
- get_recommended_preset() suggests VRAM-appropriate settings

ComfyUI flags:
- --lowvram: Aggressive offloading
- --novram: CPU-only (very slow)
- --gpu-only: Keep everything on GPU""",
        examples=[
            'from comfy_headless import full_health_check\nreport = full_health_check()\nvram = report.components["comfyui"].details.get("vram_used_pct")\nprint(f"VRAM usage: {vram}%")',
            'from comfy_headless import get_recommended_preset\npreset = get_recommended_preset()\nprint(f"Recommended preset: {preset}")',
            "# Clear CUDA cache\nimport torch\ntorch.cuda.empty_cache()",
        ],
        related=["generation", "presets", "health"],
        keywords={"vram", "memory", "gpu", "cuda", "oom", "out of memory"},
    )
)

_registry.register(
    HelpTopic(
        id="error:validation",
        title="Validation Errors",
        category="error",
        eli5="""Something in your request wasn't quite right!

It's like filling out a form - some fields need
to be filled in a certain way.""",
        casual="""Your input didn't pass validation checks.

Common issues:
- Prompt too long (max ~500 characters recommended)
- Invalid image dimensions (must be multiples of 8)
- Unsafe characters in prompt
- Invalid preset name
- Missing required parameter

The error message will tell you exactly what went wrong.""",
        developer="""Pydantic-based validation system.

Validation functions:
- validate_prompt(text): Check prompt validity
- validate_dimensions(width, height): Check resolution
- validate_in_range(value, min, max): Range check
- validate_choice(value, options): Enum check
- validate_generation_params(**kwargs): Full validation

Decorators:
- @validated_prompt: Auto-validate prompt parameter
- @validated_dimensions: Auto-validate width/height

Sanitization:
- sanitize_prompt(text): Remove unsafe characters
- clamp_dimensions(w, h): Force to valid multiples

Dimension rules:
- Must be multiples of 8 (VAE requirement)
- Min: 64x64
- Max: 4096x4096 (VRAM permitting)
- Recommended max: 2048x2048

Custom validation:
- Extend ValidationError with custom codes
- Use InvalidPromptError, InvalidParameterError for specifics""",
        examples=[
            'from comfy_headless import validate_prompt, sanitize_prompt\nif not validate_prompt(user_input):\n    clean = sanitize_prompt(user_input)\n    print(f"Sanitized prompt: {clean}")',
            "from comfy_headless import validate_dimensions, clamp_dimensions\nif not validate_dimensions(1000, 1000):\n    w, h = clamp_dimensions(1000, 1000)  # Returns 1000, 1000 (valid multiple)",
            'from comfy_headless import validate_generation_params\ntry:\n    validate_generation_params(prompt="test", steps=100, width=100)\nexcept InvalidParameterError as e:\n    print(f"Validation error: {e}")',
        ],
        related=["prompts", "generation"],
        keywords={"validation", "invalid", "error", "parameter", "check"},
    )
)


# =============================================================================
# BUILT-IN TOPICS: API REFERENCE
# =============================================================================

_registry.register(
    HelpTopic(
        id="api:client",
        title="ComfyClient API",
        category="api",
        eli5="""The main way to make pictures with code!

ComfyClient is your remote control for making images.""",
        casual="""ComfyClient is the main interface for image generation.

Basic usage:
  from comfy_headless import ComfyClient

  client = ComfyClient()
  result = client.generate_image("a sunset")
  print(f"Image saved to: {result.path}")

Methods:
- generate_image(prompt, **kwargs): Generate one image
- generate_batch(prompts, **kwargs): Generate multiple
- queue_workflow(workflow): Queue custom workflow
- get_status(): Check ComfyUI status""",
        developer="""ComfyClient: High-level synchronous API wrapper.

Constructor:
```python
ComfyClient(
    url: str = None,  # Falls back to settings.comfyui.url
    timeout: float = None,  # Falls back to settings.comfyui.timeout_image
)
```

Methods:
- generate_image(prompt, steps, cfg_scale, width, height, seed, preset)
- generate_batch(prompts, **shared_kwargs)
- queue_workflow(workflow: dict) -> str (prompt_id)
- get_status() -> dict
- get_history(prompt_id) -> dict
- interrupt() -> None

Error handling:
- ComfyUIConnectionError: Can't reach server
- GenerationTimeoutError: Timeout exceeded
- GenerationFailedError: ComfyUI error during generation
- CircuitOpenError: Circuit breaker tripped

Context manager support:
```python
with ComfyClient() as client:
    result = client.generate_image("test")
```""",
        examples=[
            'from comfy_headless import ComfyClient\nwith ComfyClient() as client:\n    result = client.generate_image(\n        "cyberpunk city",\n        steps=30,\n        cfg_scale=7.5,\n        preset="cinematic"\n    )\n    print(f"Generated: {result.path}")',
            'client = ComfyClient(url="http://192.168.1.100:8188")\nfor prompt in ["cat", "dog", "bird"]:\n    result = client.generate_image(prompt)\n    print(f"{prompt}: {result.path}")',
        ],
        related=["generation", "api:websocket"],
        keywords={"client", "api", "generate", "comfyclient"},
    )
)

_registry.register(
    HelpTopic(
        id="api:websocket",
        title="WebSocket Client API",
        category="api",
        eli5="""A way to watch your picture being made!

Like watching a progress bar while downloading a file.""",
        casual="""ComfyWSClient provides real-time progress updates.

Use this when you want to:
- Show a progress bar
- Know exactly when generation completes
- Cancel generation mid-way

Basic usage:
  from comfy_headless import ComfyWSClient

  async with ComfyWSClient() as ws:
      prompt_id = await ws.queue_workflow(workflow)
      async for progress in ws.track_progress(prompt_id):
          print(f"Progress: {progress.percent}%")

Requires: pip install comfy-headless[websocket]""",
        developer="""ComfyWSClient: Async WebSocket client for real-time updates.

Requires [websocket] extra: `pip install comfy-headless[websocket]`

Constructor:
```python
ComfyWSClient(
    url: str = None,  # ws://localhost:8188/ws
    client_id: str = None,  # Auto-generated UUID
)
```

Async context manager:
```python
async with ComfyWSClient() as ws:
    # WebSocket connected
    ...
# Auto-disconnected
```

Methods:
- queue_workflow(workflow) -> str (prompt_id)
- track_progress(prompt_id) -> AsyncGenerator[WSProgress]
- wait_for_completion(prompt_id) -> dict (outputs)
- cancel(prompt_id) -> None

WSProgress dataclass:
- node: str (current node ID)
- value: int (current step)
- max: int (total steps)
- percent: float (0-100)

WSMessageType enum:
- EXECUTION_START, EXECUTION_CACHED, EXECUTING
- PROGRESS, EXECUTION_COMPLETE, EXECUTION_ERROR""",
        examples=[
            'import asyncio\nfrom comfy_headless import ComfyWSClient, compile_workflow\n\nasync def generate():\n    workflow = compile_workflow("sunset", preset="quality")\n    async with ComfyWSClient() as ws:\n        pid = await ws.queue_workflow(workflow)\n        async for p in ws.track_progress(pid):\n            print(f"\\r[{"=" * int(p.percent/5)}] {p.percent:.0f}%", end="")\n\nasyncio.run(generate())',
            'async with ComfyWSClient() as ws:\n    pid = await ws.queue_workflow(workflow)\n    outputs = await ws.wait_for_completion(pid)\n    print(f"Images: {outputs["images"]}")',
        ],
        related=["api:client", "generation"],
        keywords={"websocket", "async", "progress", "realtime"},
    )
)


# =============================================================================
# BUILT-IN TOPICS: CONFIGURATION
# =============================================================================

_registry.register(
    HelpTopic(
        id="config",
        title="Configuration",
        category="config",
        eli5="""Settings that control how everything works!

Like the settings on your phone - you can customize how things behave.""",
        casual="""Configuration controls all aspects of comfy_headless.

Quick access:
  from comfy_headless import settings

  print(settings.comfyui.url)  # ComfyUI URL
  print(settings.ollama.model)  # AI model

Environment variables override settings:
  COMFY_HEADLESS_COMFYUI__URL=http://192.168.1.100:8188

Config file: Create .env or comfy_headless.toml""",
        developer="""Pydantic-based settings with environment variable support.

Settings hierarchy:
1. Environment variables (highest priority)
2. Config file (comfy_headless.toml / .env)
3. Defaults (lowest priority)

Settings object structure:
```python
settings.comfyui.url          # http://localhost:8188
settings.comfyui.timeout_image  # 120.0
settings.comfyui.timeout_video  # 600.0
settings.ollama.url           # http://localhost:11434
settings.ollama.model         # qwen2.5:1.5b
settings.ollama.quality_model # qwen2.5:7b
settings.ollama.timeout       # 30.0
settings.temp_dir             # system temp
settings.log_level            # INFO
```

Environment variable format:
COMFY_HEADLESS_{SECTION}__{KEY}

Examples:
- COMFY_HEADLESS_COMFYUI__URL
- COMFY_HEADLESS_OLLAMA__MODEL
- COMFY_HEADLESS_LOG_LEVEL

Hot reload:
```python
from comfy_headless import reload_settings
reload_settings()  # Re-reads config files
```""",
        examples=[
            'from comfy_headless import settings, reload_settings\nprint(f"ComfyUI: {settings.comfyui.url}")\nprint(f"Ollama: {settings.ollama.url}")\n\n# Modify at runtime\nsettings.ollama.model = "qwen2.5:7b"\n\n# Reload from files\nreload_settings()',
            'import os\nos.environ["COMFY_HEADLESS_COMFYUI__URL"] = "http://remote:8188"\n\nfrom comfy_headless import reload_settings\nreload_settings()',
        ],
        related=["health"],
        keywords={"config", "settings", "environment", "url", "timeout"},
    )
)


# =============================================================================
# PUBLIC API
# =============================================================================


def get_help(topic: str, level: HelpLevel | None = None) -> str:
    """
    Get help for a topic.

    Args:
        topic: Topic ID or search term
        level: Override current verbosity level

    Returns:
        Formatted help text

    Examples:
        >>> get_help("generation")
        >>> get_help("timeout", level=HelpLevel.DEVELOPER)
    """
    level = level or _current_level

    # Normalize topic ID
    normalized = topic.lower().replace(" ", "_").replace("-", "_")

    # Handle error: prefix
    if normalized.startswith("error:"):
        pass  # Already formatted
    elif normalized in ["comfyui_offline", "ollama_offline", "timeout", "vram", "validation"]:
        normalized = f"error:{normalized}"

    # Direct lookup
    help_topic = _registry.get(normalized)
    if help_topic:
        return help_topic.format(level)

    # Try search
    matches = _registry.search(topic, level)
    if matches:
        if len(matches) == 1:
            return _registry.get(matches[0]).format(level)
        return f"Multiple matches for '{topic}':\n" + "\n".join(f"  - {m}" for m in matches)

    # Not found
    available = _registry.list_all()
    return f"Help topic '{topic}' not found.\n\nAvailable topics:\n" + "\n".join(
        f"  - {t}" for t in available[:15]
    )


def get_help_for_error(error_code: str, level: HelpLevel | None = None) -> str:
    """Get help specifically for an error code."""
    return get_help(f"error:{error_code.lower()}", level)


def list_topics() -> list[str]:
    """List all available help topics."""
    return _registry.list_all()


def search_help(query: str, level: HelpLevel | None = None) -> list[str]:
    """
    Search help topics by query.

    Returns list of matching topic IDs.
    """
    return _registry.search(query, level)


def format_quick_help(topic: str) -> str:
    """Get a one-line summary (ELI5) for a topic."""
    help_topic = _registry.get(topic.lower())
    if help_topic:
        # Get first line of ELI5
        return help_topic.eli5.split("\n")[0].strip()
    return f"No quick help for '{topic}'"


def format_help_list() -> str:
    """Format a categorized list of all topics."""
    lines = ["# Comfy Headless Help Topics", ""]

    for category in _registry.categories():
        # Format category name
        cat_title = category.replace("_", " ").title()
        lines.append(f"## {cat_title}")
        lines.append("")

        for topic_id in _registry.list_by_category(category):
            topic = _registry.get(topic_id)
            if topic:
                summary = topic.eli5.split("\n")[0][:60]
                lines.append(f"  - **{topic.title}** (`{topic_id}`): {summary}...")

        lines.append("")

    return "\n".join(lines)


def get_command_help(command: str) -> str:
    """Get help for a CLI command."""
    commands = {
        "generate": 'Generate an image from a text prompt.\n  comfy-headless generate "your prompt here" --preset quality',
        "video": 'Generate a video from a text prompt.\n  comfy-headless video "your prompt" --model ltx --frames 49',
        "enhance": 'Enhance a prompt using AI.\n  comfy-headless enhance "simple prompt" --style detailed',
        "status": "Check ComfyUI and system status.\n  comfy-headless status",
        "presets": "List available presets.\n  comfy-headless presets",
        "health": "Run health checks.\n  comfy-headless health --verbose",
    }
    return commands.get(command.lower(), f"Unknown command: {command}")


def get_api_reference(module: str = None) -> str:
    """
    Get API reference for a module.

    Args:
        module: Specific module or None for overview
    """
    if module:
        topic = _registry.get(f"api:{module.lower()}")
        if topic:
            return topic.format(HelpLevel.DEVELOPER)
        return f"No API reference for module: {module}"

    # Overview
    lines = [
        "# Comfy Headless API Reference",
        "",
        "## Core Classes",
        "  - ComfyClient: Synchronous image generation",
        "  - ComfyWSClient: Async WebSocket client [websocket]",
        "  - PromptIntelligence: AI prompt enhancement [ai]",
        "",
        "## Workflows",
        "  - WorkflowCompiler: Compile templates to workflows",
        "  - TemplateLibrary: Manage workflow templates",
        "  - VideoWorkflowBuilder: Build video generation workflows",
        "",
        "## Utilities",
        "  - HealthChecker: System health monitoring",
        "  - CircuitBreaker: Fault tolerance",
        "  - TempFileManager: Automatic cleanup",
        "",
        "For detailed help: get_help('api:client', level=HelpLevel.DEVELOPER)",
    ]
    return "\n".join(lines)
