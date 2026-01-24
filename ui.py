"""
Comfy Headless - Gradio UI
Headless ComfyUI interface for image and video generation

Features:
- Template-based workflow compilation
- Multi-model video generation support
- 2026 Design Best Practices (Warm Neutral + Emerald themes)
"""

import random
import tempfile
import time

import gradio as gr

from .client import ComfyClient
from .config import settings
from .help_system import format_help_list, get_help, get_help_for_error, list_topics
from .theme import create_comfy_theme, get_css
from .video import (
    VIDEO_PRESETS,
)
from .workflows import (
    GENERATION_PRESETS,
    ParameterDef,
    ParameterType,
    WorkflowCategory,
    WorkflowTemplate,
    get_library,
)

# Initialize client
client = ComfyClient()

# ============================================================================
# PROMPT STUDIO - LLM Enhancement
# ============================================================================

STYLE_MODIFIERS = {
    "Photorealistic": "photorealistic, ultra detailed, sharp focus, professional photography, 8K UHD",
    "Artistic": "artistic, painterly, expressive brushstrokes, fine art, masterpiece",
    "Anime": "anime style, cel shaded, vibrant colors, clean lines, Studio Ghibli inspired",
    "Fantasy": "fantasy art, magical, ethereal lighting, detailed illustration, concept art",
    "Sci-Fi": "sci-fi, futuristic, sleek design, technological, cyberpunk aesthetic",
    "Abstract": "abstract art, geometric shapes, bold colors, modern art, contemporary",
}

ENHANCEMENT_PROMPTS = {
    "Fast": """Enhance this image prompt by adding 2-3 quality tags. Keep it concise.
Original: {prompt}
Style: {style}
Enhanced prompt (just the prompt, no explanation):""",
    "Standard": """You are a Stable Diffusion prompt expert. Enhance this prompt with:
- Specific artistic details (lighting, atmosphere, composition)
- Technical quality tags (detailed, sharp focus, professional)
- Style-appropriate modifiers

Original: {prompt}
Style: {style}

Return ONLY the enhanced prompt, nothing else:""",
    "Cinematic": """You are a master cinematographer and Stable Diffusion expert. Transform this prompt into a cinematic masterpiece with:
- Dramatic lighting and atmosphere
- Professional composition (rule of thirds, leading lines)
- Film-quality descriptors (cinematic, award-winning, 8K)
- Mood and emotion enhancers

Original: {prompt}
Style: {style}

Return ONLY the enhanced prompt as a single paragraph:""",
}


def enhance_prompt_with_ollama(prompt: str, mode: str, style: str) -> tuple[str, str]:
    """
    Enhance a prompt using Ollama LLM.

    Returns: (enhanced_prompt, status_message)
    """
    import httpx

    if not prompt.strip():
        return "", "Please enter a prompt first"

    style_modifier = STYLE_MODIFIERS.get(style, "")
    system_prompt = ENHANCEMENT_PROMPTS.get(mode, ENHANCEMENT_PROMPTS["Standard"])

    try:
        # Call Ollama API
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",  # or qwen2.5, mistral, etc.
                "prompt": system_prompt.format(prompt=prompt, style=style_modifier),
                "stream": False,
            },
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            enhanced = result.get("response", "").strip()
            # Clean up any quotes or extra formatting
            enhanced = enhanced.strip("\"'")
            return enhanced, f"Enhanced with {mode} mode"
        else:
            return prompt, f"Ollama error: {response.status_code}"

    except httpx.ConnectError:
        return prompt, "Ollama not running - using original prompt"
    except Exception as e:
        return prompt, f"Error: {str(e)[:50]}"


# Cache for online status during UI construction to avoid repeated slow checks
# This is reset after create_ui() completes
_cached_online_status: bool | None = None


def _is_online_cached() -> bool:
    """
    Check if ComfyUI is online with caching for UI construction.

    During create_ui(), multiple components need to check online status.
    Without caching, each check takes up to 2 seconds when offline,
    causing the UI to hang for 18+ seconds total.
    """
    global _cached_online_status
    if _cached_online_status is not None:
        return _cached_online_status
    return client.is_online()


def format_error_with_suggestions(error_msg: str, error_code: str | None = None) -> str:
    """
    Format an error message with helpful suggestions.

    Args:
        error_msg: The base error message
        error_code: Optional error code like 'COMFYUI_OFFLINE' for help lookup

    Returns:
        Formatted error string with suggestions
    """
    suggestions = []

    # Try to get suggestions from help system
    if error_code:
        help_text = get_help_for_error(error_code)
        suggestions = [
            line.strip("- ") for line in help_text.split("\n") if line.strip().startswith("-")
        ][:3]

    # Build message
    if suggestions:
        tips = "\n".join(f"• {s}" for s in suggestions[:3])  # Max 3 suggestions
        return f"**Error**: {error_msg}\n\n**Try:**\n{tips}"

    return f"**Error**: {error_msg}"


# ============================================================================
# CONSTANTS
# ============================================================================

# Map friendly names to workflow presets
PRESETS = {
    "Draft (Fast)": "draft",
    "Fast (768px)": "fast",
    "Quality (1024px)": "quality",
    "HD (High Detail)": "hd",
    "Portrait": "portrait",
    "Landscape": "landscape",
    "Cinematic (Wide)": "cinematic",
    "Square": "square",
}

# Video preset friendly names
VIDEO_PRESET_NAMES = {
    "Quick (Lightning)": "quick",
    "Standard": "standard",
    "Quality": "quality",
    "Cinematic": "cinematic",
    "Portrait": "portrait",
    "Action (Dynamic)": "action",
    "SVD Short": "svd_short",
    "SVD Long": "svd_long",
    "CogVideoX": "cogvideo",
    "Hunyuan": "hunyuan",
    "Hunyuan Fast": "hunyuan_fast",
}

DEFAULT_NEGATIVE = "ugly, blurry, low quality, distorted, deformed, disfigured, bad anatomy, watermark, signature, text"

EXAMPLE_PROMPTS = [
    "A mystical forest with glowing mushrooms and fairy lights, fantasy art",
    "Portrait of an elegant woman with flowing hair, art nouveau style",
    "A steampunk city at sunset with airships in the sky",
    "Cyberpunk street scene with neon signs and rain reflections",
    "A cozy cabin in snowy mountains, warm light from windows",
    "Abstract fluid art with vibrant colors, macro photography",
]

# ============================================================================
# STATUS FUNCTIONS
# ============================================================================


def get_status() -> str:
    """Get ComfyUI connection status"""
    if not _is_online_cached():
        return "🔴 **Offline** - ComfyUI not running"

    stats = client.get_system_stats()
    if stats:
        devices = stats.get("devices", [])
        if devices:
            gpu = devices[0]
            vram_used = gpu.get("vram_total", 0) - gpu.get("vram_free", 0)
            vram_total = gpu.get("vram_total", 0)
            vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
            vram_used_gb = vram_used / (1024**3)
            vram_total_gb = vram_total / (1024**3)
            return f"🟢 **Online** | GPU: {gpu.get('name', 'Unknown')} | VRAM: {vram_used_gb:.1f}/{vram_total_gb:.1f} GB ({vram_pct:.0f}%)"

    return "🟢 **Online**"


def get_queue_status() -> str:
    """Get current queue status"""
    if not _is_online_cached():
        return "Offline"

    queue = client.get_queue()
    running = len(queue.get("queue_running", []))
    pending = len(queue.get("queue_pending", []))
    return f"Running: {running} | Pending: {pending}"


def _get_detailed_queue_status() -> str:
    """Get detailed queue status with formatting."""
    if not _is_online_cached():
        return "🔴 **ComfyUI Offline** - Start ComfyUI to see queue status"

    queue = client.get_queue()
    running = len(queue.get("queue_running", []))
    pending = len(queue.get("queue_pending", []))

    if running == 0 and pending == 0:
        return "🟢 **Queue Empty** - Ready for new jobs"
    elif running > 0:
        return f"🟡 **Processing** - {running} running, {pending} pending"
    else:
        return f"🔵 **Queued** - {pending} jobs waiting"


def _get_current_job_info() -> str:
    """Get info about the currently running job."""
    if not _is_online_cached():
        return "*ComfyUI offline*"

    queue = client.get_queue()
    running = queue.get("queue_running", [])

    if not running:
        return "*No job currently running*"

    # Get the first running job
    job = running[0]
    prompt_id = job[1] if len(job) > 1 else "unknown"

    return f"""**Currently Processing:**
- Job ID: `{prompt_id[:16]}...`
- Status: Running
"""


def refresh_models() -> tuple:
    """Get available checkpoints. Returns (choices, interactive) for Gradio."""
    if not _is_online_cached():
        return ["ComfyUI offline - start to load models"]
    return client.get_checkpoints() or ["(No models found)"]


def get_models_update():
    """Get models with interactive state for dropdown update."""
    if not _is_online_cached():
        return gr.update(choices=["ComfyUI offline"], value=None, interactive=False)
    models = client.get_checkpoints() or ["(No models found)"]
    return gr.update(choices=models, value=models[0] if models else None, interactive=True)


def refresh_samplers() -> list[str]:
    """Get available samplers"""
    if not _is_online_cached():
        return ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde"]
    return client.get_samplers() or ["euler", "euler_ancestral"]


def refresh_schedulers() -> list[str]:
    """Get available schedulers"""
    if not _is_online_cached():
        return ["normal", "karras", "exponential", "sgm_uniform"]
    return client.get_schedulers() or ["normal", "karras"]


# Sampler/Scheduler info for the Models tab
SAMPLER_INFO = {
    "euler": ("Fast", "Good"),
    "euler_ancestral": ("Fast", "Creative"),
    "dpmpp_2m": ("Medium", "Excellent"),
    "dpmpp_2m_sde": ("Medium", "Excellent"),
    "dpmpp_sde": ("Slow", "Excellent"),
    "ddim": ("Fast", "Good"),
    "uni_pc": ("Fast", "Good"),
    "lcm": ("Very Fast", "Good"),
}

SCHEDULER_INFO = {
    "normal": "General use",
    "karras": "Detailed images",
    "exponential": "Smooth transitions",
    "sgm_uniform": "SDv2 models",
    "simple": "Basic",
    "ddim_uniform": "DDIM sampler",
}


def _get_samplers_with_info() -> list:
    """Get samplers with speed/quality info."""
    samplers = refresh_samplers()
    result = []
    for s in samplers:
        info = SAMPLER_INFO.get(s, ("Medium", "Good"))
        result.append([s, info[0], info[1]])
    return result


def _get_schedulers_with_info() -> list:
    """Get schedulers with usage info."""
    schedulers = refresh_schedulers()
    result = []
    for s in schedulers:
        info = SCHEDULER_INFO.get(s, "General")
        result.append([s, info])
    return result


# Common custom node packs and their prefixes
CUSTOM_NODE_SOURCES = {
    # AnimateDiff
    "ADE_": ("ComfyUI-AnimateDiff-Evolved", "Kosinkadink/ComfyUI-AnimateDiff-Evolved"),
    "AnimateDiff": ("ComfyUI-AnimateDiff-Evolved", "Kosinkadink/ComfyUI-AnimateDiff-Evolved"),
    # Impact Pack
    "Impact": ("ComfyUI Impact Pack", "ltdrdata/ComfyUI-Impact-Pack"),
    "SAM": ("ComfyUI Impact Pack", "ltdrdata/ComfyUI-Impact-Pack"),
    "SEGS": ("ComfyUI Impact Pack", "ltdrdata/ComfyUI-Impact-Pack"),
    "Detector": ("ComfyUI Impact Pack", "ltdrdata/ComfyUI-Impact-Pack"),
    # ControlNet
    "ControlNet": ("ComfyUI ControlNet Aux", "Fannovel16/comfyui_controlnet_aux"),
    "Aux": ("ComfyUI ControlNet Aux", "Fannovel16/comfyui_controlnet_aux"),
    # IPAdapter
    "IPAdapter": ("ComfyUI IPAdapter Plus", "cubiq/ComfyUI_IPAdapter_plus"),
    # Video
    "VHS_": ("ComfyUI-VideoHelperSuite", "Kosinkadink/ComfyUI-VideoHelperSuite"),
    "Video": ("ComfyUI-VideoHelperSuite", "Kosinkadink/ComfyUI-VideoHelperSuite"),
    # Flux
    "Flux": ("x-flux-comfyui", "XLabs-AI/x-flux-comfyui"),
    # CogVideo
    "CogVideo": ("ComfyUI-CogVideoXWrapper", "kijai/ComfyUI-CogVideoXWrapper"),
    # Hunyuan
    "Hunyuan": ("ComfyUI-HunyuanVideoWrapper", "kijai/ComfyUI-HunyuanVideoWrapper"),
    # LTX Video
    "LTX": ("ComfyUI-LTXVideo", "Lightricks/ComfyUI-LTXVideo"),
    # WAN
    "WAN": ("ComfyUI-WAN", "Wan-Video/ComfyUI-WAN"),
    # Efficiency nodes
    "Efficient": ("efficiency-nodes-comfyui", "jags111/efficiency-nodes-comfyui"),
    # WAS Suite
    "WAS": ("was-node-suite-comfyui", "WASasquatch/was-node-suite-comfyui"),
    # ComfyUI Essentials
    "Essentials": ("ComfyUI_essentials", "cubiq/ComfyUI_essentials"),
    # Frame Interpolation
    "FILM": ("ComfyUI-Frame-Interpolation", "Fannovel16/ComfyUI-Frame-Interpolation"),
    "RIFE": ("ComfyUI-Frame-Interpolation", "Fannovel16/ComfyUI-Frame-Interpolation"),
}


def _get_node_source_hint(node_type: str) -> str:
    """Get a hint about where a custom node might come from."""
    for prefix, (pack_name, _repo) in CUSTOM_NODE_SOURCES.items():
        if prefix in node_type:
            return f"→ likely from **{pack_name}**"

    # Generic hint for unknown nodes
    if "_" in node_type:
        # Might be from a custom pack
        prefix = node_type.split("_")[0]
        return f"→ search for '{prefix}' in ComfyUI Manager"

    return "→ search in ComfyUI Manager"


def refresh_motion_models() -> list[str]:
    """Get available motion models for AnimateDiff"""
    if not _is_online_cached():
        return ["ComfyUI offline - start to load models"]
    models = client.get_motion_models()
    return models if models else ["v3_sd15_mm.ckpt", "mm_sd_v15_v2.ckpt"]


def get_motion_models_update():
    """Get motion models with interactive state for dropdown update."""
    if not _is_online_cached():
        return gr.update(choices=["ComfyUI offline"], value=None, interactive=False)
    models = client.get_motion_models() or ["v3_sd15_mm.ckpt", "mm_sd_v15_v2.ckpt"]
    return gr.update(choices=models, value=models[0] if models else None, interactive=True)


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================


def apply_preset(preset_name: str):
    """Apply a preset's settings"""
    # Map friendly name to preset key
    preset_key = PRESETS.get(preset_name, "quality")
    preset = GENERATION_PRESETS.get(preset_key, GENERATION_PRESETS["quality"])
    return preset.get("width", 1024), preset.get("height", 1024), preset.get("steps", 25)


def apply_video_preset(preset_name: str):
    """Apply a video preset's settings"""
    # Map friendly name to preset key
    preset_key = VIDEO_PRESET_NAMES.get(preset_name, "standard")
    preset = VIDEO_PRESETS.get(preset_key)
    if preset:
        return (
            preset.width,
            preset.height,
            preset.frames,
            preset.fps,
            preset.steps,
        )
    # Default fallback
    return 512, 512, 16, 8, 20


def random_prompt() -> str:
    """Get a random example prompt"""
    return random.choice(EXAMPLE_PROMPTS)


def generate_image(
    prompt: str,
    negative_prompt: str,
    checkpoint: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    progress=gr.Progress(),
) -> tuple:
    """Generate an image and return it with metadata"""

    if not prompt.strip():
        return None, format_error_with_suggestions("Please enter a prompt", "INVALID_PROMPT")

    if not client.is_online():
        return None, format_error_with_suggestions("ComfyUI is not running", "COMFYUI_OFFLINE")

    # Use random seed if -1
    actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)

    progress(0.1, desc="Building workflow...")

    # Build and queue workflow
    workflow = client.build_txt2img_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
        checkpoint=checkpoint if checkpoint and checkpoint != "(ComfyUI offline)" else "",
        width=int(width),
        height=int(height),
        steps=int(steps),
        cfg=float(cfg),
        sampler=sampler,
        scheduler=scheduler,
        seed=actual_seed,
    )

    progress(0.2, desc="Queuing prompt...")
    prompt_id = client.queue_prompt(workflow)

    if not prompt_id:
        return None, format_error_with_suggestions("Failed to queue prompt", "QUEUE_ERROR")

    # Poll for completion
    progress(0.3, desc="Generating...")
    start_time = time.time()
    timeout = 300

    while time.time() - start_time < timeout:
        history = client.get_history(prompt_id)

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})

            if status.get("status_str") == "error":
                error_msg = status.get("messages", [["Unknown error"]])[0]
                return None, format_error_with_suggestions(
                    f"Generation failed: {error_msg}", "GENERATION_FAILED"
                )

            if status.get("completed", False):
                # Extract images
                outputs = entry.get("outputs", {})
                for _node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            filename = img.get("filename")
                            subfolder = img.get("subfolder", "")

                            progress(0.9, desc="Downloading image...")
                            image_bytes = client.get_image(filename, subfolder)

                            if image_bytes:
                                # Save to temp file for Gradio
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                                    f.write(image_bytes)
                                    temp_path = f.name

                                elapsed = time.time() - start_time
                                metadata = f"""**Generation Complete** ({elapsed:.1f}s)
- Seed: `{actual_seed}`
- Model: `{checkpoint}`
- Size: {width}x{height}
- Steps: {steps} | CFG: {cfg}
- Sampler: {sampler} ({scheduler})"""

                                progress(1.0, desc="Done!")
                                return temp_path, metadata

                return None, format_error_with_suggestions("No images in output", "NO_OUTPUT")

        # Update progress based on elapsed time (estimate)
        elapsed = time.time() - start_time
        est_progress = min(0.3 + (elapsed / 60) * 0.6, 0.85)
        progress(est_progress, desc=f"Generating... ({elapsed:.0f}s)")
        time.sleep(0.5)

    return None, format_error_with_suggestions("Generation timed out", "GENERATION_TIMEOUT")


def cancel_generation():
    """Cancel current generation"""
    if client.cancel_current():
        return "Cancelled current job"
    return "Failed to cancel (nothing running?)"


def generate_video(
    prompt: str,
    negative_prompt: str,
    checkpoint: str,
    motion_model: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    steps: int,
    cfg: float,
    motion_scale: float,
    seed: int,
    progress=gr.Progress(),
) -> tuple:
    """Generate a video and return it with metadata"""

    if not prompt.strip():
        return None, format_error_with_suggestions("Please enter a prompt", "INVALID_PROMPT")

    if not client.is_online():
        return None, format_error_with_suggestions("ComfyUI is not running", "COMFYUI_OFFLINE")

    # Use random seed if -1
    actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)

    progress(0.1, desc="Building video workflow...")

    # Build and queue workflow
    workflow = client.build_video_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
        checkpoint=checkpoint if checkpoint and checkpoint != "(ComfyUI offline)" else "",
        motion_model=motion_model if motion_model and motion_model != "(ComfyUI offline)" else "",
        width=int(width),
        height=int(height),
        frames=int(frames),
        fps=int(fps),
        steps=int(steps),
        cfg=float(cfg),
        seed=actual_seed,
        motion_scale=float(motion_scale),
    )

    progress(0.2, desc="Queuing prompt...")
    prompt_id = client.queue_prompt(workflow)

    if not prompt_id:
        return None, format_error_with_suggestions("Failed to queue prompt", "QUEUE_ERROR")

    # Poll for completion
    progress(0.3, desc="Generating video...")
    start_time = time.time()
    timeout = 600  # 10 minutes for video

    while time.time() - start_time < timeout:
        history = client.get_history(prompt_id)

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})

            if status.get("status_str") == "error":
                error_msg = status.get("messages", [["Unknown error"]])[0]
                return None, format_error_with_suggestions(
                    f"Generation failed: {error_msg}", "GENERATION_FAILED"
                )

            if status.get("completed", False):
                # Extract videos (VHS_VideoCombine outputs to 'gifs' key)
                outputs = entry.get("outputs", {})
                for _node_id, node_output in outputs.items():
                    if "gifs" in node_output:
                        for vid in node_output["gifs"]:
                            filename = vid.get("filename")
                            subfolder = vid.get("subfolder", "")

                            progress(0.9, desc="Downloading video...")
                            video_bytes = client.get_video(filename, subfolder)

                            if video_bytes:
                                # Save to temp file for Gradio
                                suffix = ".mp4" if filename.endswith(".mp4") else ".webm"
                                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                                    f.write(video_bytes)
                                    temp_path = f.name

                                elapsed = time.time() - start_time
                                metadata = f"""**Video Complete** ({elapsed:.1f}s)
- Seed: `{actual_seed}`
- Model: `{checkpoint}`
- Motion: `{motion_model}`
- Size: {width}x{height} @ {frames} frames
- Steps: {steps} | CFG: {cfg} | Motion: {motion_scale}x"""

                                progress(1.0, desc="Done!")
                                return temp_path, metadata

                return None, format_error_with_suggestions("No video in output", "NO_OUTPUT")

        # Update progress based on elapsed time
        elapsed = time.time() - start_time
        est_progress = min(0.3 + (elapsed / 120) * 0.6, 0.85)
        progress(est_progress, desc=f"Generating video... ({elapsed:.0f}s)")
        time.sleep(1.0)

    return None, format_error_with_suggestions("Video generation timed out", "GENERATION_TIMEOUT")


# ============================================================================
# HISTORY FUNCTIONS
# ============================================================================


def get_history_list() -> list[list[str]]:
    """Get generation history as table data"""
    if not _is_online_cached():
        return []

    history = client.get_history()
    rows = []

    for prompt_id, entry in list(history.items())[:20]:  # Last 20
        status = entry.get("status", {})
        outputs = entry.get("outputs", {})

        # Count images
        image_count = 0
        for node_output in outputs.values():
            if "images" in node_output:
                image_count += len(node_output["images"])

        status_str = (
            "✅"
            if status.get("completed")
            else "❌"
            if status.get("status_str") == "error"
            else "⏳"
        )

        rows.append(
            [
                prompt_id[:8] + "...",
                status_str,
                str(image_count),
            ]
        )

    return rows


# ============================================================================
# BUILD UI
# ============================================================================


def create_ui():
    """Create the Gradio interface with 2026 design best practices."""
    global _cached_online_status

    # Cache online status once at start of UI construction
    # This prevents 9+ slow network checks (2 sec each) when ComfyUI is offline
    _cached_online_status = client.is_online()

    # Cache buster version
    import datetime

    _ui_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"UI Version: {_ui_version}")

    # Store theme/css for launch() - Gradio 6.0 requirement
    global _theme, _custom_css
    _theme = create_comfy_theme()
    _custom_css = get_css()

    with gr.Blocks(title=f"Comfy Headless v{_ui_version}") as app:
        # Prompt Studio Sidebar
        with gr.Sidebar(
            position="right", open=False, elem_id="prompt-studio-sidebar"
        ):
            gr.Markdown("## Prompt Studio")
            gr.Markdown("Enhance your prompts with AI assistance.")

            enhance_mode = gr.Radio(
                choices=["Fast", "Standard", "Cinematic"],
                value="Standard",
                label="Enhancement Mode",
            )

            with gr.Accordion("Style Presets", open=False):
                style_category = gr.Dropdown(
                    choices=[
                        "Photorealistic",
                        "Artistic",
                        "Anime",
                        "Fantasy",
                        "Sci-Fi",
                        "Abstract",
                    ],
                    value="Photorealistic",
                    label="Style Category",
                )

            enhance_btn = gr.Button("Enhance Prompt", variant="primary")
            enhance_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("*Powered by local LLM via Ollama*")

        with gr.Tabs():
            # =================================================================
            # TAB: Image
            # =================================================================
            with gr.Tab("Image", id="image"):
                with gr.Row():
                    # Left column - inputs
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                        )

                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Optional: things to avoid in the image...",
                            lines=2,
                        )

                        with gr.Accordion("Settings", open=True):
                            preset = gr.Dropdown(
                                choices=list(PRESETS.keys()),
                                value="Quality (1024px)",
                                label="Preset",
                            )

                            with gr.Row():
                                width = gr.Slider(256, 2048, value=768, step=64, label="Width")
                                height = gr.Slider(256, 2048, value=768, step=64, label="Height")

                            steps = gr.Slider(1, 100, value=25, step=1, label="Steps")
                            cfg = gr.Slider(1, 20, value=7.0, step=0.5, label="CFG Scale")

                            with gr.Row():
                                sampler = gr.Dropdown(
                                    choices=refresh_samplers(),
                                    value="euler_ancestral",
                                    label="Sampler",
                                )
                                scheduler = gr.Dropdown(
                                    choices=refresh_schedulers(),
                                    value="normal",
                                    label="Scheduler",
                                )

                            checkpoint = gr.Dropdown(
                                choices=refresh_models(),
                                label="Model",
                                allow_custom_value=True,
                            )

                            seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

                        with gr.Row():
                            generate_btn = gr.Button(
                                "Generate",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                scale=3,
                            )
                            cancel_btn = gr.Button("Cancel", scale=1)

                    # Right column - output
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Output",
                            type="filepath",
                            height=512,
                        )
                        output_info = gr.Markdown("")

                # Event handlers
                preset.change(
                    apply_preset,
                    inputs=[preset],
                    outputs=[width, height, steps],
                )

                generate_btn.click(
                    generate_image,
                    inputs=[
                        prompt,
                        negative_prompt,
                        checkpoint,
                        width,
                        height,
                        steps,
                        cfg,
                        sampler,
                        scheduler,
                        seed,
                    ],
                    outputs=[output_image, output_info],
                )

                cancel_btn.click(cancel_generation, outputs=[output_info])

                # Connect Prompt Studio enhance button to Image tab prompt
                def enhance_image_prompt(current_prompt, mode, style):
                    enhanced, status = enhance_prompt_with_ollama(current_prompt, mode, style)
                    return enhanced, status

                enhance_btn.click(
                    enhance_image_prompt,
                    inputs=[prompt, enhance_mode, style_category],
                    outputs=[prompt, enhance_status],
                )

            # =================================================================
            # TAB: Video Generation
            # =================================================================
            with gr.Tab("Video", id="video"):
                with gr.Row():
                    # Left column - inputs
                    with gr.Column(scale=1):
                        video_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=3,
                        )

                        video_negative = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Optional: things to avoid in the video...",
                            lines=2,
                        )

                        with gr.Accordion("Video Settings", open=True):
                            video_preset = gr.Dropdown(
                                choices=list(VIDEO_PRESET_NAMES.keys()),
                                value="Standard",
                                label="Video Preset",
                                info="Quick settings for different video styles",
                            )

                            with gr.Row():
                                video_width = gr.Slider(
                                    256, 1024, value=512, step=64, label="Width"
                                )
                                video_height = gr.Slider(
                                    256, 1024, value=512, step=64, label="Height"
                                )

                            with gr.Row():
                                video_frames = gr.Slider(8, 32, value=16, step=1, label="Frames")
                                video_fps = gr.Slider(4, 24, value=8, step=1, label="FPS")

                            video_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                            video_cfg = gr.Slider(1, 15, value=7.0, step=0.5, label="CFG Scale")
                            video_motion_scale = gr.Slider(
                                0.5, 2.0, value=1.0, step=0.1, label="Motion Scale"
                            )

                            video_checkpoint = gr.Dropdown(
                                choices=refresh_models(),
                                label="Base Model",
                                allow_custom_value=True,
                            )

                            video_motion_model = gr.Dropdown(
                                choices=refresh_motion_models(),
                                label="Motion Model",
                                allow_custom_value=True,
                            )

                            video_seed = gr.Number(
                                value=-1, label="Seed (-1 = random)", precision=0
                            )

                        with gr.Row():
                            video_generate_btn = gr.Button(
                                "Generate Video",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                scale=3,
                            )
                            video_cancel_btn = gr.Button("Cancel", scale=1)

                    # Right column - output
                    with gr.Column(scale=1):
                        output_video = gr.Video(
                            label="Output",
                            height=512,
                        )
                        video_output_info = gr.Markdown("")

                # Video event handlers
                video_preset.change(
                    apply_video_preset,
                    inputs=[video_preset],
                    outputs=[video_width, video_height, video_frames, video_fps, video_steps],
                )

                video_generate_btn.click(
                    generate_video,
                    inputs=[
                        video_prompt,
                        video_negative,
                        video_checkpoint,
                        video_motion_model,
                        video_width,
                        video_height,
                        video_frames,
                        video_fps,
                        video_steps,
                        video_cfg,
                        video_motion_scale,
                        video_seed,
                    ],
                    outputs=[output_video, video_output_info],
                )

                video_cancel_btn.click(cancel_generation, outputs=[video_output_info])

            # =================================================================
            # TAB: Queue & History
            # =================================================================
            with gr.Tab("Queue & History", id="history"):
                # -------------------------------------------------------------
                # Queue Status Header
                # -------------------------------------------------------------
                gr.Markdown("### 📋 Queue Manager")

                with gr.Row():
                    with gr.Column(scale=3):
                        queue_status_detailed = gr.Markdown(_get_detailed_queue_status())
                    with gr.Column(scale=1):
                        refresh_queue_btn = gr.Button("🔄 Refresh", variant="secondary")

                # -------------------------------------------------------------
                # Queue Controls
                # -------------------------------------------------------------
                with gr.Row():
                    cancel_current_btn = gr.Button("⏹️ Cancel Current", variant="secondary", scale=1)
                    clear_queue_btn = gr.Button("🗑️ Clear Queue", variant="secondary", scale=1)
                    pause_queue_btn = gr.Button("⏸️ Pause Queue", variant="secondary", scale=1)

                queue_action_status = gr.Markdown("")

                gr.Markdown("---")

                # -------------------------------------------------------------
                # Currently Running
                # -------------------------------------------------------------
                with gr.Accordion("🔄 Currently Running", open=True):
                    current_job_info = gr.Markdown(_get_current_job_info())
                    gr.Markdown("")

                # -------------------------------------------------------------
                # Pending Queue
                # -------------------------------------------------------------
                with gr.Accordion("⏳ Pending Jobs", open=False):

                    def get_pending_jobs():
                        """Get pending jobs as table data."""
                        if not _is_online_cached():
                            return [["ComfyUI offline", "—", "—"]]
                        queue = client.get_queue()
                        pending = queue.get("queue_pending", [])
                        if not pending:
                            return [["(No pending jobs)", "—", "—"]]
                        rows = []
                        for i, job in enumerate(pending[:10]):  # Max 10
                            prompt_id = job[1] if len(job) > 1 else "unknown"
                            rows.append(
                                [
                                    str(i + 1),
                                    prompt_id[:12] + "..." if len(prompt_id) > 12 else prompt_id,
                                    "Waiting",
                                ]
                            )
                        return rows

                    pending_table = gr.Dataframe(
                        headers=["#", "Job ID", "Status"],
                        value=get_pending_jobs(),
                        label="",
                        interactive=False,
                    )

                gr.Markdown("---")

                # -------------------------------------------------------------
                # Generation History
                # -------------------------------------------------------------
                gr.Markdown("### 📜 Generation History")
                gr.Markdown("*Recent generations from this session*")

                def get_enhanced_history():
                    """Get enhanced history with more details."""
                    if not _is_online_cached():
                        return [["ComfyUI offline", "—", "—", "—", "—"]]

                    history = client.get_history()
                    rows = []

                    for prompt_id, entry in list(history.items())[:25]:  # Last 25
                        status = entry.get("status", {})
                        outputs = entry.get("outputs", {})

                        # Count images and videos
                        image_count = 0
                        video_count = 0
                        for node_output in outputs.values():
                            if "images" in node_output:
                                image_count += len(node_output["images"])
                            if "gifs" in node_output or "videos" in node_output:
                                video_count += len(node_output.get("gifs", [])) + len(
                                    node_output.get("videos", [])
                                )

                        # Status with icon
                        if status.get("completed"):
                            status_str = "✅ Done"
                        elif status.get("status_str") == "error":
                            status_str = "❌ Error"
                        else:
                            status_str = "⏳ Running"

                        # Output summary
                        output_str = ""
                        if image_count > 0:
                            output_str += f"🖼️ {image_count}"
                        if video_count > 0:
                            output_str += f" 🎬 {video_count}"
                        if not output_str:
                            output_str = "—"

                        rows.append(
                            [
                                prompt_id[:10] + "...",
                                status_str,
                                output_str,
                                "View",
                            ]
                        )

                    return rows if rows else [["(No history yet)", "—", "—", "—"]]

                history_table = gr.Dataframe(
                    headers=["Job ID", "Status", "Outputs", "Action"],
                    value=get_enhanced_history(),
                    label="",
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    selected_history_id = gr.Textbox(
                        label="Selected Job",
                        placeholder="Click a row to select...",
                        interactive=False,
                        scale=2,
                    )
                    view_history_btn = gr.Button("👁️ View Details", scale=1)
                    reload_history_btn = gr.Button("📥 Load Settings", scale=1)

                # History details panel
                with gr.Accordion("📋 Job Details", open=False):
                    history_details = gr.Markdown("*Select a job to view details*")

                    with gr.Row():
                        history_preview = gr.Image(
                            label="Preview",
                            type="filepath",
                            height=256,
                            visible=False,
                        )

                # -------------------------------------------------------------
                # Event Handlers
                # -------------------------------------------------------------
                def refresh_all_queue():
                    """Refresh all queue-related displays."""
                    return (
                        _get_detailed_queue_status(),
                        _get_current_job_info(),
                        get_pending_jobs(),
                        get_enhanced_history(),
                        "",
                    )

                def clear_queue_action():
                    """Clear the queue and return status."""
                    client.clear_queue()
                    return "✅ Queue cleared", _get_detailed_queue_status()

                def cancel_current_action():
                    """Cancel current job."""
                    client.interrupt()
                    return "✅ Cancellation requested", _get_current_job_info()

                def on_history_select(evt: gr.SelectData, data):
                    """Handle history row selection."""
                    if evt.index is not None and len(data) > evt.index[0]:
                        full_id = data[evt.index[0]][0]
                        # The ID is truncated, need to find full ID
                        history = client.get_history()
                        for prompt_id in history:
                            if prompt_id.startswith(full_id.replace("...", "")):
                                return prompt_id
                    return ""

                def view_history_details(job_id: str):
                    """View details of a history job."""
                    if not job_id:
                        return "*Select a job first*", gr.update(visible=False)

                    history = client.get_history()
                    entry = history.get(job_id)

                    if not entry:
                        return f"*Job '{job_id}' not found*", gr.update(visible=False)

                    status = entry.get("status", {})
                    outputs = entry.get("outputs", {})
                    prompt = entry.get("prompt", {})

                    # Build details
                    details = f"""**Job ID:** `{job_id}`

**Status:** {"Completed" if status.get("completed") else "Error" if status.get("status_str") == "error" else "Running"}
"""

                    # Count outputs
                    images = []
                    for _node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img in node_output["images"]:
                                images.append(img)

                    details += f"\n**Images Generated:** {len(images)}"

                    # Try to extract prompt text
                    for _node_id, node_data in prompt.items():
                        if isinstance(node_data, dict):
                            inputs = node_data.get("inputs", {})
                            if "text" in inputs and len(str(inputs["text"])) > 5:
                                details += f"\n\n**Prompt:** {str(inputs['text'])[:200]}..."
                                break

                    # Get first image for preview if available
                    preview_update = gr.update(visible=False)
                    if images:
                        first_img = images[0]
                        f"{client.base_url}/view?filename={first_img.get('filename')}&subfolder={first_img.get('subfolder', '')}&type={first_img.get('type', 'output')}"
                        # Note: This would need actual image fetching logic
                        # For now, just show the info
                        details += f"\n\n*Preview: {first_img.get('filename')}*"

                    return details, preview_update

                refresh_queue_btn.click(
                    refresh_all_queue,
                    outputs=[
                        queue_status_detailed,
                        current_job_info,
                        pending_table,
                        history_table,
                        queue_action_status,
                    ],
                )

                clear_queue_btn.click(
                    clear_queue_action,
                    outputs=[queue_action_status, queue_status_detailed],
                )

                cancel_current_btn.click(
                    cancel_current_action,
                    outputs=[queue_action_status, current_job_info],
                )

                history_table.select(
                    on_history_select,
                    inputs=[history_table],
                    outputs=[selected_history_id],
                )

                view_history_btn.click(
                    view_history_details,
                    inputs=[selected_history_id],
                    outputs=[history_details, history_preview],
                )

                def reload_history_settings(job_id: str):
                    """Load settings from a history job back into the UI."""
                    if not job_id:
                        return "⚠️ Select a job first"
                    # Note: This is a placeholder - actual implementation would
                    # extract settings and return them to Image/Video tab inputs
                    return f"✅ Settings from job `{job_id[:16]}...` would be loaded to Image tab"

                reload_history_btn.click(
                    reload_history_settings,
                    inputs=[selected_history_id],
                    outputs=[queue_action_status],
                )

                def pause_queue_action():
                    """Toggle queue pause state (placeholder - ComfyUI doesn't have native pause)."""
                    # Note: ComfyUI doesn't have a native pause API
                    # This would need custom implementation or extension
                    return "⚠️ Pause not supported by ComfyUI (use Cancel Current instead)"

                pause_queue_btn.click(
                    pause_queue_action,
                    outputs=[queue_action_status],
                )

            # =================================================================
            # TAB: Workflows
            # =================================================================
            with gr.Tab("Workflows", id="workflows"):
                # -------------------------------------------------------------
                # Workflows Header
                # -------------------------------------------------------------
                gr.Markdown("### 🔧 Workflow Manager")
                gr.Markdown("*Browse, create, and manage ComfyUI workflows*")

                with gr.Tabs():
                    # ---------------------------------------------------------
                    # Sub-tab: Browse Workflows
                    # ---------------------------------------------------------
                    with gr.Tab("📚 Browse", id="workflows-browse"):

                        def get_workflow_list():
                            """Get list of available workflow templates."""
                            library = get_library()
                            templates = library.list_all()
                            return [
                                [
                                    t.id,
                                    t.name,
                                    t.category.value,
                                    t.description[:50] + "..."
                                    if len(t.description) > 50
                                    else t.description,
                                    f"{t.min_vram_gb}GB",
                                    len(t.parameters),
                                ]
                                for t in templates
                            ]

                        workflows_table = gr.Dataframe(
                            headers=["ID", "Name", "Category", "Description", "Min VRAM", "Params"],
                            value=get_workflow_list(),
                            label="Available Workflows",
                            interactive=False,
                            wrap=True,
                        )

                        with gr.Row():
                            selected_workflow_id = gr.Textbox(
                                label="Selected Workflow",
                                placeholder="Click a row to select...",
                                interactive=False,
                                scale=2,
                            )
                            view_workflow_btn = gr.Button("👁️ View Details", scale=1)
                            refresh_workflows_btn = gr.Button("🔄 Refresh", scale=1)

                        # Workflow details panel
                        with gr.Accordion(
                            "Workflow Details", open=False
                        ):
                            workflow_details = gr.Markdown("*Select a workflow to view details*")

                            workflow_params_table = gr.Dataframe(
                                headers=["Parameter", "Type", "Required", "Default", "Description"],
                                value=[],
                                label="Parameters",
                                interactive=False,
                            )

                        def on_workflow_select(evt: gr.SelectData, data):
                            """Handle workflow row selection."""
                            if evt.index is not None and len(data) > evt.index[0]:
                                return data[evt.index[0]][0]  # Return the ID
                            return ""

                        def view_workflow_details(workflow_id: str):
                            """Get detailed info about a workflow."""
                            if not workflow_id:
                                return "*Select a workflow first*", []

                            library = get_library()
                            template = library.get(workflow_id)

                            if not template:
                                return f"*Workflow '{workflow_id}' not found*", []

                            # Build details markdown
                            details = f"""
**{template.name}** (`{template.id}`)

**Category:** {template.category.value}
**Description:** {template.description}
**Minimum VRAM:** {template.min_vram_gb} GB
**Tags:** {", ".join(template.tags) if template.tags else "None"}

**Presets Available:** {len(template.presets)}
"""
                            if template.presets:
                                details += "\n" + "\n".join(
                                    [
                                        f"- `{k}`: {v.description}"
                                        for k, v in template.presets.items()
                                    ]
                                )

                            # Build params table
                            params_data = []
                            for name, param in template.parameters.items():
                                params_data.append(
                                    [
                                        name,
                                        param.type.value,
                                        "✓" if param.required else "",
                                        str(param.default) if param.default is not None else "—",
                                        param.description or param.label or "—",
                                    ]
                                )

                            return details, params_data

                        workflows_table.select(
                            on_workflow_select,
                            inputs=[workflows_table],
                            outputs=[selected_workflow_id],
                        )

                        view_workflow_btn.click(
                            view_workflow_details,
                            inputs=[selected_workflow_id],
                            outputs=[workflow_details, workflow_params_table],
                        )

                        refresh_workflows_btn.click(
                            get_workflow_list,
                            outputs=[workflows_table],
                        )

                    # ---------------------------------------------------------
                    # Sub-tab: Generation Presets
                    # ---------------------------------------------------------
                    with gr.Tab("⚡ Presets", id="workflows-presets"):
                        gr.Markdown("**Generation Presets** - Quick settings for common use cases")

                        def get_presets_data():
                            """Get all generation presets."""
                            return [
                                [
                                    name,
                                    info.get("description", ""),
                                    f"{info.get('width', 0)}x{info.get('height', 0)}",
                                    str(info.get("steps", 0)),
                                    str(info.get("cfg", 0)),
                                ]
                                for name, info in GENERATION_PRESETS.items()
                            ]

                        gr.Dataframe(
                            headers=["Preset", "Description", "Resolution", "Steps", "CFG"],
                            value=get_presets_data(),
                            label="",
                            interactive=False,
                        )

                        gr.Markdown("""
**Usage Tips:**
- **Draft**: Quick previews, testing prompts
- **Fast**: Good balance for iteration
- **Quality**: Final renders, detailed work
- **HD**: Maximum quality, more VRAM needed
- **Portrait/Landscape/Cinematic**: Aspect ratio optimized
                        """)

                        # Video presets
                        gr.Markdown("---")
                        gr.Markdown("**Video Presets** - Settings for video generation")

                        def get_video_presets_data():
                            """Get all video presets."""
                            return [
                                [
                                    name,
                                    f"{preset.width}x{preset.height}",
                                    str(preset.frames),
                                    str(preset.fps),
                                    str(preset.steps),
                                    preset.model.value
                                    if hasattr(preset.model, "value")
                                    else str(preset.model),
                                ]
                                for name, preset in VIDEO_PRESETS.items()
                            ]

                        gr.Dataframe(
                            headers=["Preset", "Resolution", "Frames", "FPS", "Steps", "Model"],
                            value=get_video_presets_data(),
                            label="",
                            interactive=False,
                        )

                    # ---------------------------------------------------------
                    # Sub-tab: Import Workflow
                    # ---------------------------------------------------------
                    with gr.Tab("📥 Import", id="workflows-import"):
                        gr.Markdown("**Import a ComfyUI Workflow**")
                        gr.Markdown(
                            "*Paste a workflow JSON exported from ComfyUI to register it as a template*"
                        )

                        with gr.Row():
                            import_workflow_name = gr.Textbox(
                                label="Workflow Name",
                                placeholder="My Custom Workflow",
                                scale=2,
                            )
                            import_workflow_category = gr.Dropdown(
                                choices=[c.value for c in WorkflowCategory],
                                value=WorkflowCategory.TEXT_TO_IMAGE.value,
                                label="Category",
                                scale=1,
                            )

                        import_workflow_desc = gr.Textbox(
                            label="Description",
                            placeholder="What does this workflow do?",
                            lines=2,
                        )

                        gr.Markdown(
                            "*Paste your ComfyUI workflow JSON below (exported from ComfyUI API format)*"
                        )
                        import_workflow_json = gr.Code(
                            label="Workflow JSON",
                            language="json",
                            lines=15,
                        )

                        with gr.Row():
                            import_workflow_btn = gr.Button(
                                "📥 Import Workflow", variant="primary", scale=2
                            )
                            validate_workflow_btn = gr.Button("✅ Validate Only", scale=1)
                            check_deps_btn = gr.Button("🔍 Check Dependencies", scale=1)

                        import_status = gr.Markdown("")

                        # Dependency details accordion (hidden by default)
                        with gr.Accordion("📦 Dependency Details", open=False):
                            deps_status = gr.Markdown(
                                "*Click 'Check Dependencies' to analyze the workflow*"
                            )

                        def validate_workflow_json(json_str: str):
                            """Validate workflow JSON."""
                            import json as json_module

                            try:
                                if not json_str.strip():
                                    return "⚠️ Please paste a workflow JSON"

                                workflow = json_module.loads(json_str)

                                if not isinstance(workflow, dict):
                                    return "❌ Invalid workflow: must be a JSON object"

                                # Check for nodes
                                node_count = len(workflow)
                                if node_count == 0:
                                    return "❌ Invalid workflow: no nodes found"

                                # Check for common node types
                                class_types = set()
                                for _node_id, node in workflow.items():
                                    if isinstance(node, dict) and "class_type" in node:
                                        class_types.add(node["class_type"])

                                # Identify workflow type
                                has_ksampler = any("KSampler" in ct for ct in class_types)
                                has_loader = any("Loader" in ct for ct in class_types)
                                has_clip = any("CLIP" in ct for ct in class_types)
                                has_vae = any("VAE" in ct for ct in class_types)

                                status_parts = [f"✅ Valid JSON with {node_count} nodes"]

                                if has_ksampler:
                                    status_parts.append("✓ Has KSampler")
                                if has_loader:
                                    status_parts.append("✓ Has Model Loader")
                                if has_clip:
                                    status_parts.append("✓ Has CLIP nodes")
                                if has_vae:
                                    status_parts.append("✓ Has VAE nodes")

                                return "\n".join(status_parts)

                            except json_module.JSONDecodeError as e:
                                return f"❌ Invalid JSON: {str(e)[:100]}"
                            except Exception as e:
                                return f"❌ Error: {str(e)[:100]}"

                        def check_workflow_dependencies(json_str: str):
                            """Check if all required nodes are installed in ComfyUI."""
                            import json as json_module

                            if not json_str.strip():
                                return "⚠️ Please paste a workflow JSON first"

                            try:
                                workflow = json_module.loads(json_str)

                                if not isinstance(workflow, dict) or len(workflow) == 0:
                                    return "❌ Invalid workflow JSON"

                                # Check dependencies using the client
                                deps = client.check_workflow_dependencies(workflow)

                                if not deps["installed"] and not deps["missing"]:
                                    return "⚠️ Could not connect to ComfyUI to check dependencies. Make sure ComfyUI is running."

                                # Build the status report
                                lines = []

                                if deps["all_installed"]:
                                    lines.append("## ✅ All Dependencies Installed!")
                                    lines.append(
                                        f"This workflow uses **{deps['unique_types']}** node types, all of which are installed."
                                    )
                                else:
                                    lines.append("## ⚠️ Missing Dependencies")
                                    lines.append(
                                        f"This workflow requires **{len(deps['missing'])}** custom node(s) that are not installed:"
                                    )
                                    lines.append("")

                                    for node_type in deps["missing"]:
                                        # Try to provide helpful info about where to get the node
                                        node_info = _get_node_source_hint(node_type)
                                        lines.append(f"- **{node_type}** {node_info}")

                                    lines.append("")
                                    lines.append("### How to install missing nodes:")
                                    lines.append(
                                        "1. Open **ComfyUI Manager** (press 'M' in ComfyUI)"
                                    )
                                    lines.append("2. Search for the missing node name")
                                    lines.append("3. Click **Install** and restart ComfyUI")

                                lines.append("")
                                lines.append("---")
                                lines.append(
                                    f"**Summary:** {len(deps['installed'])} installed, {len(deps['missing'])} missing, {deps['total_nodes']} total nodes"
                                )

                                if deps["installed"]:
                                    lines.append("")
                                    lines.append(
                                        "<details><summary>Installed nodes (click to expand)</summary>"
                                    )
                                    lines.append("")
                                    for node_type in deps["installed"]:
                                        lines.append(f"- ✓ {node_type}")
                                    lines.append("</details>")

                                return "\n".join(lines)

                            except json_module.JSONDecodeError as e:
                                return f"❌ Invalid JSON: {str(e)[:100]}"
                            except Exception as e:
                                return f"❌ Error checking dependencies: {str(e)[:100]}"

                        def import_workflow(
                            name: str, category: str, description: str, json_str: str
                        ):
                            """Import a workflow as a template."""
                            import json as json_module

                            if not name.strip():
                                return "❌ Please provide a workflow name"

                            if not json_str.strip():
                                return "❌ Please paste a workflow JSON"

                            try:
                                workflow = json_module.loads(json_str)

                                if not isinstance(workflow, dict) or len(workflow) == 0:
                                    return "❌ Invalid workflow JSON"

                                # Create a simple template
                                # Generate ID from name
                                workflow_id = name.lower().replace(" ", "_").replace("-", "_")

                                # Detect parameters (look for common input patterns)
                                parameters = {}

                                for node_id, node in workflow.items():
                                    if not isinstance(node, dict):
                                        continue

                                    class_type = node.get("class_type", "")
                                    inputs = node.get("inputs", {})

                                    # Auto-detect prompt inputs
                                    if "CLIPTextEncode" in class_type:
                                        if "text" in inputs:
                                            param_name = (
                                                "prompt"
                                                if "positive" not in parameters
                                                else "negative"
                                            )
                                            parameters[param_name] = ParameterDef(
                                                type=ParameterType.STRING,
                                                node_id=node_id,
                                                input_name="text",
                                                label=param_name.title(),
                                                description=f"Text input for {class_type}",
                                            )

                                    # Auto-detect seed
                                    if "KSampler" in class_type:
                                        if "seed" in inputs:
                                            parameters["seed"] = ParameterDef(
                                                type=ParameterType.INT,
                                                node_id=node_id,
                                                input_name="seed",
                                                default=-1,
                                                label="Seed",
                                                description="Random seed (-1 for random)",
                                            )
                                        if "steps" in inputs:
                                            parameters["steps"] = ParameterDef(
                                                type=ParameterType.INT,
                                                node_id=node_id,
                                                input_name="steps",
                                                default=inputs.get("steps", 20),
                                                min=1,
                                                max=100,
                                                label="Steps",
                                            )
                                        if "cfg" in inputs:
                                            parameters["cfg"] = ParameterDef(
                                                type=ParameterType.FLOAT,
                                                node_id=node_id,
                                                input_name="cfg",
                                                default=inputs.get("cfg", 7.0),
                                                min=1.0,
                                                max=20.0,
                                                label="CFG Scale",
                                            )

                                # Create template
                                template = WorkflowTemplate(
                                    id=workflow_id,
                                    name=name,
                                    description=description or f"Imported workflow: {name}",
                                    category=WorkflowCategory(category),
                                    workflow=workflow,
                                    parameters=parameters,
                                    min_vram_gb=8,  # Default estimate
                                    tags=["imported", "custom"],
                                )

                                # Register it
                                library = get_library()
                                library.add(template)

                                return f"""✅ **Workflow Imported Successfully!**

**ID:** `{workflow_id}`
**Name:** {name}
**Category:** {category}
**Parameters detected:** {len(parameters)}
{chr(10).join([f"  - {k}: {v.type.value}" for k, v in parameters.items()])}

*The workflow is now available in the Browse tab and can be used for generation.*
"""

                            except Exception as e:
                                return f"❌ Import failed: {str(e)[:200]}"

                        validate_workflow_btn.click(
                            validate_workflow_json,
                            inputs=[import_workflow_json],
                            outputs=[import_status],
                        )

                        import_workflow_btn.click(
                            import_workflow,
                            inputs=[
                                import_workflow_name,
                                import_workflow_category,
                                import_workflow_desc,
                                import_workflow_json,
                            ],
                            outputs=[import_status],
                        )

                        check_deps_btn.click(
                            check_workflow_dependencies,
                            inputs=[import_workflow_json],
                            outputs=[deps_status],
                        )

                    # ---------------------------------------------------------
                    # Sub-tab: Create Workflow
                    # ---------------------------------------------------------
                    with gr.Tab("✨ Create", id="workflows-create"):
                        gr.Markdown("**Create a Simple Workflow**")
                        gr.Markdown("*Build a basic txt2img workflow with custom parameters*")

                        with gr.Row():
                            create_name = gr.Textbox(
                                label="Workflow Name",
                                placeholder="My txt2img Workflow",
                                scale=2,
                            )
                            create_base = gr.Dropdown(
                                choices=["txt2img_standard", "txt2img_hires"],
                                value="txt2img_standard",
                                label="Base Template",
                                scale=1,
                            )

                        create_desc = gr.Textbox(
                            label="Description",
                            placeholder="Describe what makes this workflow special...",
                            lines=2,
                        )

                        gr.Markdown("**Default Settings**")

                        with gr.Row():
                            create_width = gr.Slider(
                                256, 2048, value=1024, step=64, label="Default Width"
                            )
                            create_height = gr.Slider(
                                256, 2048, value=1024, step=64, label="Default Height"
                            )

                        with gr.Row():
                            create_steps = gr.Slider(
                                1, 100, value=25, step=1, label="Default Steps"
                            )
                            create_cfg = gr.Slider(1, 20, value=7.0, step=0.5, label="Default CFG")

                        create_negative = gr.Textbox(
                            label="Default Negative Prompt",
                            value="ugly, blurry, low quality, distorted, deformed",
                            lines=2,
                        )

                        create_btn = gr.Button("✨ Create Workflow", variant="primary")
                        create_status = gr.Markdown("")

                        def create_custom_workflow(
                            name, base, desc, width, height, steps, cfg, negative
                        ):
                            """Create a custom workflow based on a template."""
                            if not name.strip():
                                return "❌ Please provide a workflow name"

                            try:
                                library = get_library()
                                base_template = library.get(base)

                                if not base_template:
                                    return f"❌ Base template '{base}' not found"

                                # Create new template based on existing
                                import copy

                                workflow_id = name.lower().replace(" ", "_").replace("-", "_")

                                # Clone the base workflow
                                new_workflow = copy.deepcopy(base_template.workflow)
                                new_params = copy.deepcopy(base_template.parameters)

                                # Update defaults
                                if "width" in new_params:
                                    new_params["width"].default = int(width)
                                if "height" in new_params:
                                    new_params["height"].default = int(height)
                                if "steps" in new_params:
                                    new_params["steps"].default = int(steps)
                                if "cfg" in new_params:
                                    new_params["cfg"].default = float(cfg)
                                if "negative" in new_params:
                                    new_params["negative"].default = negative

                                template = WorkflowTemplate(
                                    id=workflow_id,
                                    name=name,
                                    description=desc or f"Custom workflow based on {base}",
                                    category=base_template.category,
                                    workflow=new_workflow,
                                    parameters=new_params,
                                    presets=base_template.presets,
                                    min_vram_gb=base_template.min_vram_gb,
                                    tags=["custom", f"based-on-{base}"],
                                )

                                library.add(template)

                                return f"""✅ **Workflow Created!**

**ID:** `{workflow_id}`
**Name:** {name}
**Based on:** {base}
**Defaults:** {width}x{height}, {steps} steps, CFG {cfg}

*Your workflow is now available in the Browse tab!*
"""

                            except Exception as e:
                                return f"❌ Creation failed: {str(e)[:200]}"

                        create_btn.click(
                            create_custom_workflow,
                            inputs=[
                                create_name,
                                create_base,
                                create_desc,
                                create_width,
                                create_height,
                                create_steps,
                                create_cfg,
                                create_negative,
                            ],
                            outputs=[create_status],
                        )

            # =================================================================
            # TAB: Models
            # =================================================================
            with gr.Tab("Models", id="models"):
                # -------------------------------------------------------------
                # Model Browser Header
                # -------------------------------------------------------------
                with gr.Row():
                    gr.Markdown("### 🎨 Model Browser")
                    models_status = gr.Markdown("", elem_id="models-status")

                # -------------------------------------------------------------
                # Search and Filter
                # -------------------------------------------------------------
                with gr.Row():
                    model_search = gr.Textbox(
                        label="🔍 Search Models",
                        placeholder="Type to filter models...",
                        scale=3,
                    )
                    model_type_filter = gr.Dropdown(
                        choices=["All Types", "Checkpoints", "LoRAs", "VAEs", "Motion Models"],
                        value="All Types",
                        label="Filter by Type",
                        scale=1,
                    )
                    refresh_models_btn = gr.Button("🔄 Refresh", scale=1)

                # -------------------------------------------------------------
                # Checkpoints Section
                # -------------------------------------------------------------
                with gr.Accordion("📦 Checkpoints (Base Models)", open=True):

                    def get_checkpoints_data():
                        """Get checkpoints with metadata."""
                        if not _is_online_cached():
                            return [["ComfyUI offline", "—", "—"]]
                        models = client.get_checkpoints() or []
                        if not models:
                            return [["(No checkpoints found)", "—", "—"]]
                        # Return with placeholder metadata
                        return [[m, _detect_model_type(m), "—"] for m in models]

                    def _detect_model_type(model_name: str) -> str:
                        """Detect model type from filename."""
                        name_lower = model_name.lower()
                        if "xl" in name_lower or "sdxl" in name_lower:
                            return "SDXL"
                        elif "flux" in name_lower:
                            return "Flux"
                        elif "sd3" in name_lower:
                            return "SD3"
                        elif "pony" in name_lower:
                            return "Pony"
                        elif "1.5" in name_lower or "v1-5" in name_lower or "sd15" in name_lower:
                            return "SD1.5"
                        elif "2.1" in name_lower or "v2-1" in name_lower:
                            return "SD2.1"
                        elif "anime" in name_lower:
                            return "Anime"
                        elif "realistic" in name_lower or "real" in name_lower:
                            return "Realistic"
                        return "Unknown"

                    checkpoints_table = gr.Dataframe(
                        headers=["Model Name", "Type", "Size"],
                        value=get_checkpoints_data(),
                        label="",
                        interactive=False,
                        wrap=True,
                    )

                    with gr.Row():
                        selected_checkpoint = gr.Textbox(
                            label="Selected Checkpoint",
                            placeholder="Click a row to select...",
                            interactive=False,
                            scale=3,
                        )
                        gr.Button("📋 Copy to Clipboard", scale=1)

                # -------------------------------------------------------------
                # LoRA Section
                # -------------------------------------------------------------
                with gr.Accordion("🎭 LoRA Models", open=False):

                    def get_loras_data():
                        """Get LoRAs with metadata."""
                        if not _is_online_cached():
                            return [["ComfyUI offline", "—"]]
                        loras = client.get_loras() or []
                        if not loras:
                            return [["(No LoRAs found)", "—"]]
                        return [[l, _detect_lora_type(l)] for l in loras]

                    def _detect_lora_type(lora_name: str) -> str:
                        """Detect LoRA type from filename."""
                        name_lower = lora_name.lower()
                        if "style" in name_lower:
                            return "Style"
                        elif "character" in name_lower or "char" in name_lower:
                            return "Character"
                        elif "concept" in name_lower:
                            return "Concept"
                        elif "pose" in name_lower:
                            return "Pose"
                        elif "detail" in name_lower:
                            return "Detail"
                        elif "lcm" in name_lower:
                            return "LCM"
                        return "General"

                    loras_table = gr.Dataframe(
                        headers=["LoRA Name", "Category"],
                        value=get_loras_data(),
                        label="",
                        interactive=False,
                        wrap=True,
                    )

                    gr.Markdown(
                        "*Select a LoRA to see details. LoRAs modify the base model's output.*"
                    )

                # -------------------------------------------------------------
                # Motion Models Section (for Video)
                # -------------------------------------------------------------
                with gr.Accordion("🎬 Motion Models (AnimateDiff)", open=False):

                    def get_motion_data():
                        """Get motion models."""
                        if not _is_online_cached():
                            return [["ComfyUI offline", "—"]]
                        models = client.get_motion_models() or []
                        if not models:
                            return [["(No motion models found)", "—"]]
                        return [[m, _detect_motion_version(m)] for m in models]

                    def _detect_motion_version(model_name: str) -> str:
                        """Detect motion model version."""
                        name_lower = model_name.lower()
                        if "v3" in name_lower:
                            return "v3 (Latest)"
                        elif "v2" in name_lower:
                            return "v2"
                        elif "v1" in name_lower:
                            return "v1"
                        elif "sdxl" in name_lower:
                            return "SDXL"
                        return "Standard"

                    motion_table = gr.Dataframe(
                        headers=["Motion Model", "Version"],
                        value=get_motion_data(),
                        label="",
                        interactive=False,
                        wrap=True,
                    )

                    gr.Markdown(
                        "*Motion models control animation style in video generation.*"
                    )

                # -------------------------------------------------------------
                # Samplers & Schedulers Section
                # -------------------------------------------------------------
                with gr.Accordion("⚙️ Samplers & Schedulers", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Samplers** control the denoising algorithm")
                            samplers_list = gr.Dataframe(
                                headers=["Sampler", "Speed", "Quality"],
                                value=_get_samplers_with_info(),
                                label="",
                                interactive=False,
                            )

                        with gr.Column():
                            gr.Markdown("**Schedulers** control the noise schedule")
                            schedulers_list = gr.Dataframe(
                                headers=["Scheduler", "Best For"],
                                value=_get_schedulers_with_info(),
                                label="",
                                interactive=False,
                            )

                    gr.Markdown("""
**Quick Guide:**
- **Fast generation**: `euler_ancestral` + `normal` (creative, varied)
- **Quality focus**: `dpmpp_2m_sde` + `karras` (detailed, consistent)
- **Balanced**: `dpmpp_2m` + `karras` (good default)
                    """)

                # -------------------------------------------------------------
                # Model Summary Stats
                # -------------------------------------------------------------
                gr.Markdown("---")
                with gr.Row():

                    def get_model_summary():
                        """Get summary of all models."""
                        if not _is_online_cached():
                            return "**ComfyUI Offline** - Connect to see model counts"

                        checkpoints = len(client.get_checkpoints() or [])
                        loras = len(client.get_loras() or [])
                        motion = len(client.get_motion_models() or [])
                        samplers = len(client.get_samplers() or [])
                        schedulers = len(client.get_schedulers() or [])

                        return f"""**Model Summary:** {checkpoints} checkpoints | {loras} LoRAs | {motion} motion models | {samplers} samplers | {schedulers} schedulers"""

                    model_summary = gr.Markdown(get_model_summary())

                # -------------------------------------------------------------
                # Event Handlers
                # -------------------------------------------------------------

                def filter_models(search_text: str, type_filter: str):
                    """Filter models based on search and type."""
                    checkpoints = get_checkpoints_data()
                    loras = get_loras_data()
                    motion = get_motion_data()

                    if search_text:
                        search_lower = search_text.lower()
                        checkpoints = [c for c in checkpoints if search_lower in c[0].lower()]
                        loras = [l for l in loras if search_lower in l[0].lower()]
                        motion = [m for m in motion if search_lower in m[0].lower()]

                    return checkpoints, loras, motion

                def refresh_all_models():
                    """Refresh all model lists."""
                    return (
                        get_checkpoints_data(),
                        get_loras_data(),
                        get_motion_data(),
                        _get_samplers_with_info(),
                        _get_schedulers_with_info(),
                        get_model_summary(),
                        "✅ Models refreshed",
                    )

                def on_checkpoint_select(evt: gr.SelectData, data):
                    """Handle checkpoint row selection."""
                    if evt.index is not None and len(data) > evt.index[0]:
                        return data[evt.index[0]][0]
                    return ""

                # Wire up events
                refresh_models_btn.click(
                    refresh_all_models,
                    outputs=[
                        checkpoints_table,
                        loras_table,
                        motion_table,
                        samplers_list,
                        schedulers_list,
                        model_summary,
                        models_status,
                    ],
                )

                model_search.change(
                    filter_models,
                    inputs=[model_search, model_type_filter],
                    outputs=[checkpoints_table, loras_table, motion_table],
                )

                model_type_filter.change(
                    filter_models,
                    inputs=[model_search, model_type_filter],
                    outputs=[checkpoints_table, loras_table, motion_table],
                )

                checkpoints_table.select(
                    on_checkpoint_select,
                    inputs=[checkpoints_table],
                    outputs=[selected_checkpoint],
                )

            # =================================================================
            # TAB: Settings
            # =================================================================
            with gr.Tab("Settings", id="settings"):
                # -------------------------------------------------------------
                # Connection & Status Section
                # -------------------------------------------------------------
                gr.Markdown("### 🔌 Connection & Status")

                with gr.Row():
                    with gr.Column(scale=2):
                        # Saved instances dropdown
                        saved_instances = gr.Dropdown(
                            choices=["Local (localhost:8188)", "Add new..."],
                            value="Local (localhost:8188)",
                            label="ComfyUI Instance",
                            info="Select or add ComfyUI servers",
                            allow_custom_value=True,
                        )
                    with gr.Column(scale=1):
                        connection_status = gr.Markdown("⚪ Not tested")

                with gr.Row():
                    comfy_url = gr.Textbox(
                        label="ComfyUI URL",
                        value=settings.comfyui.url,
                        info="URL of your ComfyUI instance",
                        scale=3,
                    )
                    test_connection_btn = gr.Button("🔍 Test Connection", scale=1)

                # Diagnostics output
                connection_diagnostics = gr.Markdown("", visible=False)

                def test_connection_detailed(url: str):
                    """Test connection with detailed diagnostics."""
                    import time

                    import httpx

                    diagnostics = []
                    diagnostics.append(f"**Testing:** `{url}`\n")

                    # Test 1: Basic connectivity
                    start = time.time()
                    try:
                        response = httpx.get(f"{url.rstrip('/')}/system_stats", timeout=5.0)
                        latency = (time.time() - start) * 1000

                        if response.status_code == 200:
                            diagnostics.append(f"✅ **Connection:** Success ({latency:.0f}ms)")
                            stats = response.json()

                            # GPU info
                            devices = stats.get("devices", [])
                            if devices:
                                gpu = devices[0]
                                gpu_name = gpu.get("name", "Unknown")
                                vram_total = gpu.get("vram_total", 0) / (1024**3)
                                vram_free = gpu.get("vram_free", 0) / (1024**3)
                                vram_used = vram_total - vram_free
                                diagnostics.append(f"✅ **GPU:** {gpu_name}")
                                diagnostics.append(
                                    f"✅ **VRAM:** {vram_used:.1f} / {vram_total:.1f} GB ({(vram_used / vram_total) * 100:.0f}% used)"
                                )

                            status = "🟢 **Connected**"
                        else:
                            diagnostics.append(f"⚠️ **Connection:** HTTP {response.status_code}")
                            status = "🟡 **Partial**"

                    except httpx.ConnectError:
                        diagnostics.append(
                            "❌ **Connection:** Failed - ComfyUI not running or wrong URL"
                        )
                        status = "🔴 **Offline**"
                    except httpx.TimeoutException:
                        diagnostics.append("❌ **Connection:** Timeout - Server not responding")
                        status = "🔴 **Timeout**"
                    except Exception as e:
                        diagnostics.append(f"❌ **Connection:** Error - {str(e)[:50]}")
                        status = "🔴 **Error**"

                    # Test 2: Queue endpoint
                    try:
                        response = httpx.get(f"{url.rstrip('/')}/queue", timeout=3.0)
                        if response.status_code == 200:
                            queue = response.json()
                            running = len(queue.get("queue_running", []))
                            pending = len(queue.get("queue_pending", []))
                            diagnostics.append(
                                f"✅ **Queue:** {running} running, {pending} pending"
                            )
                    except:
                        diagnostics.append("⚠️ **Queue:** Could not fetch")

                    # Test 3: Models endpoint
                    try:
                        response = httpx.get(
                            f"{url.rstrip('/')}/object_info/CheckpointLoaderSimple", timeout=3.0
                        )
                        if response.status_code == 200:
                            data = response.json()
                            checkpoints = (
                                data.get("CheckpointLoaderSimple", {})
                                .get("input", {})
                                .get("required", {})
                                .get("ckpt_name", [[]])[0]
                            )
                            diagnostics.append(
                                f"✅ **Models:** {len(checkpoints)} checkpoints available"
                            )
                    except:
                        diagnostics.append("⚠️ **Models:** Could not fetch")

                    return status, gr.update(value="\n".join(diagnostics), visible=True)

                test_connection_btn.click(
                    test_connection_detailed,
                    inputs=[comfy_url],
                    outputs=[connection_status, connection_diagnostics],
                )

                # Instance management
                def handle_instance_change(selected, current_url):
                    """Handle instance dropdown change."""
                    if selected == "Add new...":
                        return current_url  # Keep current, user will edit
                    elif selected.startswith("Local"):
                        return "http://localhost:8188"
                    else:
                        # Extract URL from "Name (url)" format
                        if "(" in selected and ")" in selected:
                            return selected.split("(")[1].rstrip(")")
                        return current_url

                saved_instances.change(
                    handle_instance_change,
                    inputs=[saved_instances, comfy_url],
                    outputs=[comfy_url],
                )

                # Apply URL change to client
                def apply_url_change(url: str):
                    """Apply URL change to the client."""
                    settings.comfyui.url = url.rstrip("/")
                    client.base_url = url.rstrip("/")
                    return f"✅ URL updated to: `{url}`"

                apply_url_btn = gr.Button("💾 Apply URL Change", variant="primary")
                url_change_status = gr.Markdown("")

                apply_url_btn.click(
                    apply_url_change,
                    inputs=[comfy_url],
                    outputs=[url_change_status],
                )

                gr.Markdown("---")

                # -------------------------------------------------------------
                # Timeout Settings
                # -------------------------------------------------------------
                gr.Markdown("### ⏱️ Timeout Settings")

                with gr.Row():
                    timeout_connect = gr.Slider(
                        1,
                        30,
                        value=settings.comfyui.timeout_connect,
                        step=1,
                        label="Connect Timeout (s)",
                        info="Time to wait for initial connection",
                    )
                    timeout_read = gr.Slider(
                        5,
                        120,
                        value=settings.comfyui.timeout_read,
                        step=5,
                        label="Read Timeout (s)",
                        info="Time to wait for response",
                    )

                with gr.Row():
                    timeout_image = gr.Slider(
                        30,
                        300,
                        value=settings.comfyui.timeout_image,
                        step=10,
                        label="Image Generation Timeout (s)",
                        info="Max time for image generation",
                    )
                    timeout_video = gr.Slider(
                        60,
                        600,
                        value=settings.comfyui.timeout_video,
                        step=30,
                        label="Video Generation Timeout (s)",
                        info="Max time for video generation",
                    )

                def apply_timeout_settings(t_connect, t_read, t_image, t_video):
                    """Apply timeout settings."""
                    settings.comfyui.timeout_connect = t_connect
                    settings.comfyui.timeout_read = t_read
                    settings.comfyui.timeout_image = t_image
                    settings.comfyui.timeout_video = t_video
                    return "✅ Timeout settings updated"

                apply_timeouts_btn = gr.Button("💾 Apply Timeout Settings")
                timeout_status = gr.Markdown("")

                apply_timeouts_btn.click(
                    apply_timeout_settings,
                    inputs=[timeout_connect, timeout_read, timeout_image, timeout_video],
                    outputs=[timeout_status],
                )

                gr.Markdown("---")

                # -------------------------------------------------------------
                # Auto-Reconnect Settings
                # -------------------------------------------------------------
                gr.Markdown("### 🔄 Auto-Reconnect")

                with gr.Row():
                    gr.Checkbox(
                        label="Enable Auto-Reconnect",
                        value=True,
                        info="Automatically try to reconnect when connection is lost",
                    )
                    gr.Slider(
                        5,
                        60,
                        value=15,
                        step=5,
                        label="Reconnect Interval (s)",
                        info="Time between reconnection attempts",
                    )

                gr.Markdown(
                    "*Auto-reconnect will attempt to restore connection if ComfyUI becomes unavailable*"
                )

                gr.Markdown("---")

                # -------------------------------------------------------------
                # System Info Section
                # -------------------------------------------------------------
                gr.Markdown("### 💻 System Info")

                def get_system_info():
                    if not client.is_online():
                        return "ComfyUI is offline - start ComfyUI and test connection above"

                    stats = client.get_system_stats()
                    if not stats:
                        return "Could not fetch system stats"

                    info = []
                    for device in stats.get("devices", []):
                        info.append(f"**{device.get('name', 'GPU')}**")
                        vram_total = device.get("vram_total", 0) / (1024**3)
                        vram_free = device.get("vram_free", 0) / (1024**3)
                        info.append(f"- VRAM: {vram_total - vram_free:.1f} / {vram_total:.1f} GB")
                        info.append(
                            f"- Torch VRAM: {device.get('torch_vram_total', 0) / (1024**3):.1f} GB"
                        )

                    return "\n".join(info) if info else "No GPU info available"

                system_info = gr.Markdown(get_system_info())
                refresh_system_btn = gr.Button("🔄 Refresh System Info")
                refresh_system_btn.click(get_system_info, outputs=[system_info])

                gr.Markdown("---")

                # -------------------------------------------------------------
                # Help Panel
                # -------------------------------------------------------------
                gr.Markdown("### ❓ Help & Documentation")

                with gr.Accordion("Quick Help", open=False):
                    gr.Markdown(format_help_list())

                    help_topic = gr.Dropdown(
                        choices=list_topics(),
                        label="Select Topic",
                        info="Choose a topic to learn more",
                    )

                    help_detail = gr.Markdown("")

                    def show_help(topic):
                        if topic:
                            return get_help(topic)
                        return "Select a topic above"

                    help_topic.change(show_help, inputs=[help_topic], outputs=[help_detail])

        # Global refresh handler - updates status and model dropdowns with proper states
    # Clear cache after UI construction - subsequent calls should check live status
    _cached_online_status = None
    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("GRADIO_SERVER_PORT", 7870))
    app = create_ui()
    # Gradio 6.0: pass theme/css to launch()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=True,
        theme=_theme,
        css=_custom_css,
    )
