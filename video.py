"""
Comfy Headless - Video Generation Module
========================================

v2.5.0: Added LTX-Video 2, Hunyuan 1.5, Wan 2.1/2.2

Complete video generation support for multiple models:
- AnimateDiff v2/v3/Lightning
- Stable Video Diffusion (SVD/SVD-XT)
- CogVideoX
- Hunyuan Video / Hunyuan Video 1.5
- LTX-Video 2 (Lightricks) - NEW
- Wan 2.1/2.2 (Alibaba) - NEW
- Mochi (Genmo) - EXPERIMENTAL

Makes video generation accessible through simple presets and settings.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    # Enums
    "VideoModel",
    "VideoFormat",
    "MotionStyle",
    # Data classes
    "VideoSettings",
    "VideoModelInfo",
    # Presets
    "VIDEO_PRESETS",
    "VIDEO_MODEL_INFO",
    # Builder
    "VideoWorkflowBuilder",
    "get_video_builder",
    # Convenience functions
    "build_video_workflow",
    "get_video_preset",
    "list_video_presets",
    "list_video_models",
    "get_recommended_preset",
]


# =============================================================================
# VIDEO ENUMS
# =============================================================================


class VideoModel(str, Enum):
    """Available video generation models."""

    # AnimateDiff family
    ANIMATEDIFF_V2 = "animatediff_v2"
    ANIMATEDIFF_V3 = "animatediff_v3"
    ANIMATEDIFF_LIGHTNING = "animatediff_lightning"

    # Stable Video Diffusion
    SVD = "svd"
    SVD_XT = "svd_xt"

    # CogVideoX
    COGVIDEOX = "cogvideox"

    # Hunyuan Video (Tencent)
    HUNYUAN = "hunyuan"
    HUNYUAN_15 = "hunyuan_15"  # v2.5.0: Hunyuan 1.5 with new encoders
    HUNYUAN_15_FAST = "hunyuan_15_fast"  # v2.5.0: 6-step distilled variant
    HUNYUAN_15_I2V = "hunyuan_15_i2v"  # v2.5.0: Image-to-video variant

    # LTX-Video (Lightricks) - v2.5.0
    LTXV = "ltxv"  # LTX-Video 2 text-to-video
    LTXV_I2V = "ltxv_i2v"  # LTX-Video 2 image-to-video

    # Wan (Alibaba/WaveSpeed) - v2.5.0
    WAN = "wan"  # Wan 2.1/2.2 standard
    WAN_FAST = "wan_fast"  # Wan 2.2 4-step with LoRA
    WAN_I2V = "wan_i2v"  # Wan image-to-video

    # Mochi (Genmo) - v2.5.0 EXPERIMENTAL
    MOCHI = "mochi"  # Mochi 1 (best text adherence)


class VideoFormat(str, Enum):
    """Output video formats."""

    MP4 = "mp4"
    GIF = "gif"
    WEBM = "webm"
    FRAMES = "frames"


class MotionStyle(str, Enum):
    """Motion intensity presets."""

    STATIC = "static"  # Minimal movement
    SUBTLE = "subtle"  # Gentle, slow movements
    MODERATE = "moderate"  # Normal motion
    DYNAMIC = "dynamic"  # Fast, energetic movement
    EXTREME = "extreme"  # Maximum motion


# =============================================================================
# VIDEO SETTINGS
# =============================================================================


@dataclass
class VideoSettings:
    """Settings for video generation."""

    model: VideoModel = VideoModel.ANIMATEDIFF_V3
    width: int = 512
    height: int = 512
    frames: int = 16
    fps: int = 8
    steps: int = 20
    cfg: float = 7.0
    seed: int = -1
    motion_scale: float = 1.0
    motion_style: MotionStyle = MotionStyle.MODERATE
    checkpoint: str | None = "dreamshaper_8.safetensors"
    format: VideoFormat = VideoFormat.MP4
    interpolate: bool = False  # Use RIFE to double frames

    # v2.5.0: New fields for advanced models
    variant: str | None = None  # Model variant (e.g., "1.3b", "14b", "distilled")
    upscale: bool = False  # Enable super-resolution (Hunyuan 1.5)
    shift: float | None = None  # ModelSamplingSD3 shift override
    precision: str = "fp16"  # Model precision (fp16, fp8, bf16)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.value,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": self.fps,
            "steps": self.steps,
            "cfg": self.cfg,
            "seed": self.seed,
            "motion_scale": self.motion_scale,
            "motion_style": self.motion_style.value,
            "checkpoint": self.checkpoint,
            "format": self.format.value,
            "interpolate": self.interpolate,
            "variant": self.variant,
            "upscale": self.upscale,
            "shift": self.shift,
            "precision": self.precision,
        }


# =============================================================================
# VIDEO PRESETS
# =============================================================================

VIDEO_PRESETS: dict[str, VideoSettings] = {
    # AnimateDiff presets
    "quick": VideoSettings(
        model=VideoModel.ANIMATEDIFF_LIGHTNING,
        width=512,
        height=512,
        frames=16,
        fps=8,
        steps=4,
        cfg=2.0,
        motion_style=MotionStyle.MODERATE,
    ),
    "standard": VideoSettings(
        model=VideoModel.ANIMATEDIFF_V3,
        width=512,
        height=512,
        frames=16,
        fps=8,
        steps=20,
        motion_style=MotionStyle.MODERATE,
    ),
    "quality": VideoSettings(
        model=VideoModel.ANIMATEDIFF_V3,
        width=768,
        height=768,
        frames=24,
        fps=12,
        steps=25,
        motion_style=MotionStyle.MODERATE,
        interpolate=True,
    ),
    "cinematic": VideoSettings(
        model=VideoModel.ANIMATEDIFF_V3,
        width=768,
        height=432,  # 16:9
        frames=32,
        fps=24,
        steps=30,
        motion_style=MotionStyle.SUBTLE,
        interpolate=True,
    ),
    "portrait": VideoSettings(
        model=VideoModel.ANIMATEDIFF_V3,
        width=512,
        height=768,  # 2:3
        frames=24,
        fps=12,
        steps=25,
        motion_style=MotionStyle.SUBTLE,
    ),
    "action": VideoSettings(
        model=VideoModel.ANIMATEDIFF_V3,
        width=512,
        height=512,
        frames=24,
        fps=16,
        steps=25,
        motion_style=MotionStyle.DYNAMIC,
        motion_scale=1.3,
    ),
    # SVD presets (image-to-video)
    "svd_short": VideoSettings(
        model=VideoModel.SVD,
        width=1024,
        height=576,
        frames=14,
        fps=6,
        steps=25,
        checkpoint=None,
    ),
    "svd_long": VideoSettings(
        model=VideoModel.SVD_XT,
        width=1024,
        height=576,
        frames=25,
        fps=6,
        steps=25,
        checkpoint=None,
    ),
    # CogVideoX preset
    "cogvideo": VideoSettings(
        model=VideoModel.COGVIDEOX,
        width=720,
        height=480,
        frames=48,
        fps=8,
        steps=50,
        checkpoint=None,
    ),
    # Hunyuan presets (original)
    "hunyuan": VideoSettings(
        model=VideoModel.HUNYUAN,
        width=1280,
        height=720,
        frames=45,
        fps=15,
        steps=30,
        checkpoint=None,
    ),
    "hunyuan_fast": VideoSettings(
        model=VideoModel.HUNYUAN,
        width=848,
        height=480,
        frames=33,
        fps=15,
        steps=20,
        checkpoint=None,
    ),
    # =========================================================================
    # v2.5.0: NEW VIDEO MODELS
    # =========================================================================
    # Hunyuan Video 1.5 presets (Tencent - improved quality, new encoders)
    "hunyuan15_720p": VideoSettings(
        model=VideoModel.HUNYUAN_15,
        width=1280,
        height=720,
        frames=121,
        fps=24,
        steps=20,
        cfg=6.0,
        shift=9.0,  # Shift for 720p T2V
        checkpoint=None,
    ),
    "hunyuan15_quality": VideoSettings(
        model=VideoModel.HUNYUAN_15,
        width=1280,
        height=720,
        frames=121,
        fps=24,
        steps=50,
        cfg=6.0,
        shift=9.0,
        checkpoint=None,
    ),
    "hunyuan15_fast": VideoSettings(
        model=VideoModel.HUNYUAN_15_FAST,
        width=848,
        height=480,
        frames=81,
        fps=24,
        steps=6,
        cfg=1.0,  # Distilled model uses CFG=1
        variant="distilled",
        checkpoint=None,
    ),
    "hunyuan15_1080p": VideoSettings(
        model=VideoModel.HUNYUAN_15,
        width=1920,
        height=1080,
        frames=121,
        fps=24,
        steps=20,
        cfg=6.0,
        shift=9.0,
        upscale=True,  # Uses super-resolution pipeline
        checkpoint=None,
    ),
    # LTX-Video 2 presets (Lightricks - fast, high quality)
    "ltx_quick": VideoSettings(
        model=VideoModel.LTXV,
        width=768,
        height=512,
        frames=97,
        fps=24,
        steps=20,
        cfg=3.0,  # LTXV uses low CFG
        checkpoint=None,
    ),
    "ltx_standard": VideoSettings(
        model=VideoModel.LTXV,
        width=768,
        height=512,
        frames=97,
        fps=24,
        steps=30,
        cfg=3.0,
        checkpoint=None,
    ),
    "ltx_quality": VideoSettings(
        model=VideoModel.LTXV,
        width=1280,
        height=720,
        frames=97,
        fps=24,
        steps=30,
        cfg=3.0,
        checkpoint=None,
    ),
    # Wan presets (Alibaba - efficient, great quality)
    "wan_1.3b": VideoSettings(
        model=VideoModel.WAN,
        width=832,
        height=480,
        frames=33,
        fps=16,
        steps=30,
        cfg=6.0,
        shift=8.0,  # Wan T2V shift
        variant="1.3b",
        checkpoint=None,
    ),
    "wan_14b": VideoSettings(
        model=VideoModel.WAN,
        width=640,
        height=640,
        frames=81,
        fps=16,
        steps=20,
        cfg=3.5,
        shift=8.0,
        variant="14b",
        precision="fp8",
        checkpoint=None,
    ),
    "wan_fast": VideoSettings(
        model=VideoModel.WAN_FAST,
        width=640,
        height=640,
        frames=81,
        fps=16,
        steps=4,
        cfg=1.0,  # 4-step with LoRA
        shift=5.0,  # Shift=5 for fast mode
        variant="14b_fast",
        checkpoint=None,
    ),
    "wan_quality": VideoSettings(
        model=VideoModel.WAN,
        width=1280,
        height=720,
        frames=81,
        fps=24,
        steps=30,
        cfg=3.5,
        shift=8.0,
        variant="14b",
        precision="fp8",
        checkpoint=None,
    ),
    # Mochi presets (Genmo - best text adherence, EXPERIMENTAL)
    "mochi": VideoSettings(
        model=VideoModel.MOCHI,
        width=848,
        height=480,
        frames=162,
        fps=30,
        steps=64,
        cfg=4.5,
        precision="bf16",
        checkpoint=None,
    ),
    "mochi_short": VideoSettings(
        model=VideoModel.MOCHI,
        width=848,
        height=480,
        frames=81,
        fps=30,
        steps=50,
        cfg=4.5,
        precision="fp8",
        checkpoint=None,
    ),
}


# =============================================================================
# VIDEO MODEL INFO
# =============================================================================


@dataclass
class VideoModelInfo:
    """Information about a video model."""

    id: str
    name: str
    description: str
    model: VideoModel
    text_to_video: bool = True
    image_to_video: bool = False
    min_vram_gb: int = 8
    estimated_time_seconds: int = 30
    max_frames: int = 32
    max_width: int = 768
    max_height: int = 768
    presets: list[str] = field(default_factory=list)


VIDEO_MODEL_INFO: dict[str, VideoModelInfo] = {
    "animatediff": VideoModelInfo(
        id="animatediff",
        name="AnimateDiff",
        description="Animate any Stable Diffusion checkpoint. Great balance of quality and speed.",
        model=VideoModel.ANIMATEDIFF_V3,
        text_to_video=True,
        image_to_video=False,
        min_vram_gb=8,
        estimated_time_seconds=30,
        max_frames=32,
        max_width=768,
        max_height=768,
        presets=["quick", "standard", "quality", "cinematic", "portrait", "action"],
    ),
    "animatediff_lightning": VideoModelInfo(
        id="animatediff_lightning",
        name="AnimateDiff Lightning",
        description="Ultra-fast 4-step video generation. Lower quality but very quick.",
        model=VideoModel.ANIMATEDIFF_LIGHTNING,
        text_to_video=True,
        image_to_video=False,
        min_vram_gb=6,
        estimated_time_seconds=10,
        max_frames=24,
        max_width=768,
        max_height=768,
        presets=["quick"],
    ),
    "svd": VideoModelInfo(
        id="svd",
        name="Stable Video Diffusion",
        description="Generate video from a single image. High quality motion.",
        model=VideoModel.SVD_XT,
        text_to_video=False,
        image_to_video=True,
        min_vram_gb=12,
        estimated_time_seconds=60,
        max_frames=25,
        max_width=1024,
        max_height=576,
        presets=["svd_short", "svd_long"],
    ),
    "cogvideo": VideoModelInfo(
        id="cogvideo",
        name="CogVideoX",
        description="High quality text-to-video with longer durations.",
        model=VideoModel.COGVIDEOX,
        text_to_video=True,
        image_to_video=False,
        min_vram_gb=16,
        estimated_time_seconds=120,
        max_frames=48,
        max_width=720,
        max_height=480,
        presets=["cogvideo"],
    ),
    "hunyuan": VideoModelInfo(
        id="hunyuan",
        name="Hunyuan Video",
        description="State-of-the-art text-to-video from Tencent. Best quality, needs 24GB+ VRAM.",
        model=VideoModel.HUNYUAN,
        text_to_video=True,
        image_to_video=False,
        min_vram_gb=24,
        estimated_time_seconds=180,
        max_frames=45,
        max_width=1280,
        max_height=720,
        presets=["hunyuan", "hunyuan_fast"],
    ),
    # =========================================================================
    # v2.5.0: NEW VIDEO MODEL INFO
    # =========================================================================
    "hunyuan_15": VideoModelInfo(
        id="hunyuan_15",
        name="Hunyuan Video 1.5",
        description="Improved Hunyuan with new encoders (Qwen 2.5 VL + ByT5). 720p/1080p via SR.",
        model=VideoModel.HUNYUAN_15,
        text_to_video=True,
        image_to_video=True,
        min_vram_gb=14,
        estimated_time_seconds=120,
        max_frames=121,
        max_width=1920,
        max_height=1080,
        presets=["hunyuan15_720p", "hunyuan15_quality", "hunyuan15_fast", "hunyuan15_1080p"],
    ),
    "ltxv": VideoModelInfo(
        id="ltxv",
        name="LTX-Video 2",
        description="Lightricks fast video model. Up to 4K, excellent quality/speed ratio.",
        model=VideoModel.LTXV,
        text_to_video=True,
        image_to_video=True,
        min_vram_gb=12,
        estimated_time_seconds=30,
        max_frames=97,
        max_width=1920,
        max_height=1080,
        presets=["ltx_quick", "ltx_standard", "ltx_quality"],
    ),
    "wan": VideoModelInfo(
        id="wan",
        name="Wan 2.1/2.2",
        description="Alibaba MoE video model. 1.3B efficient or 14B high-quality variants.",
        model=VideoModel.WAN,
        text_to_video=True,
        image_to_video=True,
        min_vram_gb=6,  # 1.3B variant
        estimated_time_seconds=90,
        max_frames=81,
        max_width=1280,
        max_height=720,
        presets=["wan_1.3b", "wan_14b", "wan_fast", "wan_quality"],
    ),
    "mochi": VideoModelInfo(
        id="mochi",
        name="Mochi 1 (Experimental)",
        description="Genmo 10B model. Best text adherence, 480p@30fps. Requires 12GB+ VRAM.",
        model=VideoModel.MOCHI,
        text_to_video=True,
        image_to_video=False,
        min_vram_gb=12,
        estimated_time_seconds=180,
        max_frames=162,
        max_width=848,
        max_height=480,
        presets=["mochi", "mochi_short"],
    ),
}


# =============================================================================
# VIDEO WORKFLOW BUILDER
# =============================================================================


class VideoWorkflowBuilder:
    """
    Builds ComfyUI workflows for video generation.

    Each video model requires a different workflow structure.
    This builder abstracts those differences so users just specify
    what they want, not how to build it.
    """

    def __init__(self):
        self._builders = {
            # AnimateDiff family
            VideoModel.ANIMATEDIFF_V2: self._build_animatediff,
            VideoModel.ANIMATEDIFF_V3: self._build_animatediff,
            VideoModel.ANIMATEDIFF_LIGHTNING: self._build_animatediff_lightning,
            # Stable Video Diffusion
            VideoModel.SVD: self._build_svd,
            VideoModel.SVD_XT: self._build_svd,
            # CogVideoX
            VideoModel.COGVIDEOX: self._build_cogvideo,
            # Hunyuan (original)
            VideoModel.HUNYUAN: self._build_hunyuan,
            # v2.5.0: Hunyuan 1.5
            VideoModel.HUNYUAN_15: self._build_hunyuan_15,
            VideoModel.HUNYUAN_15_FAST: self._build_hunyuan_15,
            VideoModel.HUNYUAN_15_I2V: self._build_hunyuan_15,
            # v2.5.0: LTX-Video
            VideoModel.LTXV: self._build_ltxv,
            VideoModel.LTXV_I2V: self._build_ltxv,
            # v2.5.0: Wan
            VideoModel.WAN: self._build_wan,
            VideoModel.WAN_FAST: self._build_wan_fast,
            VideoModel.WAN_I2V: self._build_wan,
            # v2.5.0: Mochi
            VideoModel.MOCHI: self._build_mochi,
        }

    def build(
        self, prompt: str, negative: str, settings: VideoSettings, init_image: str | None = None
    ) -> dict[str, Any]:
        """
        Build a ComfyUI workflow for video generation.

        Args:
            prompt: Positive prompt
            negative: Negative prompt
            settings: Video settings
            init_image: Base64 image for img2vid (SVD)

        Returns:
            ComfyUI workflow JSON
        """
        builder = self._builders.get(settings.model)
        if not builder:
            raise ValueError(f"Unknown video model: {settings.model}")

        seed = settings.seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        return builder(prompt, negative, settings, seed, init_image)

    def _get_motion_scale(self, settings: VideoSettings) -> float:
        """Calculate motion scale from style and multiplier."""
        style_scales = {
            MotionStyle.STATIC: 0.5,
            MotionStyle.SUBTLE: 0.75,
            MotionStyle.MODERATE: 1.0,
            MotionStyle.DYNAMIC: 1.25,
            MotionStyle.EXTREME: 1.5,
        }
        base = style_scales.get(settings.motion_style, 1.0)
        return base * settings.motion_scale

    def _build_animatediff(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """Build AnimateDiff v2/v3 workflow."""
        motion_scale = self._get_motion_scale(settings)
        motion_module = (
            "v3_sd15_mm.ckpt"
            if settings.model == VideoModel.ANIMATEDIFF_V3
            else "mm_sd_v15_v2.ckpt"
        )

        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": settings.checkpoint or "dreamshaper_8.safetensors"},
            },
            "2": {
                "class_type": "ADE_LoadAnimateDiffModel",
                "inputs": {"model_name": motion_module},
            },
            "3": {
                "class_type": "ADE_ApplyAnimateDiffModel",
                "inputs": {
                    "model": ["1", 0],
                    "motion_model": ["2", 0],
                    "scale_multival": motion_scale,
                },
            },
            "4": {
                "class_type": "ADE_EmptyLatentImageLarge",
                "inputs": {
                    "width": settings.width,
                    "height": settings.height,
                    "batch_size": settings.frames,
                },
            },
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["3", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": settings.steps,
                    "cfg": settings.cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
            "9": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": settings.fps,
                    "loop_count": 0,
                    "filename_prefix": "comfy_headless_video",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

        # Add RIFE interpolation if requested
        if settings.interpolate:
            workflow["10"] = {
                "class_type": "RIFE VFI",
                "inputs": {"frames": ["8", 0], "multiplier": 2, "ckpt_name": "rife49.pth"},
            }
            workflow["9"]["inputs"]["images"] = ["10", 0]
            workflow["9"]["inputs"]["frame_rate"] = settings.fps * 2

        return workflow

    def _build_animatediff_lightning(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """Build AnimateDiff Lightning (4-step fast) workflow."""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": settings.checkpoint or "dreamshaper_8.safetensors"},
            },
            "2": {
                "class_type": "ADE_LoadAnimateDiffModel",
                "inputs": {"model_name": "animatediff_lightning_4step.safetensors"},
            },
            "3": {
                "class_type": "ADE_ApplyAnimateDiffModel",
                "inputs": {
                    "model": ["1", 0],
                    "motion_model": ["2", 0],
                },
            },
            "4": {
                "class_type": "ADE_EmptyLatentImageLarge",
                "inputs": {
                    "width": settings.width,
                    "height": settings.height,
                    "batch_size": settings.frames,
                },
            },
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["3", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": 4,  # Lightning is 4-step
                    "cfg": 2.0,  # Lower CFG for Lightning
                    "sampler_name": "euler",
                    "scheduler": "sgm_uniform",
                    "denoise": 1.0,
                },
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
            "9": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": settings.fps,
                    "loop_count": 0,
                    "filename_prefix": "comfy_headless_lightning",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

    def _build_svd(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """Build Stable Video Diffusion workflow (img2vid)."""
        if not init_image:
            raise ValueError("SVD requires an init_image")

        model_name = (
            "svd_xt.safetensors" if settings.model == VideoModel.SVD_XT else "svd.safetensors"
        )
        num_frames = 25 if settings.model == VideoModel.SVD_XT else 14

        return {
            "1": {"class_type": "LoadImageFromBase64", "inputs": {"base64_data": init_image}},
            "2": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": model_name}},
            "3": {
                "class_type": "SVD_img2vid_Conditioning",
                "inputs": {
                    "clip_vision": ["2", 1],
                    "init_image": ["1", 0],
                    "vae": ["2", 2],
                    "width": settings.width,
                    "height": settings.height,
                    "video_frames": num_frames,
                    "motion_bucket_id": 127,
                    "fps": 6,
                    "augmentation_level": 0.0,
                },
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["3", 1],
                    "latent_image": ["3", 2],
                    "seed": seed,
                    "steps": settings.steps,
                    "cfg": 2.5,
                    "sampler_name": "euler",
                    "scheduler": "karras",
                    "denoise": 1.0,
                },
            },
            "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["2", 2]}},
            "6": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["5", 0],
                    "frame_rate": settings.fps,
                    "loop_count": 0,
                    "filename_prefix": "comfy_headless_svd",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

    def _build_cogvideo(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """Build CogVideoX workflow."""
        return {
            "1": {
                "class_type": "CogVideoModelLoader",
                "inputs": {
                    "model_path": "CogVideoX-5b-transformer.safetensors",
                    "vae_path": "CogVideoX-5b-vae.safetensors",
                    "dtype": "bf16",
                },
            },
            "2": {
                "class_type": "CogVideoTextEncode",
                "inputs": {"prompt": prompt, "pipe": ["1", 0]},
            },
            "3": {
                "class_type": "CogVideoSampler",
                "inputs": {
                    "pipe": ["1", 0],
                    "embeds": ["2", 0],
                    "width": settings.width,
                    "height": settings.height,
                    "num_frames": settings.frames,
                    "steps": settings.steps,
                    "cfg": settings.cfg,
                    "seed": seed,
                },
            },
            "4": {
                "class_type": "CogVideoVAEDecode",
                "inputs": {"pipe": ["1", 0], "samples": ["3", 0]},
            },
            "5": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["4", 0],
                    "frame_rate": settings.fps,
                    "filename_prefix": "comfy_headless_cogvideo",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

    def _build_hunyuan(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """Build Hunyuan Video workflow."""
        workflow = {
            "1": {
                "class_type": "HunyuanVideoModelLoader",
                "inputs": {
                    "model_path": "hunyuan_video_720_fp8_e4m3fn.safetensors",
                    "vae_path": "hunyuan_video_vae_bf16.safetensors",
                    "text_encoder_path": "llava_llama3_fp8_scaled.safetensors",
                    "precision": "fp8_e4m3fn",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "clip-vit-large-patch14/model.safetensors"},
            },
            "3": {
                "class_type": "HunyuanVideoTextEncode",
                "inputs": {
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "hunyuan_pipe": ["1", 0],
                    "clip": ["2", 0],
                },
            },
            "4": {
                "class_type": "HunyuanVideoSampler",
                "inputs": {
                    "hunyuan_pipe": ["1", 0],
                    "conditioning": ["3", 0],
                    "width": settings.width,
                    "height": settings.height,
                    "num_frames": settings.frames,
                    "steps": settings.steps,
                    "cfg": 6.0,
                    "seed": seed,
                    "embedded_guidance_scale": 6.0,
                },
            },
            "5": {
                "class_type": "HunyuanVideoVAEDecode",
                "inputs": {"hunyuan_pipe": ["1", 0], "samples": ["4", 0]},
            },
            "6": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["5", 0],
                    "frame_rate": settings.fps,
                    "filename_prefix": "comfy_headless_hunyuan",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

        # Add RIFE interpolation if requested
        if settings.interpolate:
            workflow["7"] = {
                "class_type": "RIFE VFI",
                "inputs": {"frames": ["5", 0], "multiplier": 2, "ckpt_name": "rife49.pth"},
            }
            workflow["6"]["inputs"]["images"] = ["7", 0]
            workflow["6"]["inputs"]["frame_rate"] = settings.fps * 2

        return workflow

    # =========================================================================
    # v2.5.0: NEW VIDEO MODEL BUILDERS
    # =========================================================================

    def _build_hunyuan_15(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """
        Build Hunyuan Video 1.5 workflow.

        Uses new architecture:
        - DualCLIPLoader (Qwen 2.5 VL + ByT5)
        - SamplerCustomAdvanced with CFGGuider
        - Optional super-resolution for 1080p
        """
        shift = settings.shift or 9.0  # Default shift for 720p T2V

        # Determine model paths based on variant
        if settings.variant == "distilled":
            unet_name = "hunyuanvideo1.5_720p_t2v_distilled_fp16.safetensors"
            cfg_value = 1.0  # Distilled uses CFG=1
        else:
            unet_name = "hunyuanvideo1.5_720p_t2v_fp16.safetensors"
            cfg_value = settings.cfg

        workflow = {
            # DualCLIPLoader for text encoders
            "1": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "clip_name2": "byt5_small_glyphxl_fp16.safetensors",
                    "type": "hunyuan_video_15",
                    "device": "default",
                },
            },
            # VAE
            "2": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "hunyuanvideo15_vae_fp16.safetensors"},
            },
            # Diffusion model
            "3": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
            },
            # Latent
            "4": {
                "class_type": "EmptyHunyuanVideo15Latent",
                "inputs": {
                    "width": settings.width,
                    "height": settings.height,
                    "length": settings.frames,
                    "batch_size": 1,
                },
            },
            # Text encoding
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 0]}},
            # Model sampling with shift
            "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": shift}},
            # CFG Guider
            "8": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "cfg": cfg_value,
                },
            },
            # Scheduler
            "9": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["3", 0],
                    "scheduler": "simple",
                    "steps": settings.steps,
                    "denoise": 1.0,
                },
            },
            # Noise
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            # Sampler
            "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            # Custom sampler
            "12": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["8", 0],
                    "sampler": ["11", 0],
                    "sigmas": ["9", 0],
                    "latent_image": ["4", 0],
                },
            },
            # Decode
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["2", 0]}},
            "14": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["13", 0],
                    "frame_rate": settings.fps,
                    "filename_prefix": "comfy_headless_hunyuan15",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

        # Add super-resolution if upscale=True
        if settings.upscale:
            workflow["15"] = {
                "class_type": "LatentUpscaleModelLoader",
                "inputs": {"model_name": "hunyuanvideo15_latent_upsampler_1080p.safetensors"},
            }
            workflow["16"] = {
                "class_type": "HunyuanVideo15LatentUpscaleWithModel",
                "inputs": {
                    "model": ["15", 0],
                    "samples": ["12", 0],
                    "interpolation": "bilinear",
                    "width": 1920,
                    "height": 1080,
                    "extend_length": "disabled",
                },
            }
            # Update decode to use upscaled latent
            workflow["13"]["inputs"]["samples"] = ["16", 0]

        return workflow

    def _build_ltxv(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """
        Build LTX-Video 2 workflow.

        Uses:
        - SamplerCustom with LTXVScheduler
        - CLIPLoader type="ltxv" for T5 encoder
        - Low CFG (typically 3.0)
        """
        workflow = {
            # Checkpoint
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "ltx-video-2b-v0.9.5.safetensors"},
            },
            # T5 text encoder
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "t5xxl_fp16.safetensors",
                    "type": "ltxv",
                    "device": "default",
                },
            },
            # Latent
            "3": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": settings.width,
                    "height": settings.height,
                    "length": settings.frames,
                    "batch_size": 1,
                },
            },
            # Text encoding
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            # Conditioning with frame rate
            "6": {
                "class_type": "LTXVConditioning",
                "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": settings.fps},
            },
            # Scheduler
            "7": {
                "class_type": "LTXVScheduler",
                "inputs": {
                    "latent": ["3", 0],
                    "steps": settings.steps,
                    "max_shift": 2.05,
                    "base_shift": 0.95,
                    "stretch": True,
                    "terminal": 0.1,
                },
            },
            # Sampler select
            "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            # Custom sampler
            "9": {
                "class_type": "SamplerCustom",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "sampler": ["8", 0],
                    "sigmas": ["7", 0],
                    "latent_image": ["3", 0],
                    "add_noise": True,
                    "noise_seed": seed,
                    "cfg": settings.cfg,  # Typically 3.0 for LTXV
                },
            },
            # Decode
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
            # Output
            "11": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["10", 0],
                    "frame_rate": settings.fps,
                    "filename_prefix": "comfy_headless_ltxv",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

        # Image-to-video variant
        if init_image:
            workflow["2.5"] = {
                "class_type": "LoadImageFromBase64",
                "inputs": {"base64_data": init_image},
            }
            workflow["3"] = {
                "class_type": "LTXVImgToVideo",
                "inputs": {
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "vae": ["1", 2],
                    "image": ["2.5", 0],
                    "width": settings.width,
                    "height": settings.height,
                    "length": settings.frames,
                    "batch_size": 1,
                    "image_noise_scale": 0.15,
                },
            }
            # Update conditioning to use I2V outputs
            workflow["6"]["inputs"]["positive"] = ["3", 0]
            workflow["6"]["inputs"]["negative"] = ["3", 1]
            workflow["7"]["inputs"]["latent"] = ["3", 2]
            workflow["9"]["inputs"]["latent_image"] = ["3", 2]

        return workflow

    def _build_wan(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """
        Build Wan 2.1/2.2 workflow.

        Uses:
        - UNETLoader for diffusion model
        - CLIPLoader type="wan" for UMT5-XXL
        - ModelSamplingSD3 with shift=8 (T2V)
        """
        # Model selection based on variant
        model_map = {
            "1.3b": "wan2.1_t2v_1.3B_fp16.safetensors",
            "14b": "wan2.2_t2v_14B_fp8_scaled.safetensors",
        }
        unet_name = model_map.get(settings.variant, model_map["1.3b"])
        shift = settings.shift or 8.0

        workflow = {
            # UNET
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
            },
            # Text encoder
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default",
                },
            },
            # VAE
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            # Latent (shared with Hunyuan)
            "4": {
                "class_type": "EmptyHunyuanLatentVideo",
                "inputs": {
                    "width": settings.width,
                    "height": settings.height,
                    "length": settings.frames,
                    "batch_size": 1,
                },
            },
            # Text encoding
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            # Model sampling with shift
            "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": shift}},
            # KSampler
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": settings.steps,
                    "cfg": settings.cfg,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            # Decode
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            # Output
            "10": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["9", 0],
                    "frame_rate": settings.fps,
                    "filename_prefix": "comfy_headless_wan",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

        # Image-to-video extension
        if init_image:
            workflow["11"] = {
                "class_type": "LoadImageFromBase64",
                "inputs": {"base64_data": init_image},
            }
            workflow["12"] = {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": "clip_vision_h.safetensors"},
            }
            workflow["13"] = {
                "class_type": "CLIPVisionEncode",
                "inputs": {"clip_vision": ["12", 0], "image": ["11", 0], "crop": "none"},
            }
            workflow["14"] = {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "vae": ["3", 0],
                    "clip_vision_output": ["13", 0],
                    "start_image": ["11", 0],
                    "width": settings.width,
                    "height": settings.height,
                    "length": settings.frames,
                    "batch_size": 1,
                },
            }
            # Reconnect sampler
            workflow["8"]["inputs"]["positive"] = ["14", 0]
            workflow["8"]["inputs"]["negative"] = ["14", 1]
            workflow["8"]["inputs"]["latent_image"] = ["14", 2]

        return workflow

    def _build_wan_fast(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """
        Build Wan 2.2 4-step fast workflow.

        Uses dual model approach:
        - High noise model (steps 0-2)
        - Low noise model (steps 2-4)
        - LoRA for each model
        """
        shift = settings.shift or 5.0  # Shift=5 for fast mode

        workflow = {
            # High noise model
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                },
            },
            # Low noise model
            "2": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                },
            },
            # LoRA for high noise
            "3": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["1", 0],
                    "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                    "strength_model": 1.0,
                },
            },
            # LoRA for low noise
            "4": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["2", 0],
                    "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
                    "strength_model": 1.0,
                },
            },
            # Text encoder
            "5": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default",
                },
            },
            # VAE
            "6": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            # Latent
            "7": {
                "class_type": "EmptyHunyuanLatentVideo",
                "inputs": {
                    "width": settings.width,
                    "height": settings.height,
                    "length": settings.frames,
                    "batch_size": 1,
                },
            },
            # Text encoding
            "8": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["5", 0]}},
            "9": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["5", 0]}},
            # Model sampling for high noise
            "10": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": shift}},
            # Model sampling for low noise
            "11": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": shift}},
            # First sampler (high noise, steps 0-2)
            "12": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["10", 0],
                    "positive": ["8", 0],
                    "negative": ["9", 0],
                    "latent_image": ["7", 0],
                    "seed": seed,
                    "steps": 4,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 0,
                    "end_at_step": 2,
                    "add_noise": "enable",
                    "return_with_leftover_noise": "enable",
                },
            },
            # Second sampler (low noise, steps 2-4)
            "13": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["11", 0],
                    "positive": ["8", 0],
                    "negative": ["9", 0],
                    "latent_image": ["12", 0],
                    "seed": seed,
                    "steps": 4,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 2,
                    "end_at_step": 4,
                    "add_noise": "disable",
                    "return_with_leftover_noise": "disable",
                },
            },
            # Decode
            "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["6", 0]}},
            # Output
            "15": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["14", 0],
                    "frame_rate": settings.fps,
                    "filename_prefix": "comfy_headless_wan_fast",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

        return workflow

    def _build_mochi(
        self,
        prompt: str,
        negative: str,
        settings: VideoSettings,
        seed: int,
        init_image: str | None = None,
    ) -> dict[str, Any]:
        """
        Build Mochi 1 workflow (EXPERIMENTAL).

        Note: This is a preliminary implementation based on expected
        ComfyUI node structure. Actual nodes may differ.
        """
        # Determine precision
        dtype = "bf16" if settings.precision == "bf16" else "fp8"

        workflow = {
            # Model loader
            "1": {
                "class_type": "MochiModelLoader",
                "inputs": {
                    "model_path": f"mochi_preview_dit_{dtype}.safetensors",
                    "precision": dtype,
                },
            },
            # T5 text encoder
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "t5xxl_fp16.safetensors",
                    "type": "mochi",
                    "device": "default",
                },
            },
            # Text encode
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            # Latent
            "4": {
                "class_type": "EmptyMochiLatentVideo",
                "inputs": {
                    "width": settings.width,
                    "height": settings.height,
                    "frames": settings.frames,
                    "batch_size": 1,
                },
            },
            # Sampler
            "5": {
                "class_type": "MochiSampler",
                "inputs": {
                    "model": ["1", 0],
                    "conditioning": ["3", 0],
                    "latent": ["4", 0],
                    "steps": settings.steps,
                    "cfg": settings.cfg,
                    "seed": seed,
                },
            },
            # Decode
            "6": {"class_type": "MochiVAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 1]}},
            # Output
            "7": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["6", 0],
                    "frame_rate": 30,  # Mochi is 30fps
                    "filename_prefix": "comfy_headless_mochi",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

        return workflow


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_builder: VideoWorkflowBuilder | None = None


def get_video_builder() -> VideoWorkflowBuilder:
    """Get singleton video workflow builder."""
    global _builder
    if _builder is None:
        _builder = VideoWorkflowBuilder()
    return _builder


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_video_workflow(
    prompt: str,
    negative: str = "ugly, blurry, low quality",
    preset: str = "standard",
    init_image: str | None = None,
    **overrides,
) -> dict[str, Any]:
    """
    Build a video workflow with preset settings.

    Simple usage:
        workflow = build_video_workflow(
            prompt="a cat walking",
            preset="quality"
        )
    """
    settings = VIDEO_PRESETS.get(preset, VIDEO_PRESETS["standard"])

    # Apply any overrides
    if overrides:
        settings_dict = settings.to_dict()
        settings_dict.update(overrides)
        settings = VideoSettings(
            model=VideoModel(settings_dict.get("model", settings.model.value)),
            width=settings_dict.get("width", settings.width),
            height=settings_dict.get("height", settings.height),
            frames=settings_dict.get("frames", settings.frames),
            fps=settings_dict.get("fps", settings.fps),
            steps=settings_dict.get("steps", settings.steps),
            cfg=settings_dict.get("cfg", settings.cfg),
            seed=settings_dict.get("seed", settings.seed),
            motion_scale=settings_dict.get("motion_scale", settings.motion_scale),
            motion_style=MotionStyle(
                settings_dict.get("motion_style", settings.motion_style.value)
            ),
            checkpoint=settings_dict.get("checkpoint", settings.checkpoint),
            format=VideoFormat(settings_dict.get("format", settings.format.value)),
            interpolate=settings_dict.get("interpolate", settings.interpolate),
        )

    builder = get_video_builder()
    return builder.build(prompt, negative, settings, init_image)


def get_video_preset(name: str) -> VideoSettings | None:
    """Get a video preset by name."""
    return VIDEO_PRESETS.get(name)


def list_video_presets() -> list[str]:
    """List available video preset names."""
    return list(VIDEO_PRESETS.keys())


def list_video_models() -> dict[str, VideoModelInfo]:
    """List available video models with their info."""
    return VIDEO_MODEL_INFO.copy()


def get_recommended_preset(
    intent: str = "general", quality: str = "standard", vram_gb: float = 8.0
) -> str:
    """
    Get recommended video preset based on intent and hardware.

    v2.5.0: Updated with new models (LTX, Wan, Hunyuan 1.5)

    Makes it easy for users - they say what they want, we pick the best preset.
    """
    # v2.5.0: Updated VRAM tiers with new models
    if vram_gb < 8:
        # Very low VRAM: AnimateDiff Lightning or Wan 1.3B
        if quality == "fast":
            return "quick"  # Lightning is fastest
        return "wan_1.3b"  # Wan 1.3B is efficient and high quality

    elif vram_gb < 12:
        # 8-12GB: Wan 1.3B, LTX quick, AnimateDiff
        candidates = ["wan_1.3b", "ltx_quick", "quick", "standard", "portrait", "action"]

    elif vram_gb < 16:
        # 12-16GB: LTX standard, Wan 14B, Mochi short
        candidates = [
            "ltx_standard",
            "ltx_quality",
            "wan_14b",
            "mochi_short",
            "standard",
            "quality",
            "cinematic",
            "portrait",
            "action",
        ]

    elif vram_gb < 24:
        # 16-24GB: Hunyuan 1.5 720p, LTX quality, Wan quality
        candidates = [
            "hunyuan15_720p",
            "hunyuan15_fast",
            "ltx_quality",
            "wan_quality",
            "wan_fast",
            "quality",
            "cinematic",
            "svd_short",
            "svd_long",
            "cogvideo",
        ]

    else:
        # 24GB+: All presets available including Hunyuan 1.5 quality and Mochi
        candidates = list(VIDEO_PRESETS.keys())

    # Quality preference
    if quality == "fast":
        # Fastest options per tier
        fast_order = ["quick", "wan_fast", "hunyuan15_fast", "ltx_quick", "wan_1.3b"]
        for preset in fast_order:
            if preset in candidates:
                return preset
        return candidates[0]

    elif quality == "best":
        # Best quality options
        best_order = [
            "hunyuan15_quality",
            "mochi",
            "hunyuan15_720p",
            "wan_quality",
            "ltx_quality",
            "hunyuan",
            "quality",
            "cinematic",
        ]
        for preset in best_order:
            if preset in candidates:
                return preset

    # Intent-based selection
    if intent in ["portrait", "character", "person"]:
        if "portrait" in candidates:
            return "portrait"
    elif intent in ["landscape", "scene", "environment", "cinematic", "film"]:
        # Prefer newer high-quality models for cinematic
        cinematic_order = ["hunyuan15_720p", "ltx_quality", "cinematic"]
        for preset in cinematic_order:
            if preset in candidates:
                return preset
    elif intent in ["action", "dynamic", "motion"]:
        # Action prefers high frame rate
        if "action" in candidates:
            return "action"
        if "ltx_standard" in candidates:
            return "ltx_standard"  # LTX has good motion

    # Default: prefer newer models
    default_order = ["ltx_standard", "wan_1.3b", "standard"]
    for preset in default_order:
        if preset in candidates:
            return preset

    return candidates[0] if candidates else "standard"
