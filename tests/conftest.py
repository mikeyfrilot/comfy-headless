"""Pytest configuration and fixtures."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_settings():
    """Provide mock settings for tests."""
    from comfy_headless.config import Settings

    return Settings()


@pytest.fixture
def mock_comfyui_response():
    """Mock ComfyUI API response."""
    return {
        "system_stats": {
            "devices": [{"name": "cuda:0", "vram_total": 16 * 1024**3, "vram_free": 12 * 1024**3}]
        }
    }


@pytest.fixture
def mock_session():
    """Mock requests session."""
    session = MagicMock()
    response = MagicMock()
    response.ok = True
    response.status_code = 200
    response.json.return_value = {"prompt_id": "test-123"}
    session.request.return_value = response
    session.get.return_value = response
    session.post.return_value = response
    return session


@pytest.fixture
def sample_workflow():
    """Sample txt2img workflow."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7.0,
                "denoise": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 12345,
                "steps": 20,
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "model.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 1024, "width": 1024},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": "a beautiful sunset"},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": "bad quality"},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "test", "images": ["8", 0]},
        },
    }
