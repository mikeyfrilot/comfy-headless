"""
Comfy Headless - ComfyUI API Client
=====================================

Production-ready HTTP client for ComfyUI communication with:
- Connection pooling via requests.Session
- Automatic retry with exponential backoff
- Circuit breaker for failure resilience
- Structured logging
- Proper error handling

Usage:
    from comfy_headless import ComfyClient

    client = ComfyClient()
    if client.is_online():
        result = client.generate_image("a beautiful sunset")
"""

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import settings
from .exceptions import (
    ComfyUIConnectionError,
    ComfyUIOfflineError,
)
from .logging_config import LogContext, get_logger
from .retry import RateLimiter, get_circuit_breaker


def _safe_json_parse(response: "requests.Response", context: str = "") -> dict:
    """
    Safely parse JSON from a response with proper error handling.

    Args:
        response: The requests Response object
        context: Description of what we were trying to do (for error messages)

    Returns:
        Parsed JSON as dict

    Raises:
        ComfyUIConnectionError: If JSON parsing fails
    """
    try:
        return response.json()
    except json.JSONDecodeError as e:
        logger.error(
            f"Invalid JSON response{f' ({context})' if context else ''}",
            extra={"error": str(e), "response_text": response.text[:200] if response.text else ""},
        )
        raise ComfyUIConnectionError(
            message=f"Invalid JSON response from ComfyUI{f' while {context}' if context else ''}",
            url=response.url,
            cause=e,
        )


def _safe_get_nested(data: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely navigate nested dictionaries without raising KeyError or TypeError.

    Args:
        data: The root dictionary to navigate
        *keys: Sequence of keys to traverse
        default: Value to return if path doesn't exist or has wrong type

    Returns:
        The value at the nested path, or default if not found

    Example:
        _safe_get_nested(workflow, "3", "inputs", "seed", default=-1)
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


# Lazy import for video module to avoid circular imports
_video_builder = None


def _get_video_builder():
    """Lazy import of VideoWorkflowBuilder."""
    global _video_builder
    if _video_builder is None:
        from .video import get_video_builder

        _video_builder = get_video_builder()
    return _video_builder


logger = get_logger(__name__)

__all__ = ["ComfyClient"]


class ComfyClient:
    """
    HTTP client for ComfyUI API with production-ready features.

    Attributes:
        base_url: ComfyUI server URL
        client_id: Unique identifier for this client instance
    """

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: int | None = None,
        rate_limit_per_seconds: float = 1.0,
    ):
        """
        Initialize the ComfyUI client.

        Args:
            base_url: ComfyUI server URL (default from settings)
            rate_limit: Max requests per time window (None = no limit)
            rate_limit_per_seconds: Time window for rate limiting
        """
        self.base_url = (base_url or settings.comfyui.url).rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._session: requests.Session | None = None
        self._circuit = get_circuit_breaker("comfyui")

        # Rate limiter (optional)
        self._rate_limiter: RateLimiter | None = None
        if rate_limit is not None and rate_limit > 0:
            self._rate_limiter = RateLimiter(rate=rate_limit, per_seconds=rate_limit_per_seconds)
            logger.debug(f"Rate limiter enabled: {rate_limit} req/{rate_limit_per_seconds}s")

        logger.info(
            "ComfyClient initialized",
            extra={"base_url": self.base_url, "client_id": self.client_id[:8]},
        )

    @property
    def session(self) -> requests.Session:
        """Get or create the HTTP session with connection pooling."""
        if self._session is None or not hasattr(self._session, "headers"):
            self._session = self._create_session()
        return self._session

    def _create_session(self) -> requests.Session:
        """Create a configured requests session."""
        session = requests.Session()

        # Configure retry for transient errors
        retry_strategy = Retry(
            total=settings.retry.max_retries,
            backoff_factor=settings.retry.backoff_base,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def close(self):
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
            logger.debug("HTTP session closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # =========================================================================
    # INTERNAL REQUEST METHODS
    # =========================================================================

    def _request(
        self, method: str, endpoint: str, timeout: float | None = None, **kwargs
    ) -> requests.Response:
        """
        Make an HTTP request with circuit breaker and rate limiter protection.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            timeout: Request timeout
            **kwargs: Additional request arguments

        Returns:
            Response object

        Raises:
            ComfyUIConnectionError: If connection fails
        """
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or settings.comfyui.timeout_read

        # Apply rate limiting if configured
        if self._rate_limiter is not None:
            if not self._rate_limiter.acquire(blocking=True, timeout=30.0):
                logger.warning(f"Rate limit timeout for {endpoint}")
                raise ComfyUIConnectionError(
                    message="Rate limit timeout - too many requests", url=self.base_url
                )

        try:
            with self._circuit:
                response = self.session.request(method, url, timeout=timeout, **kwargs)
                return response

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error: {endpoint}", extra={"error": str(e)})
            raise ComfyUIConnectionError(
                message=f"Failed to connect to ComfyUI at {self.base_url}",
                url=self.base_url,
                cause=e,
            )
        except requests.exceptions.Timeout as e:
            logger.warning(f"Request timeout: {endpoint}", extra={"timeout": timeout})
            raise ComfyUIConnectionError(
                message=f"Request timed out after {timeout}s", url=url, cause=e
            )
        except Exception as e:
            logger.error(f"Request failed: {endpoint}", extra={"error": str(e)})
            raise ComfyUIConnectionError(message=f"Request failed: {e}", url=url, cause=e)

    def _get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request."""
        return self._request("GET", endpoint, **kwargs)

    def _post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a POST request."""
        return self._request("POST", endpoint, **kwargs)

    # =========================================================================
    # CONNECTION
    # =========================================================================

    def is_online(self) -> bool:
        """
        Check if ComfyUI is running and responsive.

        Uses a quick check that bypasses circuit breaker and session retries
        for fast UI initialization when ComfyUI is offline.

        Returns:
            True if ComfyUI is online, False otherwise
        """
        try:
            # Bypass session (which has retry adapter) - use raw requests
            # Use tuple timeout: (connect_timeout, read_timeout) for faster failure
            response = requests.get(
                f"{self.base_url}/system_stats",
                timeout=(1.0, 1.0),  # Fast fail - 1 second connect, 1 second read
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_system_stats(self) -> dict | None:
        """
        Get ComfyUI system stats including GPU and VRAM info.

        Returns:
            Dict with system stats or None if unavailable
        """
        try:
            response = self._get("/system_stats")
            if response.ok:
                return _safe_json_parse(response, "getting system stats")
        except ComfyUIConnectionError:
            # JSON parse error - already logged
            pass
        except Exception as e:
            logger.debug(f"Failed to get system stats: {e}")
        return None

    def get_vram_gb(self) -> float:
        """
        Get available GPU VRAM in gigabytes.

        Queries ComfyUI's /system_stats endpoint for GPU memory info.
        Falls back to 8GB if unable to detect.

        Returns:
            VRAM in GB (total, not free)
        """
        try:
            stats = self.get_system_stats()
            if stats:
                # ComfyUI returns devices array with vram_total in bytes
                devices = stats.get("devices", [])
                if devices:
                    vram_bytes = devices[0].get("vram_total", 0)
                    if vram_bytes > 0:
                        vram_gb = vram_bytes / (1024**3)
                        logger.debug(f"Detected VRAM: {vram_gb:.1f}GB")
                        return vram_gb
        except Exception as e:
            logger.debug(f"VRAM detection failed: {e}")

        # Fallback to reasonable default
        logger.debug("Using default VRAM estimate: 8GB")
        return 8.0

    def get_free_vram_gb(self) -> float:
        """
        Get free GPU VRAM in gigabytes.

        Returns:
            Free VRAM in GB
        """
        try:
            stats = self.get_system_stats()
            if stats:
                devices = stats.get("devices", [])
                if devices:
                    vram_free = devices[0].get("vram_free", 0)
                    if vram_free > 0:
                        return vram_free / (1024**3)
        except Exception as e:
            logger.debug(f"Free VRAM detection failed: {e}")
        return 0.0

    def ensure_online(self):
        """
        Ensure ComfyUI is online, raising an exception if not.

        Raises:
            ComfyUIOfflineError: If ComfyUI is not running
        """
        if not self.is_online():
            raise ComfyUIOfflineError(url=self.base_url)

    def check_vram_available(self, required_gb: float, raise_on_insufficient: bool = False) -> bool:
        """
        Check if sufficient VRAM is available for an operation.

        Args:
            required_gb: Required VRAM in gigabytes
            raise_on_insufficient: If True, raise InsufficientVRAMError

        Returns:
            True if sufficient VRAM available

        Raises:
            InsufficientVRAMError: If raise_on_insufficient=True and not enough VRAM
        """
        from .exceptions import InsufficientVRAMError

        free_vram = self.get_free_vram_gb()

        if free_vram <= 0:
            # Can't detect free VRAM, assume it's ok
            logger.debug("Cannot detect free VRAM, assuming sufficient")
            return True

        if free_vram < required_gb:
            logger.warning(
                "Insufficient VRAM", extra={"required_gb": required_gb, "available_gb": free_vram}
            )
            if raise_on_insufficient:
                raise InsufficientVRAMError(required_gb=required_gb, available_gb=free_vram)
            return False

        logger.debug(
            "VRAM check passed", extra={"required_gb": required_gb, "available_gb": free_vram}
        )
        return True

    def estimate_vram_for_image(
        self, width: int = 1024, height: int = 1024, batch_size: int = 1
    ) -> float:
        """
        Estimate VRAM required for image generation.

        Based on empirical measurements with SDXL.

        Args:
            width: Image width
            height: Image height
            batch_size: Number of images to generate

        Returns:
            Estimated VRAM in GB
        """
        # Base model VRAM (SDXL fp16)
        base_vram = 4.0

        # Additional VRAM per megapixel
        megapixels = (width * height) / 1_000_000
        latent_vram = megapixels * 1.5

        # Batch size multiplier (not linear, ~60% per additional image)
        batch_multiplier = 1 + (batch_size - 1) * 0.6

        total = (base_vram + latent_vram) * batch_multiplier

        logger.debug(
            f"VRAM estimate for {width}x{height}",
            extra={"estimated_gb": total, "batch_size": batch_size},
        )
        return total

    def estimate_vram_for_video(
        self, width: int = 512, height: int = 512, frames: int = 16, model: str = "animatediff"
    ) -> float:
        """
        Estimate VRAM required for video generation.

        Args:
            width: Video width
            height: Video height
            frames: Number of frames
            model: Video model type

        Returns:
            Estimated VRAM in GB
        """
        # Base model VRAM
        base_vram = 4.0

        # Resolution component
        megapixels = (width * height) / 1_000_000
        resolution_vram = megapixels * 1.5

        # Temporal component (per frame)
        frame_vram = frames * 0.3

        # Model-specific multipliers
        model_multipliers = {
            # AnimateDiff family
            "animatediff": 1.0,
            "animatediff_v3": 1.1,
            "animatediff_lightning": 0.8,
            # SVD
            "svd": 1.3,
            "svd_xt": 1.3,
            # CogVideoX
            "cogvideo": 1.5,
            "cogvideox": 1.5,
            # Hunyuan (original)
            "hunyuan": 2.0,
            # v2.5.0: Hunyuan 1.5
            "hunyuan_15": 1.8,  # 720p FP16
            "hunyuan_15_fast": 1.4,  # Distilled, 6-step
            "hunyuan_15_i2v": 1.8,
            # v2.5.0: LTX-Video 2
            "ltxv": 1.3,  # Fast and efficient
            "ltxv_i2v": 1.4,
            # v2.5.0: Wan
            "wan": 1.0,  # 1.3B variant (very efficient)
            "wan_14b": 1.6,  # 14B FP8
            "wan_fast": 2.0,  # Dual model approach
            "wan_i2v": 1.4,
            # v2.5.0: Mochi
            "mochi": 2.2,  # 10B parameters
            "mochi_fp8": 1.6,
        }
        multiplier = model_multipliers.get(model.lower(), 1.0)

        total = (base_vram + resolution_vram + frame_vram) * multiplier

        logger.debug(
            f"VRAM estimate for {width}x{height}x{frames}f",
            extra={"estimated_gb": total, "model": model},
        )
        return total

    def recommend_image_preset(self, intent: str = "general") -> str:
        """
        Recommend an image generation preset based on detected VRAM.

        Args:
            intent: Generation intent (portrait, landscape, quality, fast)

        Returns:
            Recommended preset name
        """
        vram = self.get_vram_gb()

        if vram < 6:
            return "draft"
        elif vram < 8:
            return "fast"
        elif vram < 12:
            if intent == "portrait":
                return "portrait"
            elif intent == "landscape":
                return "landscape"
            return "quality"
        else:  # 12GB+
            if intent in ("cinematic", "film"):
                return "cinematic"
            return "hd"

    def recommend_video_preset(self, intent: str = "general") -> str:
        """
        Recommend a video generation preset based on detected VRAM.

        Args:
            intent: Generation intent (portrait, action, cinematic, quality)

        Returns:
            Recommended preset name
        """
        vram = self.get_vram_gb()

        try:
            from .video import get_recommended_preset

            return get_recommended_preset(intent=intent, vram_gb=vram)
        except ImportError:
            # Fallback if video module unavailable
            # v2.5.0: Updated recommendations with new models
            if vram < 8:
                return "quick"  # AnimateDiff Lightning
            elif vram < 12:
                return "wan_1.3b"  # Wan 1.3B is efficient
            elif vram < 16:
                return "ltx_standard"  # LTX-Video is great at 16GB
            elif vram < 24:
                return "hunyuan15_720p"  # Hunyuan 1.5 at 720p
            else:
                return "hunyuan15_quality"  # Full quality Hunyuan 1.5

    # =========================================================================
    # MODELS & INFO
    # =========================================================================

    def _get_object_info(self, node_type: str, input_name: str) -> list[str]:
        """Get input options for a node type."""
        try:
            response = self._get(f"/object_info/{node_type}")
            if response.ok:
                data = _safe_json_parse(response, f"getting object info for {node_type}")
                # Safe dictionary navigation with bounds checking
                node_data = data.get(node_type, {})
                if not isinstance(node_data, dict):
                    return []
                input_data = node_data.get("input", {})
                if not isinstance(input_data, dict):
                    return []
                required_data = input_data.get("required", {})
                if not isinstance(required_data, dict):
                    return []
                options = required_data.get(input_name, [])
                # Options should be a list with at least one element (the options list)
                if isinstance(options, list) and len(options) > 0:
                    first_element = options[0]
                    if isinstance(first_element, list):
                        return first_element
                return []
        except ComfyUIConnectionError:
            # JSON parse error - already logged
            pass
        except Exception as e:
            logger.debug(f"Failed to get object info for {node_type}: {e}")
        return []

    def get_checkpoints(self) -> list[str]:
        """Get available checkpoint models."""
        return self._get_object_info("CheckpointLoaderSimple", "ckpt_name")

    def get_samplers(self) -> list[str]:
        """Get available samplers."""
        samplers = self._get_object_info("KSampler", "sampler_name")
        return samplers or ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"]

    def get_schedulers(self) -> list[str]:
        """Get available schedulers."""
        schedulers = self._get_object_info("KSampler", "scheduler")
        return schedulers or ["normal", "karras", "exponential", "sgm_uniform"]

    def get_loras(self) -> list[str]:
        """Get available LoRA models."""
        return self._get_object_info("LoraLoader", "lora_name")

    def get_motion_models(self) -> list[str]:
        """Get available AnimateDiff motion models."""
        return self._get_object_info("ADE_LoadAnimateDiffModel", "model_name")

    def get_all_installed_nodes(self) -> list[str]:
        """
        Get all installed node types (class_types) in ComfyUI.

        This queries the /object_info endpoint to get all available nodes,
        which is useful for checking workflow dependencies.

        Returns:
            List of all installed node class_type names (e.g., ["KSampler", "CLIPTextEncode", ...])
        """
        try:
            response = self._get("/object_info")
            if response.ok:
                data = _safe_json_parse(response, "getting all object info")
                if isinstance(data, dict):
                    return list(data.keys())
        except ComfyUIConnectionError:
            # JSON parse error - already logged
            pass
        except Exception as e:
            logger.debug(f"Failed to get all installed nodes: {e}")
        return []

    def check_workflow_dependencies(self, workflow: dict) -> dict[str, Any]:
        """
        Check if all nodes in a workflow are installed in ComfyUI.

        Args:
            workflow: The ComfyUI workflow JSON (dict of node_id -> node_data)

        Returns:
            Dict with:
                - installed: List of installed node types used by the workflow
                - missing: List of missing node types
                - all_installed: bool indicating if all dependencies are met
                - details: Dict mapping class_type to list of node_ids using it
        """
        # Extract all class_types from the workflow
        workflow_nodes = {}
        for node_id, node in workflow.items():
            if isinstance(node, dict) and "class_type" in node:
                class_type = node["class_type"]
                if class_type not in workflow_nodes:
                    workflow_nodes[class_type] = []
                workflow_nodes[class_type].append(node_id)

        # Get all installed nodes
        installed_nodes = set(self.get_all_installed_nodes())

        # Check which are installed/missing
        installed = []
        missing = []

        for class_type in workflow_nodes:
            if class_type in installed_nodes:
                installed.append(class_type)
            else:
                missing.append(class_type)

        return {
            "installed": sorted(installed),
            "missing": sorted(missing),
            "all_installed": len(missing) == 0,
            "details": workflow_nodes,
            "total_nodes": sum(len(ids) for ids in workflow_nodes.values()),
            "unique_types": len(workflow_nodes),
        }

    # =========================================================================
    # QUEUE MANAGEMENT
    # =========================================================================

    def get_queue(self) -> dict:
        """Get current queue status."""
        try:
            response = self._get("/queue")
            if response.ok:
                return _safe_json_parse(response, "getting queue status")
        except ComfyUIConnectionError:
            # JSON parse error - already logged
            pass
        except Exception as e:
            logger.debug(f"Failed to get queue: {e}")
        return {"queue_running": [], "queue_pending": []}

    def get_history(self, prompt_id: str | None = None) -> dict:
        """Get execution history, optionally for a specific prompt."""
        try:
            endpoint = f"/history/{prompt_id}" if prompt_id else "/history"
            response = self._get(endpoint)
            if response.ok:
                return _safe_json_parse(response, "getting history")
        except ComfyUIConnectionError:
            # JSON parse error - already logged
            pass
        except Exception as e:
            logger.debug(f"Failed to get history: {e}")
        return {}

    def cancel_current(self) -> bool:
        """Cancel the currently running job."""
        try:
            response = self._post("/interrupt")
            if response.ok:
                logger.info("Cancelled current job")
                return True
        except Exception as e:
            logger.warning(f"Failed to cancel job: {e}")
        return False

    def clear_queue(self) -> bool:
        """Clear all pending jobs from the queue."""
        try:
            response = self._post("/queue", json={"clear": True})
            if response.ok:
                logger.info("Cleared queue")
                return True
        except Exception as e:
            logger.warning(f"Failed to clear queue: {e}")
        return False

    # =========================================================================
    # PROMPT EXECUTION
    # =========================================================================

    def queue_prompt(self, workflow: dict) -> str | None:
        """
        Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow dict

        Returns:
            prompt_id if successful, None otherwise

        Raises:
            QueueError: If queueing fails
        """
        # Input validation
        if not isinstance(workflow, dict):
            logger.error("Invalid workflow: must be a dictionary")
            return None

        try:
            payload = {"prompt": workflow, "client_id": self.client_id}
            response = self._post("/prompt", json=payload, timeout=settings.comfyui.timeout_queue)

            if response.ok:
                data = _safe_json_parse(response, "queueing prompt")
                prompt_id = data.get("prompt_id")
                if not isinstance(prompt_id, str):
                    logger.warning("Queue response missing prompt_id")
                    return None
                logger.info(
                    "Queued prompt", extra={"prompt_id": prompt_id[:8] if prompt_id else None}
                )
                return prompt_id
            else:
                error_msg = f"Queue failed with status {response.status_code}"
                logger.warning(error_msg, extra={"status": response.status_code})
                return None

        except ComfyUIConnectionError:
            raise
        except Exception as e:
            logger.error(f"Queue error: {e}", exc_info=True)
            return None

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float | None = None,
        poll_interval: float = 0.5,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> dict | None:
        """
        Wait for a prompt to complete.

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks
            on_progress: Optional callback called with (progress: 0.0-1.0, status: str)

        Returns:
            History entry when complete, or None on timeout
        """
        timeout = timeout or settings.generation.generation_timeout
        start = time.time()

        logger.debug("Waiting for completion", extra={"prompt_id": prompt_id[:8]})

        last_progress = 0.0

        while time.time() - start < timeout:
            try:
                elapsed = time.time() - start
                history = self.get_history(prompt_id)

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})

                    if status.get("completed", False):
                        if on_progress:
                            on_progress(1.0, "Completed")
                        logger.info(
                            "Generation completed",
                            extra={"prompt_id": prompt_id[:8], "elapsed": f"{elapsed:.1f}s"},
                        )
                        return entry

                    if status.get("status_str") == "error":
                        if on_progress:
                            on_progress(last_progress, "Error")
                        logger.warning(
                            "Generation failed",
                            extra={"prompt_id": prompt_id[:8], "status": status},
                        )
                        return entry

                    # Calculate progress from execution info
                    if on_progress:
                        # Try to get node progress from status messages
                        messages = status.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, list) and len(msg) >= 2:
                                if msg[0] == "execution_cached":
                                    # Some nodes were cached
                                    pass
                                elif msg[0] == "executing":
                                    msg[1] if len(msg) > 1 else None

                        # Estimate progress based on elapsed time
                        time_progress = min(0.95, elapsed / timeout)

                        # If we have execution info, use it
                        status_str = status.get("status_str", "processing")
                        if status_str == "queued":
                            progress = 0.05
                            status_msg = "Queued"
                        else:
                            progress = max(0.1, time_progress)
                            status_msg = f"Processing ({elapsed:.0f}s)"

                        if progress > last_progress:
                            last_progress = progress
                            on_progress(progress, status_msg)
                else:
                    # Not in history yet - still in queue
                    if on_progress:
                        queue = self.get_queue()
                        pending = queue.get("queue_pending", [])
                        running = queue.get("queue_running", [])

                        # Find position in queue
                        queue_pos = None
                        for i, item in enumerate(pending):
                            if isinstance(item, list) and len(item) > 1:
                                if item[1] == prompt_id:
                                    queue_pos = i + 1
                                    break

                        is_running = any(
                            isinstance(item, list) and len(item) > 1 and item[1] == prompt_id
                            for item in running
                        )

                        if is_running:
                            progress = 0.1
                            status_msg = "Starting"
                        elif queue_pos is not None:
                            progress = 0.02
                            status_msg = f"Queue position {queue_pos}"
                        else:
                            progress = 0.05
                            status_msg = "Waiting"

                        if progress > last_progress:
                            last_progress = progress
                            on_progress(progress, status_msg)

            except Exception as e:
                logger.debug(f"Poll error: {e}")

            time.sleep(poll_interval)

        logger.warning(
            "Generation timed out", extra={"prompt_id": prompt_id[:8], "timeout": timeout}
        )
        return None

    # =========================================================================
    # FILE DOWNLOADS
    # =========================================================================

    def get_image(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> bytes | None:
        """
        Download a generated image.

        Args:
            filename: Image filename
            subfolder: Subfolder within output directory
            folder_type: Folder type (usually "output")

        Returns:
            Image bytes or None if download fails
        """
        try:
            params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
            response = self._get("/view", params=params, timeout=settings.comfyui.timeout_image)
            if response.ok:
                logger.debug(f"Downloaded image: {filename}")
                return response.content
        except Exception as e:
            logger.warning(f"Failed to download image {filename}: {e}")
        return None

    def get_video(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> bytes | None:
        """Download a generated video."""
        try:
            params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
            response = self._get("/view", params=params, timeout=settings.comfyui.timeout_video)
            if response.ok:
                logger.debug(f"Downloaded video: {filename}")
                return response.content
        except Exception as e:
            logger.warning(f"Failed to download video {filename}: {e}")
        return None

    # =========================================================================
    # WORKFLOW BUILDERS
    # =========================================================================

    def build_txt2img_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        checkpoint: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        batch_size: int = 1,
    ) -> dict:
        """Build a basic txt2img workflow."""
        if seed == -1:
            seed = int(time.time() * 1000) % (2**32)

        if not checkpoint:
            checkpoints = self.get_checkpoints()
            checkpoint = checkpoints[0] if checkpoints else "model.safetensors"

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "seed": seed,
                    "steps": steps,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": batch_size, "height": height, "width": width},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": prompt},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt or "bad quality, blurry, distorted",
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "comfy_headless", "images": ["8", 0]},
            },
        }

    def build_video_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        checkpoint: str = "",
        motion_model: str = "",
        width: int = 512,
        height: int = 512,
        frames: int = 16,
        fps: int = 8,
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1,
        motion_scale: float = 1.0,
    ) -> dict:
        """Build an AnimateDiff video workflow."""
        if seed == -1:
            seed = int(time.time() * 1000) % (2**32)

        if not checkpoint:
            checkpoints = self.get_checkpoints()
            checkpoint = checkpoints[0] if checkpoints else "dreamshaper_8.safetensors"

        if not motion_model:
            motion_models = self.get_motion_models()
            motion_model = motion_models[0] if motion_models else "v3_sd15_mm.ckpt"

        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint},
            },
            "2": {
                "class_type": "ADE_LoadAnimateDiffModel",
                "inputs": {"model_name": motion_model},
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
                "inputs": {"width": width, "height": height, "batch_size": frames},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["1", 1]},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt or "bad quality, blurry, distorted",
                    "clip": ["1", 1],
                },
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["3", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["1", 2]},
            },
            "9": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": "comfy_headless_video",
                    "format": "video/h264-mp4",
                    "save_output": True,
                },
            },
        }

    # =========================================================================
    # HIGH-LEVEL GENERATION
    # =========================================================================

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        preset: str = "",
        checkpoint: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        wait: bool = True,
        timeout: float | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        High-level image generation with optional preset support.

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            preset: Optional preset (draft, fast, quality, hd, portrait, landscape, cinematic)
                   When set, overrides width/height/steps/cfg with preset values
            checkpoint: Model checkpoint name
            width: Image width (ignored if preset is set)
            height: Image height (ignored if preset is set)
            steps: Sampling steps (ignored if preset is set)
            cfg: CFG scale (ignored if preset is set)
            sampler: Sampler name
            scheduler: Scheduler name
            seed: Random seed (-1 for random)
            wait: Whether to wait for completion
            timeout: Generation timeout
            on_progress: Optional callback(progress: 0.0-1.0, status: str) for progress updates

        Returns:
            Dict with success, prompt_id, images, error, seed, preset
        """
        request_id = str(uuid.uuid4())[:8]
        timeout = timeout or settings.generation.generation_timeout

        with LogContext(request_id):
            result = {
                "success": False,
                "prompt_id": None,
                "images": [],
                "error": None,
                "seed": seed,
                "preset": preset or None,
            }

            logger.info(
                "Starting image generation",
                extra={"width": width, "height": height, "steps": steps, "preset": preset},
            )

            try:
                self.ensure_online()
            except ComfyUIOfflineError as e:
                result["error"] = str(e)
                return result

            # Try using WorkflowCompiler if preset is specified
            if preset:
                try:
                    from .workflows import GENERATION_PRESETS, compile_workflow

                    if preset in GENERATION_PRESETS:
                        compiled = compile_workflow(
                            prompt=prompt,
                            negative=negative_prompt,
                            preset=preset,
                            checkpoint=checkpoint or "auto",
                            sampler=sampler,
                            scheduler=scheduler,
                            seed=seed,
                        )
                        if compiled.is_valid:
                            workflow = compiled.workflow
                            # Extract seed from compiled workflow (safe nested access)
                            result["seed"] = _safe_get_nested(
                                workflow, "3", "inputs", "seed", default=seed
                            )
                            logger.debug(f"Using WorkflowCompiler with preset '{preset}'")
                        else:
                            logger.warning(f"Workflow compilation errors: {compiled.errors}")
                            preset = ""  # Fall back to legacy
                    else:
                        logger.warning(f"Unknown preset '{preset}', falling back to legacy")
                        preset = ""
                except Exception as e:
                    logger.warning(f"WorkflowCompiler failed: {e}, using legacy builder")
                    preset = ""

            # Legacy workflow builder (when no preset or compiler failed)
            if not preset:
                workflow = self.build_txt2img_workflow(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    checkpoint=checkpoint,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=cfg,
                    sampler=sampler,
                    scheduler=scheduler,
                    seed=seed,
                )
                # Store actual seed used (safe nested access)
                result["seed"] = _safe_get_nested(workflow, "3", "inputs", "seed", default=seed)

            prompt_id = self.queue_prompt(workflow)
            if not prompt_id:
                result["error"] = "Failed to queue prompt"
                return result

            result["prompt_id"] = prompt_id

            if not wait:
                result["success"] = True
                return result

            history = self.wait_for_completion(prompt_id, timeout=timeout, on_progress=on_progress)
            if not history:
                result["error"] = f"Generation timed out after {timeout}s"
                return result

            status = history.get("status", {})
            if status.get("status_str") == "error":
                error_msgs = status.get("messages", [["Unknown error"]])
                result["error"] = str(error_msgs[0] if error_msgs else "Unknown error")
                return result

            # Extract images (with type validation)
            outputs = history.get("outputs", {})
            if isinstance(outputs, dict):
                for node_output in outputs.values():
                    if isinstance(node_output, dict) and "images" in node_output:
                        images_list = node_output["images"]
                        if isinstance(images_list, list):
                            for img in images_list:
                                if isinstance(img, dict):
                                    result["images"].append(
                                        {
                                            "filename": img.get("filename"),
                                            "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output"),
                                        }
                                    )

            result["success"] = len(result["images"]) > 0

            if result["success"]:
                logger.info("Generation complete", extra={"image_count": len(result["images"])})
            else:
                logger.warning("Generation produced no images")

            return result

    def generate_batch(
        self,
        prompts: list[str],
        negative_prompt: str = "",
        preset: str = "fast",
        checkpoint: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seeds: list[int] | None = None,
        max_concurrent: int = 1,
        check_vram: bool = True,
        on_progress: Callable[[int, int, float, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Generate multiple images from a list of prompts.

        Args:
            prompts: List of prompts to generate
            negative_prompt: Shared negative prompt
            preset: Generation preset
            checkpoint: Model checkpoint
            width, height, steps, cfg, sampler, scheduler: Generation params
            seeds: Optional list of seeds (one per prompt, -1 for random)
            max_concurrent: Max concurrent generations (use 1 for sequential)
            check_vram: If True, check VRAM before starting
            on_progress: Callback(current_idx, total, progress, status)

        Returns:
            Dict with success, results (list of individual results), errors
        """
        import time as time_module

        total = len(prompts)
        if total == 0:
            return {"success": False, "results": [], "errors": ["No prompts provided"]}

        # Prepare seeds
        if seeds is None:
            seeds = [-1] * total
        elif len(seeds) < total:
            seeds = seeds + [-1] * (total - len(seeds))

        # Check VRAM if requested
        if check_vram:
            estimated = self.estimate_vram_for_image(width, height, max_concurrent)
            if not self.check_vram_available(estimated):
                logger.warning(
                    "Batch may exceed VRAM", extra={"estimated_gb": estimated, "batch_size": total}
                )

        results = []
        errors = []
        start_time = time_module.time()

        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            try:
                # Progress callback
                if on_progress:
                    on_progress(idx, total, 0.0, f"Starting {idx + 1}/{total}")

                # Wrap individual progress
                def item_progress(prog: float, status: str):
                    if on_progress:
                        overall = (idx + prog) / total
                        on_progress(idx, total, overall, f"[{idx + 1}/{total}] {status}")

                result = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    preset=preset,
                    checkpoint=checkpoint,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=cfg,
                    sampler=sampler,
                    scheduler=scheduler,
                    seed=seed,
                    wait=True,
                    on_progress=item_progress,
                )

                results.append(result)

                if not result["success"]:
                    errors.append(f"Prompt {idx}: {result.get('error', 'Unknown error')}")

                # Final progress for this item
                if on_progress:
                    overall = (idx + 1) / total
                    status = "Complete" if result["success"] else "Failed"
                    on_progress(idx, total, overall, f"[{idx + 1}/{total}] {status}")

            except Exception as e:
                logger.error(f"Batch item {idx} failed: {e}")
                errors.append(f"Prompt {idx}: {str(e)}")
                results.append(
                    {
                        "success": False,
                        "error": str(e),
                        "images": [],
                        "prompt_id": None,
                        "seed": seed,
                    }
                )

        elapsed = time_module.time() - start_time
        success_count = sum(1 for r in results if r.get("success", False))

        logger.info(
            "Batch complete",
            extra={
                "total": total,
                "success": success_count,
                "failed": total - success_count,
                "elapsed": f"{elapsed:.1f}s",
            },
        )

        return {
            "success": success_count == total,
            "results": results,
            "errors": errors,
            "total": total,
            "success_count": success_count,
            "elapsed_seconds": elapsed,
        }

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        preset: str = "standard",
        init_image: str | None = None,
        wait: bool = True,
        timeout: float | None = None,
        on_progress: Callable[[float, str], None] | None = None,
        # Override individual settings (backwards compatible)
        checkpoint: str = "",
        motion_model: str = "",
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int = -1,
        motion_scale: float | None = None,
    ) -> dict[str, Any]:
        """
        High-level video generation using multi-model VideoWorkflowBuilder.

        Supports AnimateDiff (v2/v3/Lightning), SVD, CogVideoX, and Hunyuan.

        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid
            preset: Video preset (quick, standard, quality, cinematic, portrait,
                   action, svd_short, svd_long, cogvideo, hunyuan, hunyuan_fast)
            init_image: Base64 image for img2vid models (SVD)
            wait: Whether to wait for completion
            timeout: Generation timeout
            on_progress: Optional callback(progress: 0.0-1.0, status: str) for progress updates
            # Override any preset settings:
            checkpoint: Model checkpoint (for AnimateDiff)
            width, height, frames, fps, steps, cfg, seed, motion_scale

        Returns:
            Dict with success, prompt_id, videos, error, seed, preset
        """
        request_id = str(uuid.uuid4())[:8]
        timeout = timeout or settings.generation.video_timeout

        with LogContext(request_id):
            result = {
                "success": False,
                "prompt_id": None,
                "videos": [],
                "error": None,
                "seed": seed,
                "preset": preset,
            }

            logger.info(
                "Starting video generation", extra={"preset": preset, "prompt_length": len(prompt)}
            )

            try:
                self.ensure_online()
            except ComfyUIOfflineError as e:
                result["error"] = str(e)
                return result

            # Build workflow using VideoWorkflowBuilder
            try:
                from .video import build_video_workflow

                # Build overrides dict from non-None parameters
                overrides = {}
                if checkpoint:
                    overrides["checkpoint"] = checkpoint
                if width is not None:
                    overrides["width"] = width
                if height is not None:
                    overrides["height"] = height
                if frames is not None:
                    overrides["frames"] = frames
                if fps is not None:
                    overrides["fps"] = fps
                if steps is not None:
                    overrides["steps"] = steps
                if cfg is not None:
                    overrides["cfg"] = cfg
                if seed != -1:
                    overrides["seed"] = seed
                if motion_scale is not None:
                    overrides["motion_scale"] = motion_scale

                workflow = build_video_workflow(
                    prompt=prompt,
                    negative=negative_prompt or "ugly, blurry, low quality, distorted",
                    preset=preset,
                    init_image=init_image,
                    **overrides,
                )

                # Extract actual seed from workflow (video.py generates random if -1)
                # Find KSampler node and extract seed (safe access)
                if isinstance(workflow, dict):
                    for node in workflow.values():
                        if isinstance(node, dict) and node.get("class_type") in (
                            "KSampler",
                            "HunyuanVideoSampler",
                        ):
                            result["seed"] = _safe_get_nested(node, "inputs", "seed", default=seed)
                            break

            except Exception as e:
                logger.warning(f"VideoWorkflowBuilder failed, falling back to legacy: {e}")
                # Fallback to legacy build_video_workflow method
                workflow = self.build_video_workflow(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    checkpoint=checkpoint,
                    motion_model=motion_model,
                    width=width or 512,
                    height=height or 512,
                    frames=frames or 16,
                    fps=fps or 8,
                    steps=steps or 20,
                    cfg=cfg or 7.0,
                    seed=seed,
                    motion_scale=motion_scale or 1.0,
                )
                # Safe nested access for legacy workflow
                result["seed"] = _safe_get_nested(workflow, "7", "inputs", "seed", default=seed)

            prompt_id = self.queue_prompt(workflow)
            if not prompt_id:
                result["error"] = "Failed to queue prompt"
                return result

            result["prompt_id"] = prompt_id

            if not wait:
                result["success"] = True
                return result

            history = self.wait_for_completion(prompt_id, timeout=timeout, on_progress=on_progress)
            if not history:
                result["error"] = f"Generation timed out after {timeout}s"
                return result

            status = history.get("status", {})
            if status.get("status_str") == "error":
                error_msgs = status.get("messages", [["Unknown error"]])
                result["error"] = str(error_msgs[0] if error_msgs else "Unknown error")
                return result

            # Extract videos (check both 'gifs' and 'videos' keys, with type validation)
            outputs = history.get("outputs", {})
            if isinstance(outputs, dict):
                for node_output in outputs.values():
                    if isinstance(node_output, dict):
                        for key in ("gifs", "videos"):
                            if key in node_output:
                                video_list = node_output[key]
                                if isinstance(video_list, list):
                                    for vid in video_list:
                                        if isinstance(vid, dict):
                                            result["videos"].append(
                                                {
                                                    "filename": vid.get("filename"),
                                                    "subfolder": vid.get("subfolder", ""),
                                                    "type": vid.get("type", "output"),
                                                }
                                            )

            result["success"] = len(result["videos"]) > 0

            if result["success"]:
                logger.info(
                    "Video generation complete", extra={"video_count": len(result["videos"])}
                )

            return result
