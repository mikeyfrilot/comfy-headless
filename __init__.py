"""
Comfy Headless - Headless ComfyUI Interface
============================================

A clean, standalone interface for ComfyUI that makes complex
workflows accessible through simple parameters and presets.

v2.5.0: Modular architecture with optional features

Installation:
    pip install comfy-headless              # Core only (minimal)
    pip install comfy-headless[ai]          # + Ollama intelligence
    pip install comfy-headless[websocket]   # + WebSocket real-time updates
    pip install comfy-headless[ui]          # + Gradio web UI
    pip install comfy-headless[standard]    # ai + websocket (recommended)
    pip install comfy-headless[full]        # Everything

Features (Core):
- Text-to-image generation
- Multi-model video generation (AnimateDiff, SVD, CogVideoX, Hunyuan, LTX, Wan)
- Template-based workflow compilation
- VRAM-aware optimization
- Circuit breaker pattern with tenacity retry
- Structured logging

Features (Optional):
- [ai] AI-powered prompt analysis and enhancement via Ollama
- [websocket] Real-time progress via WebSocket
- [ui] Gradio web interface
- [health] System health monitoring
- [validation] Pydantic configuration validation
- [observability] OpenTelemetry distributed tracing

Usage:
    # Core functionality (always available)
    from comfy_headless import ComfyClient
    client = ComfyClient()
    result = client.generate_image("a beautiful sunset")

    # Video generation
    from comfy_headless import VIDEO_PRESETS
    result = client.generate_video("a cat walking", preset="ltx_quality")

    # AI features (requires [ai] extra)
    from comfy_headless import analyze_prompt, enhance_prompt
    analysis = analyze_prompt("a cyberpunk city at night")

    # WebSocket (requires [websocket] extra)
    from comfy_headless import ComfyWSClient
    async with ComfyWSClient() as ws:
        result = await ws.wait_for_completion(prompt_id)

    # UI (requires [ui] extra)
    from comfy_headless import launch
    launch()
"""

# Feature flags (detect available optional features)
# Configuration (import first - other modules depend on it)
from .config import (
    HttpConfig,
    Settings,
    get_settings,
    get_temp_dir,
    reload_settings,
    settings,
)

# Exceptions (with verbosity levels)
from .exceptions import (
    CircuitOpenError,
    ComfyHeadlessError,
    ComfyHeadlessExceptionGroup,
    ComfyUIConnectionError,
    ComfyUIOfflineError,
    ErrorLevel,
    GenerationFailedError,
    GenerationTimeoutError,
    InvalidParameterError,
    InvalidPromptError,
    OllamaConnectionError,
    QueueError,
    Result,
    RetryExhaustedError,
    SecurityError,
    TemplateNotFoundError,
    ValidationError,
    VerbosityLevel,
    WorkflowCompilationError,
    format_error_for_user,
    get_verbosity,
    set_verbosity,
)
from .feature_flags import (
    FEATURES,
    FeatureNotAvailable,
    check_feature,
    get_install_hint,
    list_available_features,
    list_missing_features,
    require_feature,
)

# HTTP Client (httpx-based) - requires [ai] extra for async client
from .http_client import (
    HttpClient,
    close_all_clients,
    get_http_client,
)

# Logging (with OpenTelemetry support)
from .logging_config import (
    LogContext,
    clear_request_id,
    get_logger,
    get_tracer,
    log_timing,
    set_log_level,
    set_request_id,
    traced_operation,
)

# Retry and resilience (tenacity-based)
from .retry import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    OperationTimeoutError,
    RateLimiter,
    async_timeout,
    circuit_registry,
    get_circuit_breaker,
    retry_async,
    retry_on_exception,
    retry_with_backoff,
    with_timeout,
)

# Async HTTP client only if httpx is available
if FEATURES["ai"]:
    from .http_client import (
        AsyncHttpClient,
        get_async_http_client,
    )
else:
    AsyncHttpClient = None
    get_async_http_client = None

# Health checks - requires [health] extra for full functionality
# Cleanup
from .cleanup import (
    CleanupThread,
    TempFileManager,
    cleanup_all,
    cleanup_temp_files,
    get_temp_manager,
    register_cleanup_callback,
    register_shutdown_handlers,
    save_temp_image,
    save_temp_video,
)

# Client
from .client import ComfyClient
from .health import (
    ComponentHealth,
    HealthChecker,
    HealthMonitor,
    HealthReport,
    HealthStatus,
    check_health,
    full_health_check,
    get_health_checker,
    is_healthy,
)

# WebSocket Client (async, real-time progress)
try:
    from .websocket_client import (
        WEBSOCKETS_AVAILABLE,
        ComfyWSClient,
        WSMessageType,
        WSProgress,
    )
except ImportError:
    # websockets not installed
    WEBSOCKETS_AVAILABLE = False
    ComfyWSClient = None
    WSProgress = None
    WSMessageType = None

# Intelligence module (v2.4: caching, few-shot, A/B testing)
# Requires [ai] extra for full functionality (Ollama via httpx)
if FEATURES["ai"]:
    from .intelligence import (
        CHAIN_OF_THOUGHT_TEMPLATE,
        # v2.4: Few-shot examples
        FEW_SHOT_ENHANCEMENT_EXAMPLES,
        EnhancedPrompt,
        PromptABTester,
        PromptAnalysis,
        # v2.4: Caching
        PromptCache,
        PromptIntelligence,
        # v2.4: Versioning and A/B testing
        PromptVersion,
        analyze_prompt,
        enhance_prompt,
        get_few_shot_examples,
        get_few_shot_prompt,
        get_intelligence,
        get_prompt_cache,
        quick_enhance,
    )
    from .intelligence import (
        sanitize_prompt as sanitize_prompt_ai,  # AI-specific sanitizer
    )
else:
    # Stubs for when AI feature is not installed
    PromptIntelligence = None
    PromptAnalysis = None
    EnhancedPrompt = None
    get_intelligence = None
    analyze_prompt = None
    enhance_prompt = None
    quick_enhance = None
    sanitize_prompt_ai = None
    PromptCache = None
    get_prompt_cache = None
    PromptVersion = None
    PromptABTester = None
    FEW_SHOT_ENHANCEMENT_EXAMPLES = None
    get_few_shot_prompt = None
    get_few_shot_examples = None
    CHAIN_OF_THOUGHT_TEMPLATE = None

# Workflow system (v2.4: versioning, caching, DAG validation, snapshots)
# Help system
from .help_system import (
    HelpLevel,
    HelpRegistry,
    HelpTopic,
    format_help_list,
    format_quick_help,
    get_api_reference,
    get_command_help,
    get_help,
    get_help_for_error,
    get_help_level,
    list_topics,
    search_help,
    set_help_level,
)

# Secrets management
from .secrets_manager import (
    SecretsManager,
    SecretValue,
    generate_api_key,
    generate_token,
    get_secret,
    get_secret_str,
    get_secrets_manager,
    hash_secret,
    mask_url_credentials,
    redact_dict,
    verify_hashed_secret,
)

# Validation (Pydantic-based)
from .validation import (
    clamp_dimensions,
    sanitize_prompt,
    validate_choice,
    validate_dimensions,
    validate_generation_params,
    validate_in_range,
    validate_path,
    validate_prompt,
    validated_dimensions,
    validated_prompt,
)

# Video module
from .video import (
    VIDEO_MODEL_INFO,
    VIDEO_PRESETS,
    MotionStyle,
    VideoModel,
    VideoSettings,
    VideoWorkflowBuilder,
    build_video_workflow,
    get_recommended_preset,
    get_video_builder,
    list_video_presets,
)
from .workflows import (
    GENERATION_PRESETS,
    # v2.4: DAG validation
    DAGValidator,
    # v2.4: Snapshot management
    SnapshotManager,
    TemplateLibrary,
    # v2.4: Caching
    WorkflowCache,
    WorkflowCompiler,
    WorkflowSnapshot,
    WorkflowTemplate,
    # v2.4: Versioning
    WorkflowVersion,
    compile_workflow,
    compute_workflow_hash,
    get_compiler,
    get_library,
    get_snapshot_manager,
    get_workflow_cache,
    list_presets,
    validate_workflow_dag,
)

__version__ = "2.5.1"

# Lazy loading for optional features with helpful error messages
_LAZY_IMPORTS = {
    # AI features
    "PromptIntelligence": ("ai", ".intelligence"),
    "analyze_prompt": ("ai", ".intelligence"),
    "enhance_prompt": ("ai", ".intelligence"),
    "quick_enhance": ("ai", ".intelligence"),
    # WebSocket features
    "ComfyWSClient": ("websocket", ".websocket_client"),
    # UI features
    "launch": ("ui", None),  # Special handling
    "create_comfy_theme": ("ui", ".theme"),
    "get_theme_info": ("ui", ".theme"),
    # Health features
    "HealthMonitor": ("health", ".health"),
}


def __getattr__(name: str):
    """
    Lazy loading for optional features with helpful error messages.

    When a user tries to import an optional feature that isn't installed,
    this provides a clear error message with installation instructions.
    """
    if name in _LAZY_IMPORTS:
        feature, module = _LAZY_IMPORTS[name]
        if not FEATURES.get(feature, False):
            hint = get_install_hint(feature)
            raise ImportError(f"'{name}' requires the [{feature}] feature. Install with: {hint}")

        # Special handling for launch
        if name == "launch":

            def _launch(port: int = 7861, share: bool = False):
                from .ui import create_ui

                app = create_ui()
                app.launch(
                    server_name="0.0.0.0",
                    server_port=port,
                    share=share,
                    inbrowser=True,
                )

            return _launch

        # Import the actual object
        if module:
            import importlib

            mod = importlib.import_module(module, __package__)
            return getattr(mod, name)

    raise AttributeError(f"module 'comfy_headless' has no attribute '{name}'")


__all__ = [
    # Feature flags
    "FEATURES",
    "check_feature",
    "require_feature",
    "get_install_hint",
    "list_available_features",
    "list_missing_features",
    "FeatureNotAvailable",
    # Configuration
    "Settings",
    "settings",
    "get_temp_dir",
    "get_settings",
    "reload_settings",
    "HttpConfig",
    # Exceptions (with verbosity)
    "ComfyHeadlessError",
    "ComfyUIConnectionError",
    "ComfyUIOfflineError",
    "OllamaConnectionError",
    "QueueError",
    "GenerationTimeoutError",
    "GenerationFailedError",
    "WorkflowCompilationError",
    "TemplateNotFoundError",
    "RetryExhaustedError",
    "CircuitOpenError",
    "ValidationError",
    "InvalidPromptError",
    "InvalidParameterError",
    "SecurityError",
    "Result",
    "ErrorLevel",
    "VerbosityLevel",
    "ComfyHeadlessExceptionGroup",
    "format_error_for_user",
    "set_verbosity",
    "get_verbosity",
    # Logging (with OpenTelemetry)
    "get_logger",
    "set_log_level",
    "set_request_id",
    "clear_request_id",
    "LogContext",
    "log_timing",
    "traced_operation",
    "get_tracer",
    # Retry and resilience (tenacity-based)
    "retry_with_backoff",
    "retry_on_exception",
    "retry_async",
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerRegistry",
    "circuit_registry",
    "get_circuit_breaker",
    "RateLimiter",
    "OperationTimeoutError",
    "with_timeout",
    "async_timeout",
    # HTTP Client (httpx)
    "HttpClient",
    "AsyncHttpClient",
    "get_http_client",
    "get_async_http_client",
    "close_all_clients",
    # Health
    "HealthStatus",
    "ComponentHealth",
    "HealthReport",
    "HealthChecker",
    "HealthMonitor",
    "check_health",
    "full_health_check",
    "is_healthy",
    "get_health_checker",
    # Cleanup
    "TempFileManager",
    "CleanupThread",
    "get_temp_manager",
    "cleanup_temp_files",
    "cleanup_all",
    "register_shutdown_handlers",
    "register_cleanup_callback",
    "save_temp_image",
    "save_temp_video",
    # Client
    "ComfyClient",
    "launch",
    # WebSocket Client
    "ComfyWSClient",
    "WSProgress",
    "WSMessageType",
    "WEBSOCKETS_AVAILABLE",
    # Intelligence (v2.4: caching, few-shot, A/B testing)
    "PromptIntelligence",
    "PromptAnalysis",
    "EnhancedPrompt",
    "get_intelligence",
    "analyze_prompt",
    "enhance_prompt",
    "quick_enhance",
    "PromptCache",
    "get_prompt_cache",
    "PromptVersion",
    "PromptABTester",
    "FEW_SHOT_ENHANCEMENT_EXAMPLES",
    "get_few_shot_prompt",
    "get_few_shot_examples",
    "CHAIN_OF_THOUGHT_TEMPLATE",
    # Workflows (v2.4: versioning, caching, DAG validation, snapshots)
    "WorkflowCompiler",
    "WorkflowTemplate",
    "TemplateLibrary",
    "get_compiler",
    "get_library",
    "compile_workflow",
    "GENERATION_PRESETS",
    "list_presets",
    "WorkflowVersion",
    "WorkflowSnapshot",
    "SnapshotManager",
    "get_snapshot_manager",
    "WorkflowCache",
    "get_workflow_cache",
    "compute_workflow_hash",
    "DAGValidator",
    "validate_workflow_dag",
    # Video
    "VideoWorkflowBuilder",
    "VideoSettings",
    "VideoModel",
    "MotionStyle",
    "get_video_builder",
    "VIDEO_PRESETS",
    "VIDEO_MODEL_INFO",
    "build_video_workflow",
    "list_video_presets",
    "get_recommended_preset",
    # Validation
    "validate_prompt",
    "sanitize_prompt",
    "validate_dimensions",
    "clamp_dimensions",
    "validate_path",
    "validate_in_range",
    "validate_choice",
    "validate_generation_params",
    "validated_prompt",
    "validated_dimensions",
    # Secrets
    "SecretValue",
    "SecretsManager",
    "get_secret",
    "get_secret_str",
    "get_secrets_manager",
    "generate_token",
    "generate_api_key",
    "hash_secret",
    "verify_hashed_secret",
    "mask_url_credentials",
    "redact_dict",
    # Help
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


# Note: launch() is now handled via __getattr__ for lazy loading
# This allows for helpful error messages when [ui] extra is not installed
