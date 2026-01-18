# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.5.1] - 2026-01-18

### Security
- **WebSocket DoS Protection**: Added `max_message_size` limit (1MB default) to prevent memory exhaustion
- **WebSocket Encryption Warning**: Now logs warning when using unencrypted `ws://` connections
- **Listener Limit**: Added `MAX_LISTENERS_PER_PROMPT` (100) to prevent memory exhaustion attacks
- **Gradio Minimum Version**: Updated to `>=5.6.0` to include fix for CVE-2025-23042 (path traversal)
- **Default Host Changed**: UI now defaults to `127.0.0.1` (localhost) instead of `0.0.0.0` to prevent accidental network exposure
- Added `SECURITY_AUDIT.md` documenting security review findings

### Fixed
- Version mismatch between pyproject.toml and code modules
- Updated WebSocket client to use modern `websockets.asyncio.client` API (fixes deprecation warnings)

### Added
- GitHub Actions CI/CD workflows for automated testing and PyPI publishing
- CHANGELOG.md for version history tracking
- CONTRIBUTING.md with contribution guidelines
- SECURITY.md with vulnerability reporting process
- `SECURITY_AUDIT.md` - Web UI security audit report
- `ERROR_HANDLING_AUDIT.md` - Exception handling audit report
- `GENERATION_AUDIT.md` - Image/Video/AI module audit report

### Improved
- Narrowed exception catches in `cleanup.py` from broad `Exception` to specific types

### Changed
- Moved archive folder outside main package directory
- WebSocket client constructor now accepts `max_message_size` parameter

## [2.5.0] - 2026-01-16

### Added
- Initial public release
- Modular architecture with feature-gating for optional dependencies
- Core client with connection pooling and retry logic
- WebSocket client for real-time progress updates
- AI-powered prompt intelligence via Ollama integration
- Video generation support with 11 preset models
- Gradio 5.0+ UI with 7 tabs and Ocean Mist theme
- Comprehensive exception hierarchy with user-friendly messages
- Circuit breaker pattern for resilience
- Structured logging with OpenTelemetry support
- Input validation and sanitization utilities
- Secrets management utilities
- Temporary file cleanup system
- Context-aware help system

### Features by Install Extra
- `[ai]` - Ollama prompt enhancement with A/B testing
- `[websocket]` - Real-time progress via WebSocket
- `[health]` - System health monitoring with psutil
- `[ui]` - Full Gradio web interface
- `[validation]` - Pydantic-based config validation
- `[observability]` - OpenTelemetry distributed tracing
- `[standard]` - ai + websocket bundle
- `[full]` - All features enabled

### Supported Video Models
- Wan 2.1 (14B, 1.3B variants)
- Hunyuan Video
- LTX Video
- Mochi
- CogVideoX (5B, 2B)
- AnimateDiff
- Stable Video Diffusion
- Custom workflow support

[Unreleased]: https://github.com/mcp-tool-shop/comfy-headless/compare/v2.5.1...HEAD
[2.5.1]: https://github.com/mcp-tool-shop/comfy-headless/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/mcp-tool-shop/comfy-headless/releases/tag/v2.5.0
