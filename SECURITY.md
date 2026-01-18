# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.5.x   | :white_check_mark: |
| < 2.5   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues by emailing:
- **Email**: [Create a private security advisory](https://github.com/mcp-tool-shop/comfy-headless/security/advisories/new) on GitHub

### What to Include

When reporting a vulnerability, please include:

1. **Description** - Clear description of the vulnerability
2. **Impact** - What could an attacker achieve?
3. **Reproduction steps** - How to reproduce the issue
4. **Affected versions** - Which versions are impacted
5. **Suggested fix** - If you have one (optional)

### What to Expect

1. **Acknowledgment** - We'll acknowledge receipt within 48 hours
2. **Assessment** - We'll assess severity and impact within 7 days
3. **Fix timeline** - Critical issues: 7 days, High: 14 days, Medium: 30 days
4. **Disclosure** - We'll coordinate disclosure timing with you
5. **Credit** - We'll credit you in the security advisory (unless you prefer anonymity)

## Security Best Practices

When using comfy-headless:

### Configuration

```python
# Use environment variables for sensitive config
import os
os.environ["COMFY_HEADLESS_COMFYUI__URL"] = "http://localhost:8188"

# Never hardcode credentials
# BAD: client = ComfyClient(api_key="sk-12345")
# GOOD: Use environment variables or secrets manager
```

### Network Security

- Run ComfyUI on localhost or trusted networks only
- Use HTTPS/WSS in production environments
- Consider using a reverse proxy with authentication

### Input Validation

comfy-headless validates inputs by default, but ensure:
- User-provided prompts are sanitized before display
- File paths don't allow directory traversal
- Dimensions are within reasonable bounds

### Secrets

```python
from comfy_headless.secrets_manager import SecretValue

# Secrets are automatically masked in logs
api_key = SecretValue("sk-12345")
print(api_key)  # Output: SecretValue(****)
```

## Known Security Considerations

### WebSocket Connections

- WebSocket connections to ComfyUI are not encrypted by default
- Use WSS (WebSocket Secure) in production
- Client IDs are UUIDs, not authentication tokens

### Temporary Files

- Generated images are stored in temp directories
- Automatic cleanup runs periodically
- Set `COMFY_HEADLESS_UI__CLEANUP_INTERVAL` for custom cleanup

### Logging

- Debug logs may contain sensitive information
- Set `COMFY_HEADLESS_LOG_LEVEL=INFO` or higher in production
- Secrets are masked but avoid logging user data unnecessarily

## Security Updates

Security updates are released as patch versions (e.g., 2.5.1 -> 2.5.2).

To stay updated:
1. Watch this repository for releases
2. Use `pip install --upgrade comfy-headless`
3. Subscribe to security advisories on GitHub
