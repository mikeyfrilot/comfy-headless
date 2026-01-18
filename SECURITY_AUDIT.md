# Security Audit Report - Comfy Headless v2.5.1

**Date:** January 18, 2026
**Auditor:** Claude Security Review
**Scope:** Web UI, WebSocket, Input Validation, Configuration

---

## Executive Summary

**Overall Security Rating: B+ (Good with recommendations)**

The codebase demonstrates strong security foundations with comprehensive input validation, secrets management, and error handling. Key areas for improvement include WebSocket security hardening and Gradio configuration.

---

## Findings by Severity

### CRITICAL (0 Found)
None identified.

### HIGH (2 Found)

#### H1: WebSocket Uses Unencrypted Connection by Default
**File:** `websocket_client.py:131-136`
**Issue:** When base_url doesn't specify protocol, defaults to `ws://` (unencrypted)
**Risk:** Man-in-the-middle attacks can intercept progress data and potentially inject malicious messages
**OWASP:** A02:2021 - Cryptographic Failures

```python
# Current (insecure default)
else:
    self.ws_url = f"ws://{http_url}/ws"
```

**Recommendation:** Log a warning when using unencrypted WebSocket, document security implications.

#### H2: No WebSocket Message Size Limits
**File:** `websocket_client.py:165-170`
**Issue:** No `max_size` parameter in `websockets.connect()`, allowing unlimited message sizes
**Risk:** Denial of Service via memory exhaustion from large messages
**OWASP:** A05:2021 - Security Misconfiguration

**Recommendation:** Add `max_size=1_048_576` (1MB) to prevent DoS.

---

### MEDIUM (4 Found)

#### M1: Gradio Binds to 0.0.0.0 by Default
**File:** `config.py:135`, `ui.py:2359`
**Issue:** UI listens on all interfaces by default
**Risk:** Exposes UI to network without authentication
**OWASP:** A01:2021 - Broken Access Control

**Recommendation:** Default to `127.0.0.1` for local-only access, document network exposure.

#### M2: No Rate Limiting on WebSocket Listeners
**File:** `websocket_client.py:344-355`
**Issue:** Unlimited listeners can be added per prompt_id
**Risk:** Memory exhaustion from listener accumulation

**Recommendation:** Add maximum listener count per prompt_id.

#### M3: No Origin Validation on WebSocket
**File:** `websocket_client.py:165`
**Issue:** No `origin` parameter validation in connect()
**Risk:** Cross-Site WebSocket Hijacking (CSWSH)
**Note:** Lower risk since this is a client library connecting to ComfyUI

#### M4: Gradio Version Should Be Pinned Higher
**File:** `pyproject.toml:57`
**Issue:** `gradio>=5.0.0` allows vulnerable versions
**Risk:** CVE-2025-23042 (path traversal), CVE-2025-0187 (DoS) affect versions < 5.6.0

**Recommendation:** Update to `gradio>=5.6.0` minimum.

---

### LOW (3 Found)

#### L1: Debug Information in Error Logs
**File:** `websocket_client.py:254-261`
**Issue:** Message preview logged on JSON decode error
**Risk:** Potential information disclosure in logs

#### L2: Global Client Instance
**File:** `ui.py:37`
**Issue:** `client = ComfyClient()` at module level
**Risk:** Shared state across requests in multi-user scenarios

#### L3: Hardcoded Ollama Model in UI
**File:** `ui.py:100`
**Issue:** Hardcoded `"llama3.2"` instead of using config
**Risk:** Inconsistent configuration, not a security issue per se

---

## Positive Security Findings

### Strong Input Validation (validation.py)
- Whitelist-based character validation
- Prompt injection detection patterns
- Path traversal prevention
- HTML escaping by default
- Dimension bounds checking

### Good Secrets Management (secrets_manager.py)
- SecretValue class masks values in logs/repr
- HMAC-SHA256 with salt for hashing
- Constant-time comparison (hmac.compare_digest)
- No hardcoded credentials found

### Robust Error Handling (exceptions.py)
- User-friendly vs developer message separation
- No stack traces exposed to users
- Structured error codes

### Resilience Patterns (retry.py)
- Circuit breaker prevents cascade failures
- Exponential backoff with jitter
- Rate limiting (token bucket)

---

## Remediation Plan

### Immediate (Before Release)

1. **Update Gradio minimum version**
2. **Add WebSocket message size limit**
3. **Add WSS recommendation warning**

### Short-term (Next Release)

4. **Default UI host to 127.0.0.1**
5. **Add listener count limits**
6. **Document network security**

---

## References

- [OWASP WebSocket Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html)
- [OWASP Top 10 2025](https://owasp.org/www-project-top-ten/)
- [Gradio 5 Security Review](https://huggingface.co/blog/gradio-5-security)
- [WebSocket Security Best Practices](https://websocket.org/guides/security/)
- [CVE-2025-23042 - Gradio Path Traversal](https://www.wiz.io/vulnerability-database/cve/cve-2025-23042)
