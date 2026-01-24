"""
Comfy Headless - Secrets Management
=====================================

Secure secrets management with hybrid approach:
- Environment variables for simple deployments
- Optional HashiCorp Vault integration for enterprise
- python-dotenv for local development
- SecretStr for safe handling in memory

Features:
- Never logs secret values
- Automatic masking in __repr__
- Secure comparison to prevent timing attacks
- Key rotation support

Usage:
    from comfy_headless.secrets import (
        get_secret,
        SecretValue,
        SecretsManager,
    )

    # Simple usage
    api_key = get_secret("COMFY_API_KEY")

    # With default
    token = get_secret("AUTH_TOKEN", default="dev-token")

    # Using the manager
    manager = SecretsManager()
    secret = manager.get("database_password")
    print(secret.get_masked())  # "data****word"
"""

import hashlib
import hmac
import os
import secrets as crypto_secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ComfyHeadlessError
from .logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    # SecretValue class
    "SecretValue",
    # SecretsManager
    "SecretsManager",
    "get_secrets_manager",
    # Convenience functions
    "get_secret",
    "get_secret_str",
    # Token generation
    "generate_token",
    "generate_api_key",
    # Hashing
    "hash_secret",
    "verify_hashed_secret",
    # Redaction utilities
    "mask_url_credentials",
    "redact_dict",
    # Constants
    "DOTENV_AVAILABLE",
    "VAULT_AVAILABLE",
]

# Try to import optional dependencies
try:
    from dotenv import dotenv_values, load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import hvac

    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False


# =============================================================================
# SECRET VALUE CLASS
# =============================================================================


class SecretValue:
    """
    A secret value that is safe to pass around.

    - Never exposed in __repr__ or __str__
    - Secure comparison using constant-time algorithm
    - Can be masked for partial display

    Usage:
        secret = SecretValue("my-api-key-12345")
        print(secret)  # "**********"
        print(secret.get_masked())  # "my-a***2345"

        # Secure comparison
        if secret == other_secret:
            ...

        # Get actual value (only when needed)
        api_call(key=secret.get_secret_value())
    """

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        """Get the actual secret value. Use sparingly."""
        return self._value

    def get_masked(self, show_chars: int = 4) -> str:
        """
        Get a partially masked version for logging/display.

        Args:
            show_chars: Number of characters to show at start and end
        """
        if len(self._value) <= show_chars * 2:
            return "*" * len(self._value)
        return f"{self._value[:show_chars]}***{self._value[-show_chars:]}"

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "SecretValue('**********')"

    def __str__(self) -> str:
        return "**********"

    def __eq__(self, other: Any) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        if isinstance(other, SecretValue):
            other_value = other._value
        elif isinstance(other, str):
            other_value = other
        else:
            return False

        return hmac.compare_digest(self._value, other_value)

    def __hash__(self) -> int:
        return hash(self._value)


# =============================================================================
# SECRETS MANAGER
# =============================================================================


@dataclass
class SecretsManagerConfig:
    """Configuration for the secrets manager."""

    # Environment variables
    env_prefix: str = "COMFY_HEADLESS_"
    load_dotenv: bool = True
    dotenv_path: Path | None = None

    # Vault settings
    vault_enabled: bool = False
    vault_url: str | None = None
    vault_token: str | None = None
    vault_mount_point: str = "secret"
    vault_path_prefix: str = "comfy-headless"

    # Security settings
    mask_in_logs: bool = True
    allow_empty_secrets: bool = False


class SecretsManager:
    """
    Centralized secrets management with multiple backends.

    Priority order:
    1. Environment variables (highest)
    2. HashiCorp Vault (if enabled)
    3. .env file (if available)
    4. Default value (lowest)

    Usage:
        manager = SecretsManager()

        # Get a secret
        api_key = manager.get("API_KEY")

        # With typed return
        token = manager.get_str("AUTH_TOKEN")

        # Check if secret exists
        if manager.has("DATABASE_URL"):
            ...
    """

    def __init__(self, config: SecretsManagerConfig | None = None):
        self.config = config or SecretsManagerConfig()
        self._cache: dict[str, SecretValue] = {}
        self._vault_client = None

        # Load .env file if available
        if self.config.load_dotenv and DOTENV_AVAILABLE:
            env_path = self.config.dotenv_path or Path.cwd() / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded .env from {env_path}")

        # Initialize Vault client if enabled
        if self.config.vault_enabled and VAULT_AVAILABLE:
            self._init_vault()

    def _init_vault(self):
        """Initialize HashiCorp Vault client."""
        vault_url = self.config.vault_url or os.environ.get("VAULT_ADDR")
        vault_token = self.config.vault_token or os.environ.get("VAULT_TOKEN")

        if vault_url and vault_token:
            try:
                self._vault_client = hvac.Client(url=vault_url, token=vault_token)
                if self._vault_client.is_authenticated():
                    logger.info("Connected to HashiCorp Vault")
                else:
                    logger.warning("Vault authentication failed")
                    self._vault_client = None
            except Exception as e:
                logger.warning(f"Failed to connect to Vault: {e}")
                self._vault_client = None

    def _get_from_env(self, key: str) -> str | None:
        """Get secret from environment variable."""
        # Try with prefix
        prefixed_key = f"{self.config.env_prefix}{key}"
        value = os.environ.get(prefixed_key)
        if value:
            return value

        # Try without prefix
        return os.environ.get(key)

    def _get_from_vault(self, key: str) -> str | None:
        """Get secret from HashiCorp Vault."""
        if not self._vault_client:
            return None

        try:
            path = f"{self.config.vault_path_prefix}/{key}"
            secret = self._vault_client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.config.vault_mount_point,
            )
            return secret["data"]["data"].get("value")
        except Exception as e:
            logger.debug(f"Vault lookup failed for {key}: {e}")
            return None

    def get(
        self,
        key: str,
        default: str | None = None,
        required: bool = False,
    ) -> SecretValue | None:
        """
        Get a secret value.

        Args:
            key: Secret key name
            default: Default value if not found
            required: Raise if secret not found

        Returns:
            SecretValue or None

        Raises:
            ComfyHeadlessError: If required and not found
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        value = None

        # Priority 1: Environment variables
        value = self._get_from_env(key)

        # Priority 2: Vault
        if value is None and self._vault_client:
            value = self._get_from_vault(key)

        # Priority 3: Default
        if value is None:
            value = default

        # Handle missing required secrets
        if value is None:
            if required:
                raise ComfyHeadlessError(
                    f"Required secret not found: {key}",
                    code="SECRET_NOT_FOUND",
                    suggestions=[
                        f"Set the {self.config.env_prefix}{key} environment variable",
                        "Add it to your .env file",
                    ],
                )
            return None

        # Handle empty secrets
        if not value and not self.config.allow_empty_secrets:
            if required:
                raise ComfyHeadlessError(
                    f"Secret is empty: {key}",
                    code="SECRET_EMPTY",
                )
            return None

        # Wrap in SecretValue and cache
        secret = SecretValue(value)
        self._cache[key] = secret

        if self.config.mask_in_logs:
            logger.debug(f"Loaded secret: {key} (masked)")
        else:
            logger.debug(f"Loaded secret: {key}")

        return secret

    def get_str(
        self,
        key: str,
        default: str | None = None,
        required: bool = False,
    ) -> str | None:
        """Get a secret as a plain string. Use with caution."""
        secret = self.get(key, default, required)
        return secret.get_secret_value() if secret else None

    def has(self, key: str) -> bool:
        """Check if a secret exists."""
        return self.get(key) is not None

    def set(self, key: str, value: str):
        """
        Set a secret in memory cache.

        Note: This does NOT persist to environment or Vault.
        """
        self._cache[key] = SecretValue(value)

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()

    def list_keys(self) -> list:
        """List all cached secret keys (not values)."""
        return list(self._cache.keys())


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_secrets_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> SecretValue | None:
    """
    Get a secret using the global manager.

    Convenience function for simple usage.
    """
    return get_secrets_manager().get(key, default, required)


def get_secret_str(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    """Get a secret as a plain string using the global manager."""
    return get_secrets_manager().get_str(key, default, required)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def generate_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Uses the secrets module (not random!).
    """
    return crypto_secrets.token_urlsafe(length)


def generate_api_key(prefix: str = "sk") -> str:
    """
    Generate an API key in standard format.

    Format: prefix_randomchars (e.g., sk_abc123def456)
    """
    token = crypto_secrets.token_urlsafe(24)
    return f"{prefix}_{token}"


def hash_secret(value: str, salt: str | None = None) -> str:
    """
    Hash a secret value securely.

    Uses SHA-256 with optional salt.
    """
    if salt is None:
        salt = crypto_secrets.token_hex(16)

    combined = f"{salt}:{value}"
    hashed = hashlib.sha256(combined.encode()).hexdigest()

    return f"{salt}:{hashed}"


def verify_hashed_secret(value: str, hashed: str) -> bool:
    """
    Verify a secret against its hash.

    Uses constant-time comparison.
    """
    try:
        salt, expected_hash = hashed.split(":", 1)
        combined = f"{salt}:{value}"
        actual_hash = hashlib.sha256(combined.encode()).hexdigest()
        return hmac.compare_digest(actual_hash, expected_hash)
    except ValueError:
        return False


def mask_url_credentials(url: str) -> str:
    """
    Mask credentials in a URL for safe logging.

    http://user:pass@host -> http://user:****@host
    """
    import re

    pattern = r"(://[^:]+:)([^@]+)(@)"
    return re.sub(pattern, r"\1****\3", url)


def redact_dict(
    data: dict[str, Any],
    sensitive_keys: list | None = None,
) -> dict[str, Any]:
    """
    Redact sensitive values from a dictionary for logging.

    Args:
        data: Dictionary to redact
        sensitive_keys: Keys to redact (default: common secret patterns)

    Returns:
        New dictionary with redacted values
    """
    if sensitive_keys is None:
        sensitive_keys = [
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "apikey",
            "auth",
            "credential",
            "key",
            "private",
            "access_token",
            "refresh_token",
        ]

    sensitive_lower = [k.lower() for k in sensitive_keys]

    result = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(s in key_lower for s in sensitive_lower):
            result[key] = "****REDACTED****"
        elif isinstance(value, dict):
            result[key] = redact_dict(value, sensitive_keys)
        elif isinstance(value, SecretValue):
            result[key] = str(value)
        else:
            result[key] = value

    return result
