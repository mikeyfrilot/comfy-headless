"""
Extended tests for comfy_headless/secrets.py

Covers:
- SecretValue masking and comparison (lines 253-265)
- Credential handling edge cases (lines 483-505)
- Token generation
- Hash verification
- URL masking
- Dict redaction
"""

import os
from unittest.mock import patch

import pytest


class TestSecretValueBasics:
    """Test SecretValue basic functionality."""

    def test_secret_value_stores_value(self):
        """SecretValue stores the secret."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("my-secret-key")
        assert secret.get_secret_value() == "my-secret-key"

    def test_secret_value_hidden_in_repr(self):
        """SecretValue hides value in __repr__."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("my-secret-key")
        assert "my-secret-key" not in repr(secret)
        assert "***" in repr(secret) or "**********" in repr(secret)

    def test_secret_value_hidden_in_str(self):
        """SecretValue hides value in __str__."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("my-secret-key")
        assert "my-secret-key" not in str(secret)
        assert "**" in str(secret)

    def test_secret_value_len(self):
        """SecretValue reports correct length."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("12345")
        assert len(secret) == 5

    def test_secret_value_bool_true(self):
        """Non-empty SecretValue is truthy."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("value")
        assert bool(secret) is True

    def test_secret_value_bool_false(self):
        """Empty SecretValue is falsy."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("")
        assert bool(secret) is False


class TestSecretValueMasking:
    """Test SecretValue masking functionality."""

    def test_get_masked_default(self):
        """get_masked shows partial value by default."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("my-secret-api-key-12345")
        masked = secret.get_masked()

        # Should show 4 chars at start and end
        assert masked.startswith("my-s")
        assert masked.endswith("2345")
        assert "***" in masked

    def test_get_masked_custom_chars(self):
        """get_masked respects show_chars parameter."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("abcdefghij")
        masked = secret.get_masked(show_chars=2)

        assert masked.startswith("ab")
        assert masked.endswith("ij")
        assert "***" in masked

    def test_get_masked_short_value(self):
        """get_masked fully masks short values."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("short")
        masked = secret.get_masked(show_chars=4)

        # Value is 5 chars, show_chars*2 = 8, so should be fully masked
        assert masked == "*****"

    def test_get_masked_very_short(self):
        """get_masked handles very short values."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("ab")
        masked = secret.get_masked()

        assert masked == "**"


class TestSecretValueComparison:
    """Test SecretValue comparison (timing-safe)."""

    def test_equal_secrets(self):
        """Equal SecretValues compare as equal."""
        from comfy_headless.secrets import SecretValue

        s1 = SecretValue("same-value")
        s2 = SecretValue("same-value")

        assert s1 == s2

    def test_different_secrets(self):
        """Different SecretValues compare as not equal."""
        from comfy_headless.secrets import SecretValue

        s1 = SecretValue("value1")
        s2 = SecretValue("value2")

        assert s1 != s2

    def test_compare_with_string(self):
        """SecretValue can compare with plain string."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("my-key")

        assert secret == "my-key"
        assert secret != "other-key"

    def test_compare_with_other_types(self):
        """SecretValue returns False for other types."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("123")

        assert secret != 123
        assert secret is not None
        assert secret != ["123"]

    def test_hash_consistent(self):
        """SecretValue hash is consistent."""
        from comfy_headless.secrets import SecretValue

        s1 = SecretValue("value")
        s2 = SecretValue("value")

        assert hash(s1) == hash(s2)

    def test_hashable_in_set(self):
        """SecretValue can be used in sets."""
        from comfy_headless.secrets import SecretValue

        s1 = SecretValue("value")
        s2 = SecretValue("value")
        s3 = SecretValue("other")

        secret_set = {s1, s2, s3}
        assert len(secret_set) == 2  # s1 and s2 are duplicates


class TestSecretsManagerConfig:
    """Test SecretsManagerConfig dataclass."""

    def test_default_config(self):
        """Config has sensible defaults."""
        from comfy_headless.secrets import SecretsManagerConfig

        config = SecretsManagerConfig()

        assert config.env_prefix == "COMFY_HEADLESS_"
        assert config.load_dotenv is True
        assert config.vault_enabled is False
        assert config.mask_in_logs is True

    def test_custom_config(self):
        """Config accepts custom values."""
        from comfy_headless.secrets import SecretsManagerConfig

        config = SecretsManagerConfig(
            env_prefix="CUSTOM_", vault_enabled=True, vault_url="http://vault:8200"
        )

        assert config.env_prefix == "CUSTOM_"
        assert config.vault_enabled is True
        assert config.vault_url == "http://vault:8200"


class TestSecretsManager:
    """Test SecretsManager class."""

    def test_manager_initialization(self):
        """SecretsManager initializes successfully."""
        from comfy_headless.secrets import SecretsManager

        manager = SecretsManager()
        assert manager is not None

    def test_get_from_env(self):
        """Manager retrieves secrets from environment."""
        from comfy_headless.secrets import SecretsManager

        with patch.dict(os.environ, {"COMFY_HEADLESS_TEST_KEY": "test-value"}):
            manager = SecretsManager()
            secret = manager.get("TEST_KEY")

            assert secret is not None
            assert secret.get_secret_value() == "test-value"

    def test_get_with_default(self):
        """Manager returns default when secret not found."""
        from comfy_headless.secrets import SecretsManager

        # Ensure key doesn't exist
        os.environ.pop("COMFY_HEADLESS_NONEXISTENT", None)

        manager = SecretsManager()
        secret = manager.get("NONEXISTENT", default="default-value")

        assert secret is not None
        assert secret.get_secret_value() == "default-value"

    def test_get_returns_none_without_default(self):
        """Manager returns None when secret not found and no default."""
        from comfy_headless.secrets import SecretsManager

        os.environ.pop("COMFY_HEADLESS_NONEXISTENT", None)

        manager = SecretsManager()
        secret = manager.get("NONEXISTENT")

        assert secret is None


class TestGetSecret:
    """Test get_secret convenience function."""

    def test_get_secret_from_env(self):
        """get_secret retrieves from environment."""
        from comfy_headless.secrets import get_secret

        with patch.dict(os.environ, {"COMFY_HEADLESS_MY_KEY": "my-value"}):
            secret = get_secret("MY_KEY")

            assert secret is not None
            assert secret.get_secret_value() == "my-value"

    def test_get_secret_with_default(self):
        """get_secret returns default when missing."""
        from comfy_headless.secrets import get_secret

        os.environ.pop("COMFY_HEADLESS_MISSING", None)

        secret = get_secret("MISSING", default="fallback")
        assert secret.get_secret_value() == "fallback"


class TestGetSecretStr:
    """Test get_secret_str convenience function."""

    def test_get_secret_str_returns_string(self):
        """get_secret_str returns plain string."""
        from comfy_headless.secrets import get_secret_str

        with patch.dict(os.environ, {"COMFY_HEADLESS_KEY": "value"}):
            result = get_secret_str("KEY")

            assert isinstance(result, str)
            assert result == "value"

    def test_get_secret_str_with_default(self):
        """get_secret_str returns default as string."""
        from comfy_headless.secrets import get_secret_str

        # Also clear unprefixed key and reset manager cache
        os.environ.pop("COMFY_HEADLESS_MISSING_STR", None)
        os.environ.pop("MISSING_STR", None)

        # Reset manager cache so it doesn't return cached values
        from comfy_headless.secrets import get_secrets_manager

        get_secrets_manager().clear_cache()

        result = get_secret_str("MISSING_STR", default="default")
        assert result == "default"


class TestGenerateToken:
    """Test generate_token function."""

    def test_generate_token_returns_string(self):
        """generate_token returns a string."""
        from comfy_headless.secrets import generate_token

        token = generate_token()
        assert isinstance(token, str)

    def test_generate_token_length(self):
        """generate_token respects length parameter."""
        from comfy_headless.secrets import generate_token

        token = generate_token(length=32)
        # URL-safe base64 encoding may vary in length
        assert len(token) >= 32

    def test_generate_token_unique(self):
        """generate_token produces unique values."""
        from comfy_headless.secrets import generate_token

        tokens = [generate_token() for _ in range(100)]
        assert len(set(tokens)) == 100  # All unique


class TestGenerateApiKey:
    """Test generate_api_key function."""

    def test_generate_api_key_returns_string(self):
        """generate_api_key returns a string."""
        from comfy_headless.secrets import generate_api_key

        key = generate_api_key()
        assert isinstance(key, str)

    def test_generate_api_key_has_prefix(self):
        """generate_api_key includes prefix."""
        from comfy_headless.secrets import generate_api_key

        key = generate_api_key(prefix="sk_")
        assert key.startswith("sk_")

    def test_generate_api_key_unique(self):
        """generate_api_key produces unique values."""
        from comfy_headless.secrets import generate_api_key

        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100


class TestHashSecret:
    """Test hash_secret function."""

    def test_hash_secret_returns_string(self):
        """hash_secret returns a hash string."""
        from comfy_headless.secrets import hash_secret

        hashed = hash_secret("my-secret")
        assert isinstance(hashed, str)

    def test_hash_secret_with_same_salt(self):
        """hash_secret produces same hash for same input with same salt."""
        from comfy_headless.secrets import hash_secret

        # hash_secret generates random salt if not provided,
        # so we need to provide the same salt for determinism
        h1 = hash_secret("same-value", salt="fixed-salt")
        h2 = hash_secret("same-value", salt="fixed-salt")
        assert h1 == h2

    def test_hash_secret_different_for_different_input(self):
        """hash_secret produces different hash for different input."""
        from comfy_headless.secrets import hash_secret

        h1 = hash_secret("value1")
        h2 = hash_secret("value2")
        assert h1 != h2

    def test_hash_secret_accepts_secret_value(self):
        """hash_secret accepts SecretValue input."""
        from comfy_headless.secrets import SecretValue, hash_secret

        secret = SecretValue("my-secret")
        hashed = hash_secret(secret)
        assert isinstance(hashed, str)


class TestVerifyHashedSecret:
    """Test verify_hashed_secret function."""

    def test_verify_correct_secret(self):
        """verify_hashed_secret returns True for correct secret."""
        from comfy_headless.secrets import hash_secret, verify_hashed_secret

        original = "my-secret"
        hashed = hash_secret(original)

        assert verify_hashed_secret(original, hashed) is True

    def test_verify_wrong_secret(self):
        """verify_hashed_secret returns False for wrong secret."""
        from comfy_headless.secrets import hash_secret, verify_hashed_secret

        hashed = hash_secret("original")

        assert verify_hashed_secret("wrong", hashed) is False


class TestMaskUrlCredentials:
    """Test mask_url_credentials function."""

    def test_mask_url_with_credentials(self):
        """mask_url_credentials masks user:pass in URL."""
        from comfy_headless.secrets import mask_url_credentials

        url = "http://user:password@example.com/path"
        masked = mask_url_credentials(url)

        assert "password" not in masked
        assert "example.com" in masked

    def test_mask_url_without_credentials(self):
        """mask_url_credentials handles URL without credentials."""
        from comfy_headless.secrets import mask_url_credentials

        url = "http://example.com/path"
        masked = mask_url_credentials(url)

        assert masked == url

    def test_mask_url_preserves_structure(self):
        """mask_url_credentials preserves URL structure."""
        from comfy_headless.secrets import mask_url_credentials

        url = "https://user:secret@api.example.com:8080/v1/endpoint?key=value"
        masked = mask_url_credentials(url)

        assert "api.example.com:8080" in masked
        assert "/v1/endpoint" in masked


class TestRedactDict:
    """Test redact_dict function."""

    def test_redact_dict_basic(self):
        """redact_dict redacts sensitive keys."""
        from comfy_headless.secrets import redact_dict

        data = {
            "username": "john",
            "password": "secret123",
            "api_key": "key-123",
        }

        redacted = redact_dict(data)

        assert redacted["username"] == "john"
        assert "secret123" not in str(redacted["password"])
        assert "key-123" not in str(redacted["api_key"])

    def test_redact_dict_preserves_structure(self):
        """redact_dict preserves dict structure."""
        from comfy_headless.secrets import redact_dict

        data = {"config": {"host": "localhost", "token": "secret-token"}}

        redacted = redact_dict(data)

        assert "config" in redacted
        assert redacted["config"]["host"] == "localhost"

    def test_redact_dict_custom_keys(self):
        """redact_dict accepts custom sensitive keys."""
        from comfy_headless.secrets import redact_dict

        data = {"custom_secret": "value", "normal": "normal-value"}

        redacted = redact_dict(data, sensitive_keys=["custom_secret"])

        assert "value" not in str(redacted["custom_secret"])
        assert redacted["normal"] == "normal-value"


class TestGetSecretsManager:
    """Test get_secrets_manager singleton."""

    def test_get_secrets_manager_returns_manager(self):
        """get_secrets_manager returns SecretsManager."""
        from comfy_headless.secrets import SecretsManager, get_secrets_manager

        manager = get_secrets_manager()
        assert isinstance(manager, SecretsManager)

    def test_get_secrets_manager_singleton(self):
        """get_secrets_manager returns same instance."""
        from comfy_headless.secrets import get_secrets_manager

        m1 = get_secrets_manager()
        m2 = get_secrets_manager()
        assert m1 is m2


class TestDotenvAvailable:
    """Test DOTENV_AVAILABLE flag."""

    def test_flag_is_bool(self):
        """DOTENV_AVAILABLE is boolean."""
        from comfy_headless.secrets import DOTENV_AVAILABLE

        assert isinstance(DOTENV_AVAILABLE, bool)


class TestVaultAvailable:
    """Test VAULT_AVAILABLE flag."""

    def test_flag_is_bool(self):
        """VAULT_AVAILABLE is boolean."""
        from comfy_headless.secrets import VAULT_AVAILABLE

        assert isinstance(VAULT_AVAILABLE, bool)


class TestAllExports:
    """Test __all__ exports."""

    def test_all_exports_defined(self):
        """All expected exports are in __all__."""
        from comfy_headless import secrets

        expected = [
            "SecretValue",
            "SecretsManager",
            "get_secrets_manager",
            "get_secret",
            "get_secret_str",
            "generate_token",
            "generate_api_key",
            "hash_secret",
            "verify_hashed_secret",
            "mask_url_credentials",
            "redact_dict",
        ]

        for name in expected:
            assert name in secrets.__all__

    def test_all_exports_accessible(self):
        """All items in __all__ are accessible."""
        from comfy_headless import secrets

        for name in secrets.__all__:
            assert hasattr(secrets, name)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_secret_value(self):
        """Empty SecretValue behaves correctly."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("")

        assert len(secret) == 0
        assert bool(secret) is False
        assert secret.get_secret_value() == ""
        assert secret.get_masked() == ""

    def test_unicode_secret(self):
        """SecretValue handles unicode."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("秘密🔐パスワード")

        assert len(secret) > 0
        assert "秘密" not in str(secret)
        assert secret.get_secret_value() == "秘密🔐パスワード"

    def test_hash_empty_string(self):
        """hash_secret handles empty string."""
        from comfy_headless.secrets import hash_secret

        hashed = hash_secret("")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_redact_empty_dict(self):
        """redact_dict handles empty dict."""
        from comfy_headless.secrets import redact_dict

        result = redact_dict({})
        assert result == {}

    def test_mask_empty_url(self):
        """mask_url_credentials handles empty string."""
        from comfy_headless.secrets import mask_url_credentials

        result = mask_url_credentials("")
        assert result == ""

    def test_secret_value_slots(self):
        """SecretValue uses __slots__ for memory efficiency."""
        from comfy_headless.secrets import SecretValue

        secret = SecretValue("value")

        # Should not be able to add arbitrary attributes
        with pytest.raises(AttributeError):
            secret.arbitrary = "value"
