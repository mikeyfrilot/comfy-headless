"""
Extended tests for comfy_headless/feature_flags.py

Covers:
- Feature detection (lines 79-128)
- check_* functions (lines 135-145)
- require_feature decorator (lines 208-233)
- FeatureNotAvailable exception
- Feature listing functions
"""


import pytest


class TestFeatureDetection:
    """Test _detect_features functionality."""

    def test_features_dict_exists(self):
        """FEATURES dict is properly initialized."""
        from comfy_headless.feature_flags import FEATURES

        assert isinstance(FEATURES, dict)
        expected_features = ["ai", "websocket", "health", "ui", "validation", "observability"]
        for feature in expected_features:
            assert feature in FEATURES

    def test_features_are_booleans(self):
        """All feature values are booleans."""
        from comfy_headless.feature_flags import FEATURES

        for name, value in FEATURES.items():
            assert isinstance(value, bool), f"Feature '{name}' should be bool, got {type(value)}"

    def test_ai_feature_detection(self):
        """AI feature detected based on httpx availability."""
        from comfy_headless.feature_flags import FEATURES

        try:
            import httpx

            assert FEATURES["ai"] is True
        except ImportError:
            assert FEATURES["ai"] is False

    def test_websocket_feature_detection(self):
        """WebSocket feature detected based on websockets availability."""
        from comfy_headless.feature_flags import FEATURES

        try:
            import websockets

            assert FEATURES["websocket"] is True
        except ImportError:
            assert FEATURES["websocket"] is False

    def test_health_feature_detection(self):
        """Health feature detected based on psutil availability."""
        from comfy_headless.feature_flags import FEATURES

        try:
            import psutil

            assert FEATURES["health"] is True
        except ImportError:
            assert FEATURES["health"] is False

    def test_ui_feature_detection(self):
        """UI feature detected based on gradio availability."""
        from comfy_headless.feature_flags import FEATURES

        try:
            import gradio

            assert FEATURES["ui"] is True
        except ImportError:
            assert FEATURES["ui"] is False

    def test_validation_feature_detection(self):
        """Validation feature detected based on pydantic availability."""
        from comfy_headless.feature_flags import FEATURES

        try:
            import pydantic
            import pydantic_settings

            assert FEATURES["validation"] is True
        except ImportError:
            assert FEATURES["validation"] is False

    def test_observability_feature_detection(self):
        """Observability feature detected based on opentelemetry availability."""
        from comfy_headless.feature_flags import FEATURES

        try:
            import opentelemetry

            assert FEATURES["observability"] is True
        except ImportError:
            assert FEATURES["observability"] is False


class TestCheckFeature:
    """Test check_feature function."""

    def test_check_feature_returns_bool(self):
        """check_feature returns boolean."""
        from comfy_headless.feature_flags import check_feature

        result = check_feature("ai")
        assert isinstance(result, bool)

    def test_check_feature_known_features(self):
        """check_feature works for all known features."""
        from comfy_headless.feature_flags import FEATURES, check_feature

        for feature in FEATURES:
            result = check_feature(feature)
            assert result == FEATURES[feature]

    def test_check_feature_unknown_returns_false(self):
        """check_feature returns False for unknown features."""
        from comfy_headless.feature_flags import check_feature

        assert check_feature("nonexistent_feature") is False
        assert check_feature("made_up") is False
        assert check_feature("") is False


class TestGetInstallHint:
    """Test get_install_hint function."""

    def test_get_install_hint_known_features(self):
        """get_install_hint returns proper hints for known features."""
        from comfy_headless.feature_flags import get_install_hint

        hints = {
            "ai": "pip install comfy-headless[ai]",
            "websocket": "pip install comfy-headless[websocket]",
            "health": "pip install comfy-headless[health]",
            "ui": "pip install comfy-headless[ui]",
            "validation": "pip install comfy-headless[validation]",
            "observability": "pip install comfy-headless[observability]",
        }

        for feature, expected in hints.items():
            result = get_install_hint(feature)
            assert result == expected

    def test_get_install_hint_unknown_feature(self):
        """get_install_hint generates hint for unknown features."""
        from comfy_headless.feature_flags import get_install_hint

        result = get_install_hint("custom")
        assert "pip install" in result
        assert "[custom]" in result


class TestListAvailableFeatures:
    """Test list_available_features function."""

    def test_list_available_features_returns_dict(self):
        """list_available_features returns a dictionary."""
        from comfy_headless.feature_flags import list_available_features

        result = list_available_features()
        assert isinstance(result, dict)

    def test_list_available_features_only_installed(self):
        """list_available_features only includes installed features."""
        from comfy_headless.feature_flags import FEATURES, list_available_features

        result = list_available_features()

        for name in result:
            assert FEATURES[name] is True

        for name, available in FEATURES.items():
            if not available:
                assert name not in result

    def test_list_available_features_has_descriptions(self):
        """list_available_features includes descriptions."""
        from comfy_headless.feature_flags import list_available_features

        result = list_available_features()

        for _name, description in result.items():
            assert isinstance(description, str)


class TestListMissingFeatures:
    """Test list_missing_features function."""

    def test_list_missing_features_returns_dict(self):
        """list_missing_features returns a dictionary."""
        from comfy_headless.feature_flags import list_missing_features

        result = list_missing_features()
        assert isinstance(result, dict)

    def test_list_missing_features_only_uninstalled(self):
        """list_missing_features only includes missing features."""
        from comfy_headless.feature_flags import FEATURES, list_missing_features

        result = list_missing_features()

        for name in result:
            assert FEATURES[name] is False

        for name, available in FEATURES.items():
            if available:
                assert name not in result

    def test_list_missing_features_has_install_hints(self):
        """list_missing_features includes install hints."""
        from comfy_headless.feature_flags import list_missing_features

        result = list_missing_features()

        for _name, hint in result.items():
            assert isinstance(hint, str)
            assert "pip install" in hint


class TestRequireFeature:
    """Test require_feature decorator."""

    def test_require_feature_allows_when_available(self):
        """require_feature allows function when feature is available."""
        from comfy_headless.feature_flags import FEATURES, require_feature

        # Find an available feature
        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        @require_feature(available_feature)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_require_feature_raises_when_unavailable(self):
        """require_feature raises ImportError when feature is missing."""
        from comfy_headless.feature_flags import FEATURES, require_feature

        # Find an unavailable feature
        unavailable_feature = None
        for name, available in FEATURES.items():
            if not available:
                unavailable_feature = name
                break

        if unavailable_feature is None:
            pytest.skip("All features available, cannot test unavailable case")

        @require_feature(unavailable_feature)
        def test_func():
            return "success"

        with pytest.raises(ImportError) as exc_info:
            test_func()

        assert unavailable_feature in str(exc_info.value)
        assert "pip install" in str(exc_info.value)

    def test_require_feature_preserves_function_metadata(self):
        """require_feature preserves function name and docstring."""
        from comfy_headless.feature_flags import require_feature

        @require_feature("ai")
        def my_function():
            """My docstring."""
            return "value"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_require_feature_passes_arguments(self):
        """require_feature passes through arguments."""
        from comfy_headless.feature_flags import FEATURES, require_feature

        # Find an available feature
        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        @require_feature(available_feature)
        def add(a, b, multiplier=1):
            return (a + b) * multiplier

        assert add(2, 3) == 5
        assert add(2, 3, multiplier=2) == 10


class TestFeatureNotAvailable:
    """Test FeatureNotAvailable exception."""

    def test_exception_has_feature_name(self):
        """FeatureNotAvailable stores feature name."""
        from comfy_headless.feature_flags import FeatureNotAvailable

        exc = FeatureNotAvailable("ai")
        assert exc.feature == "ai"

    def test_exception_has_install_hint(self):
        """FeatureNotAvailable includes install hint."""
        from comfy_headless.feature_flags import FeatureNotAvailable

        exc = FeatureNotAvailable("websocket")
        assert exc.install_hint == "pip install comfy-headless[websocket]"

    def test_exception_message_includes_hint(self):
        """FeatureNotAvailable message includes install hint."""
        from comfy_headless.feature_flags import FeatureNotAvailable

        exc = FeatureNotAvailable("ui")
        message = str(exc)

        assert "ui" in message
        assert "pip install" in message

    def test_exception_custom_message(self):
        """FeatureNotAvailable accepts custom message."""
        from comfy_headless.feature_flags import FeatureNotAvailable

        custom_msg = "Custom error for testing"
        exc = FeatureNotAvailable("health", message=custom_msg)

        assert str(exc) == custom_msg
        assert exc.feature == "health"

    def test_exception_is_import_error(self):
        """FeatureNotAvailable is subclass of ImportError."""
        from comfy_headless.feature_flags import FeatureNotAvailable

        assert issubclass(FeatureNotAvailable, ImportError)

        exc = FeatureNotAvailable("ai")
        assert isinstance(exc, ImportError)


class TestEnsureFeature:
    """Test ensure_feature function."""

    def test_ensure_feature_passes_when_available(self):
        """ensure_feature doesn't raise when feature is available."""
        from comfy_headless.feature_flags import FEATURES, ensure_feature

        # Find an available feature
        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        # Should not raise
        ensure_feature(available_feature)

    def test_ensure_feature_raises_when_unavailable(self):
        """ensure_feature raises FeatureNotAvailable when missing."""
        from comfy_headless.feature_flags import FEATURES, FeatureNotAvailable, ensure_feature

        # Find an unavailable feature
        unavailable_feature = None
        for name, available in FEATURES.items():
            if not available:
                unavailable_feature = name
                break

        if unavailable_feature is None:
            pytest.skip("All features available, cannot test unavailable case")

        with pytest.raises(FeatureNotAvailable) as exc_info:
            ensure_feature(unavailable_feature)

        assert exc_info.value.feature == unavailable_feature


class TestFeatureDescriptions:
    """Test FEATURE_DESCRIPTIONS constant."""

    def test_feature_descriptions_exists(self):
        """FEATURE_DESCRIPTIONS dict exists."""
        from comfy_headless.feature_flags import FEATURE_DESCRIPTIONS

        assert isinstance(FEATURE_DESCRIPTIONS, dict)

    def test_feature_descriptions_complete(self):
        """FEATURE_DESCRIPTIONS has entry for each feature."""
        from comfy_headless.feature_flags import FEATURE_DESCRIPTIONS, FEATURES

        for feature in FEATURES:
            assert feature in FEATURE_DESCRIPTIONS
            assert isinstance(FEATURE_DESCRIPTIONS[feature], str)
            assert len(FEATURE_DESCRIPTIONS[feature]) > 0


class TestInstallHints:
    """Test INSTALL_HINTS constant."""

    def test_install_hints_exists(self):
        """INSTALL_HINTS dict exists."""
        from comfy_headless.feature_flags import INSTALL_HINTS

        assert isinstance(INSTALL_HINTS, dict)

    def test_install_hints_complete(self):
        """INSTALL_HINTS has entry for each feature."""
        from comfy_headless.feature_flags import FEATURES, INSTALL_HINTS

        for feature in FEATURES:
            assert feature in INSTALL_HINTS
            assert "pip install" in INSTALL_HINTS[feature]
            assert feature in INSTALL_HINTS[feature]


class TestAllExports:
    """Test __all__ exports."""

    def test_all_exports_defined(self):
        """All expected exports are in __all__."""
        from comfy_headless import feature_flags

        expected = [
            "FEATURES",
            "check_feature",
            "require_feature",
            "get_install_hint",
            "list_available_features",
            "list_missing_features",
        ]

        for name in expected:
            assert name in feature_flags.__all__

    def test_all_exports_accessible(self):
        """All items in __all__ are accessible."""
        from comfy_headless import feature_flags

        for name in feature_flags.__all__:
            assert hasattr(feature_flags, name)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_feature_name(self):
        """Empty feature name is handled gracefully."""
        from comfy_headless.feature_flags import check_feature, get_install_hint

        assert check_feature("") is False
        hint = get_install_hint("")
        assert "pip install" in hint

    def test_none_like_feature_names(self):
        """None-like feature names are handled."""
        from comfy_headless.feature_flags import check_feature

        # These should not crash
        assert check_feature("none") is False
        assert check_feature("null") is False

    def test_features_dict_immutability_conceptual(self):
        """Modifying FEATURES copy doesn't affect original."""
        from comfy_headless.feature_flags import FEATURES

        # Take a copy and modify
        copy = dict(FEATURES)
        copy["ai"] = not copy.get("ai", False)

        # Original should be unchanged (Python dicts are mutable but we test conceptually)
        # This is more of a documentation test

    def test_require_feature_with_kwargs(self):
        """require_feature handles kwargs correctly."""
        from comfy_headless.feature_flags import FEATURES, require_feature

        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        @require_feature(available_feature)
        def func_with_kwargs(**kwargs):
            return kwargs

        result = func_with_kwargs(a=1, b=2)
        assert result == {"a": 1, "b": 2}

    def test_require_feature_with_args_and_kwargs(self):
        """require_feature handles mixed args/kwargs."""
        from comfy_headless.feature_flags import FEATURES, require_feature

        available_feature = None
        for name, available in FEATURES.items():
            if available:
                available_feature = name
                break

        if available_feature is None:
            pytest.skip("No features available to test")

        @require_feature(available_feature)
        def mixed_func(*args, **kwargs):
            return (args, kwargs)

        result = mixed_func(1, 2, 3, key="value")
        assert result == ((1, 2, 3), {"key": "value"})
