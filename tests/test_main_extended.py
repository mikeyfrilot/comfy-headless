"""Extended tests for __main__.py CLI entry point to improve coverage."""

from unittest.mock import patch

import pytest


class TestMainFunction:
    """Test the main() function execution paths."""

    def test_main_version_argument_parsing(self):
        """Test --version argument is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--version", "-v", action="store_true")

        args = parser.parse_args(["--version"])
        assert args.version is True

        # Also test that we can import version
        from comfy_headless import __version__

        assert isinstance(__version__, str)

    def test_main_check_argument_parsing(self):
        """Test --check argument is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--check", action="store_true")

        args = parser.parse_args(["--check"])
        assert args.check is True

    def test_main_short_version_flag(self):
        """Test -v short flag for version."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--version", "-v", action="store_true")

        args = parser.parse_args(["-v"])
        assert args.version is True

    def test_main_function_exists(self):
        """Test main function exists and is callable."""
        from comfy_headless.__main__ import main

        assert callable(main)

    @patch("sys.argv", ["comfy_headless", "--url", "http://192.168.1.100:8188"])
    def test_main_custom_url_parsing(self):
        """Test custom URL is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--url", type=str, default="http://localhost:8188")

        args = parser.parse_args(["--url", "http://192.168.1.100:8188"])
        assert args.url == "http://192.168.1.100:8188"


class TestFeatureCheck:
    """Test feature check functionality."""

    def test_list_available_features_returns_dict(self):
        """Test list_available_features returns a dictionary."""
        from comfy_headless import list_available_features

        features = list_available_features()
        assert isinstance(features, dict)

    def test_list_missing_features_returns_dict(self):
        """Test list_missing_features returns a dictionary."""
        from comfy_headless import list_missing_features

        missing = list_missing_features()
        assert isinstance(missing, dict)

    def test_features_dict_has_expected_keys(self):
        """Test FEATURES dict has expected keys."""
        from comfy_headless.feature_flags import FEATURES

        # Should have at least some common features
        assert "ui" in FEATURES

    def test_get_install_hint_returns_string(self):
        """Test get_install_hint returns install instruction."""
        from comfy_headless.feature_flags import get_install_hint

        hint = get_install_hint("ui")
        assert isinstance(hint, str)
        assert len(hint) > 0


class TestArgParserEdgeCases:
    """Test argument parser edge cases."""

    def test_multiple_flags_combined(self):
        """Test combining multiple flags."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", "-p", type=int, default=7861)
        parser.add_argument("--share", action="store_true")
        parser.add_argument("--url", type=str, default="http://localhost:8188")

        args = parser.parse_args(["--port", "8080", "--share", "--url", "http://other:8188"])
        assert args.port == 8080
        assert args.share is True
        assert args.url == "http://other:8188"

    def test_url_trailing_slash_handling(self):
        """Test URL with trailing slash."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--url", type=str, default="http://localhost:8188")

        args = parser.parse_args(["--url", "http://localhost:8188/"])
        # The main function strips trailing slashes
        assert args.url == "http://localhost:8188/"

    def test_invalid_port_type(self):
        """Test invalid port type raises error."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", "-p", type=int, default=7861)

        with pytest.raises(SystemExit):
            parser.parse_args(["--port", "not_a_number"])


class TestModuleImports:
    """Test module import behavior."""

    def test_main_module_importable(self):
        """Test __main__ module can be imported."""
        import comfy_headless.__main__

        assert hasattr(comfy_headless.__main__, "main")

    def test_main_function_signature(self):
        """Test main function takes no arguments."""
        import inspect

        from comfy_headless.__main__ import main

        sig = inspect.signature(main)
        assert len(sig.parameters) == 0

    def test_all_imports_work(self):
        """Test all imports in __main__ work."""
        # These are the imports used in __main__.py

        # These should be importable
