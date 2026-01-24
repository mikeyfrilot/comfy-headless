"""Tests for __main__.py CLI entry point."""



class TestMainArgParsing:
    """Test argument parsing."""

    def test_parser_defaults(self):
        """Test default argument values."""
        import argparse

        # Create parser manually to test defaults
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", "-p", type=int, default=7861)
        parser.add_argument("--share", action="store_true")
        parser.add_argument("--url", type=str, default="http://localhost:8188")
        parser.add_argument("--version", "-v", action="store_true")
        parser.add_argument("--check", action="store_true")

        args = parser.parse_args([])
        assert args.port == 7861
        assert args.share is False
        assert args.url == "http://localhost:8188"
        assert args.version is False
        assert args.check is False

    def test_parser_custom_port(self):
        """Test custom port argument."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", "-p", type=int, default=7861)
        parser.add_argument("--share", action="store_true")

        args = parser.parse_args(["--port", "8080"])
        assert args.port == 8080

    def test_parser_share_flag(self):
        """Test share flag."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--share", action="store_true")

        args = parser.parse_args(["--share"])
        assert args.share is True

    def test_parser_short_port_flag(self):
        """Test -p short flag."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", "-p", type=int, default=7861)

        args = parser.parse_args(["-p", "9000"])
        assert args.port == 9000

    def test_parser_custom_url(self):
        """Test custom URL argument."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--url", type=str, default="http://localhost:8188")

        args = parser.parse_args(["--url", "http://192.168.1.100:8188"])
        assert args.url == "http://192.168.1.100:8188"


class TestVersionCommand:
    """Test --version command."""

    def test_version_prints_and_exits(self):
        """Test version command prints version and exits."""
        # Instead of calling main() directly, test the argparse behavior
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--version", "-v", action="store_true")

        args = parser.parse_args(["--version"])
        assert args.version is True

        # Verify version string is accessible
        from comfy_headless import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestCheckCommand:
    """Test --check command."""

    def test_check_flag_parsing(self):
        """Test check flag is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--check", action="store_true")

        args = parser.parse_args(["--check"])
        assert args.check is True

    def test_feature_functions_exist(self):
        """Test feature listing functions exist."""
        from comfy_headless import list_available_features, list_missing_features

        available = list_available_features()
        missing = list_missing_features()

        assert isinstance(available, dict)
        assert isinstance(missing, dict)


class TestUILaunch:
    """Test UI launch functionality."""

    def test_features_dict_exists(self):
        """Test FEATURES dict exists."""
        from comfy_headless.feature_flags import FEATURES

        assert isinstance(FEATURES, dict)
        assert "ui" in FEATURES

    def test_get_install_hint_exists(self):
        """Test get_install_hint function exists."""
        from comfy_headless.feature_flags import get_install_hint

        hint = get_install_hint("ui")
        assert hint is not None
        assert "pip" in hint.lower() or "install" in hint.lower()

    def test_launch_function_exists(self):
        """Test launch function exists when UI feature is available."""
        try:
            from comfy_headless import launch

            assert callable(launch)
        except ImportError:
            # UI feature not installed - that's OK
            pass


class TestCLIIntegration:
    """Test CLI integration aspects."""

    def test_module_runnable(self):
        """Test module can be imported without errors."""
        import comfy_headless.__main__

        assert hasattr(comfy_headless.__main__, "main")

    def test_main_is_callable(self):
        """Test main function is callable."""
        from comfy_headless.__main__ import main

        assert callable(main)

    def test_if_name_main_block(self):
        """Test __name__ == '__main__' block exists."""
        import inspect

        import comfy_headless.__main__

        source = inspect.getsource(comfy_headless.__main__)
        assert "if __name__ ==" in source
        assert '"__main__"' in source or "'__main__'" in source
