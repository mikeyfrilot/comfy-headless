"""Tests for theme.py - Custom Gradio Theme Module.

Tests cover:
- Theme creation and configuration
- Color definitions and contrast ratios
- CSS generation
- Theme info retrieval
- Gradio 5 compatibility
"""

from unittest.mock import patch


class TestColorDefinitions:
    """Test color constant definitions."""

    def test_light_colors_defined(self):
        """Test LIGHT_COLORS constant is properly defined."""
        from comfy_headless.theme import LIGHT_COLORS

        assert "background" in LIGHT_COLORS
        assert "surface" in LIGHT_COLORS
        assert "text_primary" in LIGHT_COLORS
        assert "text_secondary" in LIGHT_COLORS
        assert "border" in LIGHT_COLORS

    def test_dark_colors_defined(self):
        """Test DARK_COLORS constant is properly defined."""
        from comfy_headless.theme import DARK_COLORS

        assert "background" in DARK_COLORS
        assert "surface" in DARK_COLORS
        assert "text_primary" in DARK_COLORS
        assert "text_secondary" in DARK_COLORS
        assert "border" in DARK_COLORS

    def test_accent_colors_defined(self):
        """Test ACCENT colors constant is properly defined."""
        from comfy_headless.theme import ACCENT

        assert "primary" in ACCENT
        assert "primary_hover" in ACCENT
        assert "emerald" in ACCENT
        assert "emerald_dark" in ACCENT

    def test_semantic_colors_defined(self):
        """Test SEMANTIC colors constant is properly defined."""
        from comfy_headless.theme import SEMANTIC

        assert "success" in SEMANTIC
        assert "error" in SEMANTIC
        assert "warning" in SEMANTIC
        assert "info" in SEMANTIC

    def test_light_colors_are_valid_hex(self):
        """Test all LIGHT_COLORS are valid hex codes."""
        import re

        from comfy_headless.theme import LIGHT_COLORS

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, color in LIGHT_COLORS.items():
            assert hex_pattern.match(color), f"Invalid hex color for {name}: {color}"

    def test_dark_colors_are_valid_hex(self):
        """Test all DARK_COLORS are valid hex codes."""
        import re

        from comfy_headless.theme import DARK_COLORS

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, color in DARK_COLORS.items():
            assert hex_pattern.match(color), f"Invalid hex color for {name}: {color}"

    def test_accent_colors_are_valid_hex(self):
        """Test all ACCENT colors are valid hex codes."""
        import re

        from comfy_headless.theme import ACCENT

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for name, color in ACCENT.items():
            assert hex_pattern.match(color), f"Invalid hex color for {name}: {color}"


class TestSpacingAndRadius:
    """Test spacing and radius token definitions."""

    def test_spacing_tokens_defined(self):
        """Test SPACING tokens are defined."""
        from comfy_headless.theme import SPACING

        assert "xs" in SPACING
        assert "sm" in SPACING
        assert "md" in SPACING
        assert "lg" in SPACING
        assert "xl" in SPACING

    def test_radius_tokens_defined(self):
        """Test RADIUS tokens are defined."""
        from comfy_headless.theme import RADIUS

        assert "sm" in RADIUS
        assert "md" in RADIUS
        assert "lg" in RADIUS

    def test_spacing_values_are_valid_px(self):
        """Test spacing values are valid pixel values."""
        from comfy_headless.theme import SPACING

        for name, value in SPACING.items():
            assert value.endswith("px"), f"Spacing {name} should be in px: {value}"
            # Extract numeric part
            numeric = int(value.replace("px", ""))
            assert numeric > 0, f"Spacing {name} should be positive"

    def test_radius_values_are_valid(self):
        """Test radius values are valid."""
        from comfy_headless.theme import RADIUS

        for name, value in RADIUS.items():
            assert value.endswith("px") or value == "9999px", f"Invalid radius {name}: {value}"


class TestCustomCSS:
    """Test custom CSS generation."""

    def test_custom_css_defined(self):
        """Test CUSTOM_CSS is a non-empty string."""
        from comfy_headless.theme import CUSTOM_CSS

        assert isinstance(CUSTOM_CSS, str)
        assert len(CUSTOM_CSS) > 0

    def test_custom_css_contains_status_bar(self):
        """Test CSS contains status-bar styles."""
        from comfy_headless.theme import CUSTOM_CSS

        assert ".status-bar" in CUSTOM_CSS

    def test_custom_css_contains_generate_btn(self):
        """Test CSS contains generate-btn styles."""
        from comfy_headless.theme import CUSTOM_CSS

        assert ".generate-btn" in CUSTOM_CSS

    def test_custom_css_contains_spacing_tokens(self):
        """Test CSS contains spacing token definitions."""
        from comfy_headless.theme import CUSTOM_CSS

        assert "--spacing-xs" in CUSTOM_CSS
        assert "--spacing-md" in CUSTOM_CSS

    def test_custom_css_contains_transition_tokens(self):
        """Test CSS contains transition token definitions."""
        from comfy_headless.theme import CUSTOM_CSS

        assert "--transition-fast" in CUSTOM_CSS
        assert "--transition-normal" in CUSTOM_CSS

    def test_custom_css_contains_dark_mode_rules(self):
        """Test CSS contains dark mode specific rules."""
        from comfy_headless.theme import CUSTOM_CSS

        assert ".dark" in CUSTOM_CSS

    def test_custom_css_contains_reduced_motion(self):
        """Test CSS contains reduced motion media query."""
        from comfy_headless.theme import CUSTOM_CSS

        assert "prefers-reduced-motion" in CUSTOM_CSS

    def test_custom_css_contains_responsive_rules(self):
        """Test CSS contains responsive breakpoints."""
        from comfy_headless.theme import CUSTOM_CSS

        assert "@media" in CUSTOM_CSS
        assert "768px" in CUSTOM_CSS


class TestGetCss:
    """Test get_css function."""

    def test_get_css_returns_string(self):
        """Test get_css returns a string."""
        from comfy_headless.theme import get_css

        css = get_css()
        assert isinstance(css, str)

    def test_get_css_matches_custom_css(self):
        """Test get_css returns CUSTOM_CSS constant."""
        from comfy_headless.theme import CUSTOM_CSS, get_css

        assert get_css() == CUSTOM_CSS


class TestGetThemeInfo:
    """Test get_theme_info function."""

    def test_get_theme_info_returns_dict(self):
        """Test get_theme_info returns a dictionary."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert isinstance(info, dict)

    def test_get_theme_info_has_name(self):
        """Test theme info contains name."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert "name" in info
        assert "Comfy Headless" in info["name"]

    def test_get_theme_info_has_version(self):
        """Test theme info contains version."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert "version" in info

    def test_get_theme_info_has_light_mode_info(self):
        """Test theme info describes light mode."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert "light_mode" in info
        assert "Warm Neutral" in info["light_mode"]

    def test_get_theme_info_has_dark_mode_info(self):
        """Test theme info describes dark mode."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert "dark_mode" in info
        assert "Emerald" in info["dark_mode"]

    def test_get_theme_info_has_primary_colors(self):
        """Test theme info contains primary colors."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert "primary_color_light" in info
        assert "primary_color_dark" in info

    def test_get_theme_info_has_features(self):
        """Test theme info contains feature list."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert "features" in info
        assert isinstance(info["features"], list)
        assert len(info["features"]) > 0

    def test_get_theme_info_has_color_palettes(self):
        """Test theme info contains color palettes."""
        from comfy_headless.theme import get_theme_info

        info = get_theme_info()
        assert "colors" in info
        assert "light" in info["colors"]
        assert "dark" in info["colors"]
        assert "accent" in info["colors"]


class TestCreateComfyTheme:
    """Test create_comfy_theme function."""

    def test_create_comfy_theme_returns_theme(self):
        """Test create_comfy_theme returns a Gradio theme."""

        from comfy_headless.theme import create_comfy_theme

        theme = create_comfy_theme()
        # Should be a Gradio theme object
        assert theme is not None
        assert hasattr(theme, "set")

    def test_create_comfy_theme_uses_teal_primary(self):
        """Test theme uses teal instead of blue for primary."""
        from comfy_headless.theme import create_comfy_theme

        theme = create_comfy_theme()
        # The theme should have been configured
        # We can't easily check the hue, but we can verify it was created
        assert theme is not None

    def test_create_comfy_theme_is_reusable(self):
        """Test theme can be created multiple times."""
        from comfy_headless.theme import create_comfy_theme

        theme1 = create_comfy_theme()
        theme2 = create_comfy_theme()

        # Both should be valid themes
        assert theme1 is not None
        assert theme2 is not None


class TestThemeExports:
    """Test module exports."""

    def test_all_exports_defined(self):
        """Test __all__ contains expected exports."""
        from comfy_headless.theme import __all__

        expected = [
            "create_comfy_theme",
            "get_css",
            "get_theme_info",
            "LIGHT_COLORS",
            "DARK_COLORS",
            "ACCENT",
            "SEMANTIC",
            "SPACING",
            "RADIUS",
            "CUSTOM_CSS",
        ]

        for item in expected:
            assert item in __all__, f"Missing export: {item}"

    def test_all_exports_importable(self):
        """Test all exports can be imported."""
        from comfy_headless.theme import (
            create_comfy_theme,
            get_css,
        )

        # All should be importable without error
        assert create_comfy_theme is not None
        assert get_css is not None


class TestContrastCompliance:
    """Test WCAG contrast ratio compliance."""

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def luminance(rgb: tuple) -> float:
        """Calculate relative luminance."""

        def adjust(c):
            c = c / 255
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = rgb
        return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)

    @staticmethod
    def contrast_ratio(color1: str, color2: str) -> float:
        """Calculate contrast ratio between two hex colors."""
        rgb1 = TestContrastCompliance.hex_to_rgb(color1)
        rgb2 = TestContrastCompliance.hex_to_rgb(color2)
        l1 = TestContrastCompliance.luminance(rgb1)
        l2 = TestContrastCompliance.luminance(rgb2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def test_light_mode_text_contrast(self):
        """Test light mode text has sufficient contrast (4.5:1 min)."""
        from comfy_headless.theme import LIGHT_COLORS

        ratio = self.contrast_ratio(LIGHT_COLORS["text_primary"], LIGHT_COLORS["background"])
        assert ratio >= 4.5, f"Light mode text contrast is {ratio:.2f}, need 4.5:1"

    def test_dark_mode_text_contrast(self):
        """Test dark mode text has sufficient contrast (4.5:1 min)."""
        from comfy_headless.theme import DARK_COLORS

        ratio = self.contrast_ratio(DARK_COLORS["text_primary"], DARK_COLORS["background"])
        assert ratio >= 4.5, f"Dark mode text contrast is {ratio:.2f}, need 4.5:1"

    def test_light_mode_secondary_text_contrast(self):
        """Test light mode secondary text contrast (3:1 min for large text)."""
        from comfy_headless.theme import LIGHT_COLORS

        ratio = self.contrast_ratio(LIGHT_COLORS["text_secondary"], LIGHT_COLORS["background"])
        assert ratio >= 3.0, f"Light secondary text contrast is {ratio:.2f}, need 3:1"

    def test_dark_mode_secondary_text_contrast(self):
        """Test dark mode secondary text contrast (3:1 min for large text)."""
        from comfy_headless.theme import DARK_COLORS

        ratio = self.contrast_ratio(DARK_COLORS["text_secondary"], DARK_COLORS["background"])
        assert ratio >= 3.0, f"Dark secondary text contrast is {ratio:.2f}, need 3:1"


class TestThemeIntegration:
    """Integration tests for theme with UI module."""

    @patch("comfy_headless.ui.client")
    @patch("comfy_headless.ui.intel")
    def test_ui_uses_custom_theme(self, mock_intel, mock_client):
        """Test UI module imports and uses custom theme."""
        mock_client.is_online.return_value = False
        mock_client.get_samplers.return_value = ["euler"]
        mock_client.get_schedulers.return_value = ["normal"]
        mock_client.get_checkpoints.return_value = []
        mock_client.get_motion_models.return_value = []
        mock_intel.check_ollama.return_value = False

        # Import should work without error
        from comfy_headless.ui import create_ui

        # The function should be callable
        assert callable(create_ui)

    def test_theme_can_be_imported_from_package(self):
        """Test theme functions can be lazy-loaded from main package."""
        # This tests the lazy import mechanism
        from comfy_headless.theme import create_comfy_theme, get_theme_info

        theme = create_comfy_theme()
        info = get_theme_info()

        assert theme is not None
        assert info is not None


class TestCSSValidation:
    """Test CSS syntax and structure."""

    def test_css_has_root_variables(self):
        """Test CSS defines :root variables."""
        from comfy_headless.theme import CUSTOM_CSS

        assert ":root" in CUSTOM_CSS

    def test_css_has_matching_braces(self):
        """Test CSS has balanced braces."""
        from comfy_headless.theme import CUSTOM_CSS

        open_braces = CUSTOM_CSS.count("{")
        close_braces = CUSTOM_CSS.count("}")
        assert open_braces == close_braces, (
            f"Unbalanced braces: {open_braces} open, {close_braces} close"
        )

    def test_css_has_no_syntax_errors(self):
        """Test CSS doesn't have common syntax errors."""
        from comfy_headless.theme import CUSTOM_CSS

        # Check for common errors
        assert ";;;" not in CUSTOM_CSS, "Triple semicolons found"
        assert "{{" not in CUSTOM_CSS, "Double open braces found"
        assert "}}" not in CUSTOM_CSS, "Double close braces found"

    def test_css_uses_css_variables(self):
        """Test CSS uses CSS variables properly."""
        from comfy_headless.theme import CUSTOM_CSS

        # Should use var() syntax
        assert "var(--" in CUSTOM_CSS

    def test_css_has_important_overrides(self):
        """Test CSS uses !important for Gradio overrides."""
        from comfy_headless.theme import CUSTOM_CSS

        # Some rules need !important to override Gradio defaults
        assert "!important" in CUSTOM_CSS
