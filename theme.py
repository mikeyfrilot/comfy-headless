"""
Comfy Headless - Ocean Mist Theme Module
Pastel Apple-inspired palette with cool, calming tones

Features:
- Ocean Mist pastel color scheme
- Cool teal and seafoam accents
- SF Pro system font stack
- Apple-style spacing and radius
- Smooth transitions

Color Palette: Ocean Mist
- Background: Deep slate (#141618)
- Surface: Storm gray (#232830)
- Primary: Pastel teal (#7EC8C8)
- Secondary: Powder blue (#A8C5E2)
- Success: Seafoam (#98D4BB)
"""

import gradio as gr
from gradio.themes import colors, sizes

# =============================================================================
# OCEAN MIST PASTEL COLORS
# =============================================================================

# Background hierarchy (Ocean Mist dark mode)
DARK_COLORS = {
    "background": "#141618",  # Deep slate
    "surface": "#232830",  # Storm gray
    "surface_alt": "#2D3540",  # Lighter storm
    "input_bg": "#232830",  # Match surface
    "border": "#3A4554",  # Steel blue border
    "border_focus": "#7EC8C8",  # Pastel teal
    "text_primary": "#F0F4F8",  # Ice white
    "text_secondary": "#B0BEC5",  # Muted ice
    "text_muted": "#78909C",  # Cool gray
}

# Light mode (Ocean Mist light)
LIGHT_COLORS = {
    "background": "#F0F4F8",  # Ice white
    "surface": "#FFFFFF",  # Pure white
    "surface_alt": "#E8EEF2",  # Light mist
    "border": "#B0BEC5",  # Cool border
    "border_focus": "#5BA3A3",  # Deeper teal
    "text_primary": "#1A2530",  # Deep slate text
    "text_secondary": "#546E7A",  # Muted slate
    "text_muted": "#90A4AE",  # Light muted
}

# Ocean Mist Accent Colors
ACCENT = {
    "blue": "#7EC8C8",  # Pastel teal (primary dark)
    "blue_light": "#5BA3A3",  # Deeper teal (primary light)
    "green": "#98D4BB",  # Seafoam
    "red": "#E57373",  # Soft coral
    "orange": "#FFB74D",  # Warm sand
    "purple": "#B39DDB",  # Soft lavender
    "pink": "#F48FB1",  # Soft rose
    "teal": "#A8C5E2",  # Powder blue
    "yellow": "#FFF176",  # Soft lemon
    "indigo": "#9FA8DA",  # Soft periwinkle
}

# Ocean Mist Gray scale
GRAYS = {
    "gray": "#90A4AE",  # Cool gray
    "gray2": "#78909C",  # Medium cool
    "gray3": "#546E7A",  # Deep cool
    "gray4": "#3A4554",  # Steel blue
    "gray5": "#2D3540",  # Storm gray light
    "gray6": "#232830",  # Storm gray
}

# Semantic Colors (Ocean Mist style)
SEMANTIC = {
    "success": "#98D4BB",  # Seafoam
    "success_dark": "#98D4BB",
    "error": "#E57373",  # Soft coral
    "error_dark": "#E57373",
    "warning": "#FFB74D",  # Warm sand
    "warning_dark": "#FFB74D",
    "info": "#A8C5E2",  # Powder blue
    "info_dark": "#7EC8C8",  # Pastel teal
}


# =============================================================================
# SPACING & SIZING TOKENS
# =============================================================================

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "2xl": "48px",
}

RADIUS = {
    "sm": "6px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "full": "9999px",
}


# =============================================================================
# CSS VARIABLES & CUSTOM STYLES
# =============================================================================

CUSTOM_CSS = """
/* ============================================================================
   Ocean Mist Design System - Pastel Apple-Inspired
   Font: SF Pro (system-ui), Colors: Cool Teal & Seafoam
   ============================================================================ */

:root {
    /* Ocean Mist Accent Colors */
    --ocean-teal: #7EC8C8;
    --ocean-teal-hover: #6BB8B8;
    --ocean-seafoam: #98D4BB;
    --ocean-coral: #E57373;
    --ocean-sand: #FFB74D;
    --ocean-lavender: #B39DDB;
    --ocean-powder: #A8C5E2;

    /* Ocean Mist Gray Scale */
    --ocean-gray: #90A4AE;
    --ocean-gray2: #78909C;
    --ocean-gray3: #546E7A;
    --ocean-gray4: #3A4554;
    --ocean-gray5: #2D3540;
    --ocean-gray6: #232830;

    /* Ocean Mist Backgrounds (Dark) */
    --ocean-bg: #141618;
    --ocean-bg-secondary: #232830;
    --ocean-bg-tertiary: #2D3540;

    /* Ocean Mist Labels */
    --ocean-label: #F0F4F8;
    --ocean-label-secondary: rgba(176, 190, 197, 0.8);
    --ocean-label-tertiary: rgba(120, 144, 156, 0.6);

    /* Ocean Mist Separators */
    --ocean-separator: rgba(58, 69, 84, 0.6);
    --ocean-separator-opaque: #3A4554;

    /* Spacing (Apple 8pt grid) */
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;

    /* Apple-style radius */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 22px;

    /* Transitions */
    --transition-fast: 150ms ease-out;
    --transition-normal: 250ms ease-out;
}

/* ============================================================================
   Global Styles - Apple Typography
   ============================================================================ */

* {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

/* Force Ocean Mist deep slate background */
.dark, body, .gradio-container {
    background-color: #141618 !important;
}

/* ============================================================================
   Apple-Style Buttons - Ocean Mist
   ============================================================================ */

button, .btn {
    font-weight: 500 !important;
    border-radius: var(--radius-md) !important;
    transition: all var(--transition-fast) !important;
    letter-spacing: -0.01em !important;
}

/* Primary button - Soft Teal (not bright orange!) */
.generate-btn,
button.primary,
button[class*="primary"],
.svelte-cmf5ev.primary {
    background-color: #5BA3A3 !important;
    background: linear-gradient(135deg, #6BB8B8 0%, #5BA3A3 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    min-height: 44px !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}

.generate-btn:hover,
button.primary:hover,
button[class*="primary"]:hover {
    background: linear-gradient(135deg, #7EC8C8 0%, #6BB8B8 100%) !important;
    transform: scale(0.98) !important;
}

.generate-btn:active,
button.primary:active,
button[class*="primary"]:active {
    transform: scale(0.96) !important;
    opacity: 0.9 !important;
}

/* Secondary button - Subtle steel gray */
button.secondary,
button[class*="secondary"],
.svelte-cmf5ev.secondary {
    background-color: var(--ocean-gray5) !important;
    color: var(--ocean-label) !important;
    border: 1px solid var(--ocean-gray4) !important;
    box-shadow: none !important;
}

button.secondary:hover,
button[class*="secondary"]:hover {
    background-color: var(--ocean-gray4) !important;
}

/* ============================================================================
   Tab Navigation - Ocean Mist Segmented Control
   ============================================================================ */

.tabs > .tab-nav {
    background-color: var(--ocean-bg-secondary) !important;
    border-radius: var(--radius-md) !important;
    padding: 2px !important;
    border: none !important;
    gap: 0 !important;
}

.tabs > .tab-nav > button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    color: var(--ocean-label-secondary) !important;
    transition: all var(--transition-fast) !important;
}

.tabs > .tab-nav > button:hover {
    color: var(--ocean-label) !important;
}

.tabs > .tab-nav > button.selected {
    background-color: var(--ocean-gray4) !important;
    color: var(--ocean-label) !important;
    border: none !important;
}

/* ============================================================================
   Input Fields - Ocean Mist Style
   ============================================================================ */

textarea,
input[type="text"],
input[type="number"] {
    background-color: var(--ocean-bg-secondary) !important;
    border: 1px solid var(--ocean-separator-opaque) !important;
    border-radius: var(--radius-md) !important;
    color: var(--ocean-label) !important;
    font-size: 17px !important;
    padding: 12px 16px !important;
    transition: all var(--transition-fast) !important;
}

textarea:focus,
input[type="text"]:focus,
input[type="number"]:focus {
    border-color: var(--ocean-teal) !important;
    box-shadow: 0 0 0 4px rgba(126, 200, 200, 0.25) !important;
    outline: none !important;
}

textarea::placeholder,
input::placeholder {
    color: var(--ocean-label-tertiary) !important;
}

/* ============================================================================
   Accordion Headers - Clean
   ============================================================================ */

.accordion {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

.accordion > .label-wrap {
    font-weight: 500 !important;
    padding: var(--spacing-sm) var(--spacing-md) !important;
    transition: background var(--transition-fast) !important;
}

.accordion > .label-wrap:hover {
    background: var(--background-fill-secondary) !important;
}

/* ============================================================================
   Sliders - Themed
   ============================================================================ */

input[type="range"] {
    accent-color: var(--primary-500) !important;
}

.dark input[type="range"] {
    accent-color: var(--primary-400) !important;
}

/* Slider track styling */
input[type="range"]::-webkit-slider-runnable-track {
    background: var(--neutral-200) !important;
    border-radius: var(--radius-full) !important;
}

.dark input[type="range"]::-webkit-slider-runnable-track {
    background: var(--neutral-700) !important;
}

/* ============================================================================
   Dropdowns - Refined
   ============================================================================ */

.dropdown,
select {
    border-radius: var(--radius-sm) !important;
    transition: border-color var(--transition-fast) !important;
}

.dropdown:focus-within,
select:focus {
    border-color: var(--primary-400) !important;
}

/* ============================================================================
   Cards & Blocks - Depth
   ============================================================================ */

.block {
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow var(--transition-normal) !important;
}

.block:hover {
    box-shadow: var(--shadow-md) !important;
}

/* ============================================================================
   Gallery & Image Display
   ============================================================================ */

.gallery {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

.gallery img {
    border-radius: var(--radius-sm) !important;
    transition: transform var(--transition-normal) !important;
}

.gallery img:hover {
    transform: scale(1.02) !important;
}

/* ============================================================================
   Data Tables - Clean
   ============================================================================ */

.dataframe {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

.dataframe th {
    background: var(--background-fill-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

.dataframe tr:hover {
    background: var(--background-fill-secondary) !important;
}

/* ============================================================================
   Progress Indicators
   ============================================================================ */

.progress-bar {
    border-radius: var(--radius-full) !important;
    overflow: hidden !important;
}

.progress-bar > div {
    background: linear-gradient(90deg, var(--primary-500), var(--primary-400)) !important;
    transition: width var(--transition-slow) !important;
}

/* ============================================================================
   Loading States - Generate Button
   ============================================================================ */

/* Button loading state - Gradio adds 'loading' class during operations */
.generate-btn.loading,
.generate-btn[disabled] {
    opacity: 0.7 !important;
    cursor: wait !important;
    pointer-events: none !important;
}

/* Loading spinner animation */
.generate-btn.loading::after {
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-left: 8px;
    border: 2px solid transparent;
    border-top-color: currentColor;
    border-radius: 50%;
    animation: btn-spin 0.8s linear infinite;
}

@keyframes btn-spin {
    to { transform: rotate(360deg); }
}

/* Disabled dropdown styling */
.dropdown-disabled,
select:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    background-color: var(--background-fill-secondary) !important;
}

/* ============================================================================
   Brain Button - Prompt Studio Toggle (inside textbox, top-right)
   ============================================================================ */

.prompt-container {
    position: relative !important;
}

.brain-btn {
    position: absolute !important;
    top: 8px !important;
    right: 8px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    width: 28px !important;
    height: 28px !important;
    font-size: 0.9rem !important;
    border-radius: 6px !important;
    background: linear-gradient(135deg, #98D4BB 0%, #7BC9A0 100%) !important;
    border: none !important;
    cursor: pointer !important;
    transition: all var(--transition-fast) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    z-index: 10 !important;
    opacity: 0.9 !important;
}

.brain-btn:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 2px 12px rgba(152, 212, 187, 0.5) !important;
    opacity: 1 !important;
}

.brain-btn:active {
    transform: scale(0.95) !important;
}

.dark .brain-btn {
    background: linear-gradient(135deg, #98D4BB 0%, #7BC9A0 100%) !important;
}

.dark .brain-btn:hover {
    box-shadow: 0 2px 12px rgba(123, 201, 160, 0.5) !important;
}

/* ============================================================================
   Prompt Studio Sidebar
   ============================================================================ */

#prompt-studio-sidebar .icon {
    font-size: 0 !important;
}

#prompt-studio-sidebar .icon::before {
    content: "🧠" !important;
    font-size: 1.2rem !important;
}

#prompt-studio-sidebar > .closed {
    background: var(--button-primary-background-fill) !important;
    border-radius: var(--radius-lg) !important;
}

#prompt-studio-sidebar > .closed:hover {
    background: var(--button-primary-background-fill-hover) !important;
}

/* ============================================================================
   Markdown Content
   ============================================================================ */

.markdown-content h1,
.markdown-content h2,
.markdown-content h3 {
    color: var(--body-text-color) !important;
    font-weight: 600 !important;
    margin-top: var(--spacing-lg) !important;
    margin-bottom: var(--spacing-sm) !important;
}

.markdown-content code {
    background: var(--background-fill-secondary) !important;
    padding: 2px 6px !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.875em !important;
}

/* ============================================================================
   Utility Classes
   ============================================================================ */

.fade-in {
    animation: fadeIn var(--transition-normal) ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Hide scrollbar but keep functionality */
.hide-scrollbar {
    -ms-overflow-style: none !important;
    scrollbar-width: none !important;
}

.hide-scrollbar::-webkit-scrollbar {
    display: none !important;
}

/* ============================================================================
   Responsive Adjustments
   ============================================================================ */

@media (max-width: 768px) {
    .generate-btn {
        min-height: 48px !important;
    }

    .status-bar {
        padding: var(--spacing-sm) !important;
    }
}

/* ============================================================================
   Reduced Motion Support
   ============================================================================ */

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""


# =============================================================================
# THEME BUILDER FUNCTIONS
# =============================================================================


def create_comfy_theme() -> gr.themes.Base:
    """
    Create Ocean Mist theme - pastel Apple-inspired palette.

    Features:
    - Ocean Mist pastel colors (teal, seafoam accents)
    - SF Pro system font stack
    - Deep slate background for dark mode
    - Apple-style spacing and radius

    Returns:
        gr.themes.Base: Configured Gradio theme
    """
    theme = gr.themes.Soft(
        # Core hues - Teal/Cyan primary
        primary_hue=colors.cyan,
        secondary_hue=colors.slate,
        neutral_hue=colors.slate,
        # Sizing - Apple 8pt grid
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_md,
        text_size=sizes.text_md,
        # Typography - SF Pro system font
        font=["-apple-system", "BlinkMacSystemFont", "SF Pro Display", "system-ui", "sans-serif"],
        font_mono=["SF Mono", "ui-monospace", "monospace"],
    )

    # Apply Ocean Mist color overrides
    theme = theme.set(
        # =================================================================
        # BACKGROUNDS (Ocean Mist Dark Mode)
        # =================================================================
        body_background_fill=LIGHT_COLORS["background"],
        body_background_fill_dark=DARK_COLORS["background"],  # Deep slate #141618
        block_background_fill=LIGHT_COLORS["surface"],
        block_background_fill_dark=DARK_COLORS["surface"],  # Storm gray #232830
        background_fill_secondary=LIGHT_COLORS["surface_alt"],
        background_fill_secondary_dark=DARK_COLORS["surface_alt"],  # #2D3540
        # =================================================================
        # BORDERS (Ocean Mist Separators)
        # =================================================================
        block_border_width="1px",
        block_border_color=LIGHT_COLORS["border"],
        block_border_color_dark=DARK_COLORS["border"],  # Steel blue #3A4554
        border_color_primary=LIGHT_COLORS["border"],
        border_color_primary_dark=DARK_COLORS["border"],
        # =================================================================
        # TEXT COLORS (Ocean Mist Labels)
        # =================================================================
        body_text_color=LIGHT_COLORS["text_primary"],
        body_text_color_dark=DARK_COLORS["text_primary"],  # Ice white #F0F4F8
        body_text_color_subdued=LIGHT_COLORS["text_secondary"],
        body_text_color_subdued_dark=DARK_COLORS["text_secondary"],
        # =================================================================
        # PRIMARY BUTTONS (Soft Teal - NOT orange!)
        # =================================================================
        button_primary_background_fill="#5BA3A3",  # Soft teal
        button_primary_background_fill_dark="#5BA3A3",  # Soft teal (same for dark)
        button_primary_background_fill_hover="#6BB8B8",  # Lighter teal on hover
        button_primary_background_fill_hover_dark="#6BB8B8",
        button_primary_text_color="#FFFFFF",  # White text
        button_primary_text_color_dark="#FFFFFF",  # White text
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
        # =================================================================
        # SECONDARY BUTTONS (Steel Blue Gray)
        # =================================================================
        button_secondary_background_fill=LIGHT_COLORS["surface"],
        button_secondary_background_fill_dark=GRAYS["gray4"],  # Steel blue #3A4554
        button_secondary_text_color=LIGHT_COLORS["text_primary"],
        button_secondary_text_color_dark=DARK_COLORS["text_primary"],
        button_secondary_border_color=LIGHT_COLORS["border"],
        button_secondary_border_color_dark=DARK_COLORS["border"],
        # =================================================================
        # SHADOWS (Subtle for Ocean Mist)
        # =================================================================
        block_shadow="0 1px 3px rgba(0, 0, 0, 0.06)",
        block_shadow_dark="0 1px 4px rgba(0, 0, 0, 0.3)",
        # =================================================================
        # INPUT FOCUS STATES (Pastel Teal)
        # =================================================================
        input_border_color_focus=ACCENT["blue_light"],
        input_border_color_focus_dark=ACCENT["blue"],
        # =================================================================
        # LINK COLORS (Powder Blue)
        # =================================================================
        link_text_color=ACCENT["blue_light"],
        link_text_color_dark=ACCENT["teal"],  # Powder blue #A8C5E2
        link_text_color_hover="#6BB8B8",
        link_text_color_hover_dark="#8AD4D4",
        # =================================================================
        # FORM ELEMENTS (Pastel Teal)
        # =================================================================
        checkbox_background_color_selected=ACCENT["blue_light"],
        checkbox_background_color_selected_dark=ACCENT["blue"],
        slider_color=ACCENT["blue_light"],
        slider_color_dark=ACCENT["blue"],
    )

    return theme


def get_css() -> str:
    """
    Get the custom CSS for the theme.

    Returns:
        str: CSS string with all custom styles
    """
    return CUSTOM_CSS


def get_theme_info() -> dict:
    """
    Get information about the current theme configuration.

    Returns:
        dict: Theme configuration details
    """
    return {
        "name": "Comfy Headless - Ocean Mist",
        "version": "2.1.0",
        "light_mode": "Ocean Mist Light",
        "dark_mode": "Ocean Mist Dark",
        "primary_color_light": ACCENT["blue_light"],
        "primary_color_dark": ACCENT["blue"],
        "features": [
            "Ocean Mist pastel colors",
            "Pastel teal & seafoam accents",
            "SF Pro font stack",
            "Deep slate dark mode",
            "Apple-style buttons and inputs",
            "Apple 8pt spacing grid",
            "Reduced motion support",
        ],
        "colors": {
            "light": LIGHT_COLORS,
            "dark": DARK_COLORS,
            "accent": ACCENT,
            "semantic": SEMANTIC,
        },
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
