"""
UI Utilities & Dynamic Theme Engine
===================================
Project: Global Life Expectancy Paradox
Centralized theme styling supporting both Light and Dark modes.
Provides backward-compatible wrappers and explicit re-exports.
"""

import streamlit as st
import theme

# Re-export core theme functions as explicit first-class wrappers
def render_theme_switcher():
    """Renders the Light / Dark theme toggle in the sidebar."""
    return theme.render_theme_switcher()


def get_current_theme() -> str:
    """Returns the current theme ('light' or 'dark')."""
    return theme.get_current_theme()


def get_theme_colors():
    """Returns active theme color dictionary."""
    return theme.get_theme_colors()


def apply_plotly_theme(fig):
    """Applies active theme to Plotly figures."""
    return theme.apply_plotly_theme(fig)


def patch_streamlit_plotly():
    """Ensures st.plotly_chart auto-themes figures."""
    return theme.patch_streamlit_plotly()


def generate_theme_css():
    """Generates dynamic CSS for the active theme."""
    return theme.generate_theme_css()


# Theme configuration mapping
THEME_CONFIG = theme.THEME_CONFIG

# Standard Light theme constants for backwards compatibility with existing imports
NAVY_PRIMARY = "#0F2B48"     # Deep Navy Blue (Light Primary)
NAVY_LIGHT = "#1E3A8A"       # Medium Navy
TEAL_ACCENT = "#0D9488"      # Primary Teal
TEAL_LIGHT = "#14B8A6"       # Light Teal
TEAL_BG = "#F0FDFA"          # Very Light Teal Background tint
BG_LIGHT = "#F8FAFC"         # Page & Container Light Background
CARD_BG = "#FFFFFF"          # Pure White for Cards
TEXT_MAIN = "#0F172A"        # Slate 900 for Primary Text
TEXT_MUTED = "#64748B"       # Slate 500 for Subtitles
BORDER_COLOR = "#E2E8F0"     # Slate 200 for Card Borders


def inject_custom_css():
    """
    Injects dynamic responsive CSS adapting to the active Light or Dark theme.
    Also patches st.plotly_chart so that all Plotly figures in all stages
    automatically inherit the active theme.
    """
    # 1. Patch Plotly so all existing figures automatically theme correctly
    patch_streamlit_plotly()

    # 2. Inject the dynamic CSS rules based on active theme
    css = generate_theme_css()
    st.markdown(css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge: str = "DSV Project Dashboard"):
    """Render a responsive hero header block using active theme colors."""
    c = get_theme_colors()
    html = f"""
    <div class="main-header">
        <span class="badge">{badge}</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(title: str, value: str, subtitle: str = "", tag: str = ""):
    """Render a styled KPI card using active theme colors."""
    c = get_theme_colors()
    tag_html = f'<span class="kpi-tag">{tag}</span>' if tag else ""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">
            <span>{title}</span>
            {tag_html}
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


__all__ = [
    "render_theme_switcher",
    "get_current_theme",
    "get_theme_colors",
    "apply_plotly_theme",
    "patch_streamlit_plotly",
    "generate_theme_css",
    "THEME_CONFIG",
    "inject_custom_css",
    "render_header",
    "render_kpi",
    "NAVY_PRIMARY",
    "NAVY_LIGHT",
    "TEAL_ACCENT",
    "TEAL_LIGHT",
    "TEAL_BG",
    "BG_LIGHT",
    "CARD_BG",
    "TEXT_MAIN",
    "TEXT_MUTED",
    "BORDER_COLOR",
]
