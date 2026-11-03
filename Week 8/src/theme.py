"""
Global Theme Management Module
==============================
Provides centralized Light & Dark theme controls, dynamic CSS injection,
session state persistence, and automated Plotly chart theming.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# Master theme configuration mapping
THEME_CONFIG: Dict[str, Dict[str, str]] = {
    "light": {
        "name": "light",
        "bg": "#F8FAFC",                    # Background: White / very light grey
        "sidebar_bg": "#FFFFFF",            # Sidebar: Pure White
        "card_bg": "#FFFFFF",               # Cards: Pure White
        "primary": "#0F2B48",               # Primary: Deep Navy Blue
        "primary_light": "#1E3A8A",
        "accent": "#0D9488",                # Accent: Primary Teal
        "accent_light": "#14B8A6",
        "accent_bg": "#F0FDFA",             # Tinted Teal bg
        "text_main": "#0F172A",             # Text: Dark navy / charcoal (Slate 900)
        "text_muted": "#64748B",            # Secondary text: Slate 500
        "border": "#E2E8F0",                # Borders: Light grey (Slate 200)
        "callout_bg": "#F8FAFC",
        "callout_border": "#0F2B48",
        "header_bg_start": "#0F2B48",
        "header_bg_end": "#173860",
        "header_badge_bg": "rgba(13, 148, 136, 0.25)",
        "header_badge_text": "#5EEAD4",
        "header_badge_border": "rgba(94, 234, 212, 0.4)",
        "chart_paper_bg": "#FFFFFF",
        "chart_plot_bg": "#FFFFFF",
        "chart_grid": "#F1F5F9",
        "chart_line": "#CBD5E1",
        "chart_text": "#334155",
        "nav_inactive_bg": "#F8FAFC",
        "nav_inactive_hover": "#F1F5F9",
        "nav_inactive_text": "#0F2B48",
        "nav_active_bg": "#F0FDFA",
        "nav_active_text": "#0F2B48",
        "nav_active_border": "#0D9488",
        "status_box_bg": "#F0FDF4",
        "status_box_border": "#BBF7D0",
        "status_box_text": "#166534"
    },
    "dark": {
        "name": "dark",
        "bg": "#0B1220",                    # Background: #0B1220
        "sidebar_bg": "#111827",            # Sidebar: #111827
        "card_bg": "#1E293B",               # Cards: #1E293B
        "primary": "#38BDF8",               # Primary: #38BDF8 (Sky Blue)
        "primary_light": "#7DD3FC",
        "accent": "#2DD4BF",                # Accent: #2DD4BF (Teal)
        "accent_light": "#5EEAD4",
        "accent_bg": "rgba(45, 212, 191, 0.15)",
        "text_main": "#F1F5F9",             # Text: #F1F5F9
        "text_muted": "#CBD5E1",            # Secondary text: #CBD5E1
        "border": "#334155",                # Borders: #334155
        "callout_bg": "#162032",
        "callout_border": "#38BDF8",
        "header_bg_start": "#111827",
        "header_bg_end": "#1E293B",
        "header_badge_bg": "rgba(45, 212, 191, 0.2)",
        "header_badge_text": "#2DD4BF",
        "header_badge_border": "rgba(45, 212, 191, 0.4)",
        "chart_paper_bg": "#1E293B",
        "chart_plot_bg": "#1E293B",
        "chart_grid": "#334155",
        "chart_line": "#475569",
        "chart_text": "#F1F5F9",
        "nav_inactive_bg": "#162032",
        "nav_inactive_hover": "#1E293B",
        "nav_inactive_text": "#E2E8F0",
        "nav_active_bg": "rgba(45, 212, 191, 0.18)",
        "nav_active_text": "#38BDF8",
        "nav_active_border": "#2DD4BF",
        "status_box_bg": "#064E3B",
        "status_box_border": "#059669",
        "status_box_text": "#D1FAE5"
    }
}


def get_current_theme() -> str:
    """Return the active theme name ('light' or 'dark'), defaulting to 'light'."""
    return st.session_state.get("global_theme", "light")


def get_theme_colors() -> Dict[str, str]:
    """Return the color dictionary for the currently active theme."""
    theme_name = get_current_theme()
    return THEME_CONFIG.get(theme_name, THEME_CONFIG["light"])


def render_theme_switcher():
    """
    Renders an interactive, clearly visible Light / Dark segmented toggle in the sidebar.
    Persists user selection across navigation via st.session_state.
    """
    current_theme = get_current_theme()
    c = get_theme_colors()

    st.markdown(
        f"""
        <div style="margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {c['text_muted']};">
                🎨 Visual Theme
            </span>
            <span style="font-size: 0.72rem; padding: 0.1rem 0.45rem; border-radius: 9999px; background: {c['accent_bg']}; color: {c['accent']}; font-weight: 700;">
                {current_theme.upper()}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    theme_choice = st.radio(
        "Dashboard Theme Switcher",
        options=["☀️ Light", "🌙 Dark"],
        index=0 if current_theme == "light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="global_theme_switcher_widget"
    )

    selected_theme = "light" if "Light" in theme_choice else "dark"
    if selected_theme != current_theme:
        st.session_state["global_theme"] = selected_theme
        st.rerun()


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """
    Adapts any Plotly figure to the active theme:
    Updates paper/plot background, text fonts, axes grid and line colors, legends,
    geo maps (Choropleth), and polar (Radar) backgrounds.
    """
    if not isinstance(fig, go.Figure):
        return fig

    c = get_theme_colors()
    theme = get_current_theme()

    # Core background & font
    fig.update_layout(
        paper_bgcolor=c["chart_paper_bg"],
        plot_bgcolor=c["chart_plot_bg"],
        font=dict(color=c["chart_text"], family="Plus Jakarta Sans")
    )

    # Title styling
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(
            title=dict(font=dict(color=c["primary"], family="Plus Jakarta Sans"))
        )

    # Axes styling
    fig.update_xaxes(
        gridcolor=c["chart_grid"],
        linecolor=c["chart_line"],
        tickfont=dict(color=c["chart_text"]),
        title_font=dict(color=c["chart_text"])
    )
    fig.update_yaxes(
        gridcolor=c["chart_grid"],
        linecolor=c["chart_line"],
        tickfont=dict(color=c["chart_text"]),
        title_font=dict(color=c["chart_text"])
    )

    # Legend styling
    fig.update_layout(
        legend=dict(
            font=dict(color=c["chart_text"]),
            title=dict(font=dict(color=c["chart_text"]))
        )
    )

    # Geo layout (Choropleth in Stage 2)
    if hasattr(fig.layout, "geo") and fig.layout.geo is not None:
        if theme == "dark":
            fig.update_geos(
                bgcolor=c["chart_paper_bg"],
                lakecolor=c["chart_paper_bg"],
                landcolor="#0B1220",
                oceancolor=c["chart_paper_bg"],
                coastlinecolor=c["border"],
                countrycolor=c["border"],
                framecolor=c["border"]
            )
        else:
            fig.update_geos(
                bgcolor="#FFFFFF",
                lakecolor="#FFFFFF",
                landcolor="#F8FAFC",
                oceancolor="#FFFFFF",
                coastlinecolor="#CBD5E1",
                countrycolor="#CBD5E1",
                framecolor="#CBD5E1"
            )

    # Polar layout (Radar in Stage 6)
    if hasattr(fig.layout, "polar") and fig.layout.polar is not None:
        fig.update_layout(
            polar=dict(
                bgcolor=c["chart_paper_bg"],
                radialaxis=dict(
                    gridcolor=c["chart_grid"],
                    linecolor=c["chart_line"],
                    tickfont=dict(color=c["text_muted"])
                ),
                angularaxis=dict(
                    gridcolor=c["chart_grid"],
                    linecolor=c["chart_line"],
                    tickfont=dict(color=c["primary"])
                )
            )
        )

    return fig


def patch_streamlit_plotly():
    """
    Patches st.plotly_chart so that EVERY chart across Stages 1-8 automatically
    adapts to the active theme without modifying existing chart functions.
    """
    if getattr(st, "_plotly_patched", False):
        return

    _original_plotly_chart = st.plotly_chart

    def themed_plotly_chart(fig, *args, **kwargs):
        apply_plotly_theme(fig)
        return _original_plotly_chart(fig, *args, **kwargs)

    st.plotly_chart = themed_plotly_chart
    st._plotly_patched = True


def generate_theme_css() -> str:
    """Generates the dynamic CSS based on the active theme."""
    c = get_theme_colors()
    theme = get_current_theme()

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        :root {{
            --bg-main: {c['bg']};
            --sidebar-bg: {c['sidebar_bg']};
            --card-bg: {c['card_bg']};
            --primary-color: {c['primary']};
            --accent-color: {c['accent']};
            --accent-bg: {c['accent_bg']};
            --text-main: {c['text_main']};
            --text-muted: {c['text_muted']};
            --border-color: {c['border']};
            --callout-bg: {c['callout_bg']};
        }}

        /* Base resets */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
            color: {c['text_main']};
        }}

        /* App Background */
        .stApp {{
            background-color: {c['bg']} !important;
            color: {c['text_main']} !important;
        }}

        /* Universal Text Contrast */
        .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {{
            color: {c['text_main']};
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {c['primary']} !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }}

        /* Streamlit Markdown small & caption text */
        .stApp [data-testid="stMarkdownContainer"] small,
        .stApp [data-testid="stCaptionContainer"] p {{
            color: {c['text_muted']} !important;
        }}

        /* Main Header Block */
        .main-header {{
            background: linear-gradient(135deg, {c['header_bg_start']} 0%, {c['header_bg_end']} 100%) !important;
            padding: 2.2rem 2.4rem;
            border-radius: 16px;
            color: #FFFFFF !important;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
            border-left: 6px solid {c['accent']} !important;
        }}

        .main-header h1 {{
            color: #FFFFFF !important;
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            margin: 0 0 0.5rem 0 !important;
        }}

        .main-header p {{
            color: #E2E8F0 !important;
            font-size: 1.05rem !important;
            margin: 0 !important;
        }}

        .main-header .badge {{
            display: inline-block;
            background-color: {c['header_badge_bg']} !important;
            color: {c['header_badge_text']} !important;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            border: 1px solid {c['header_badge_border']} !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        /* KPI Card Styling */
        .kpi-card {{
            background: {c['card_bg']} !important;
            border: 1px solid {c['border']} !important;
            border-radius: 14px;
            padding: 1.4rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
            position: relative;
            overflow: hidden;
            border-top: 3.5px solid {c['accent']} !important;
            height: 100%;
        }}

        .kpi-title {{
            color: {c['text_muted']} !important;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .kpi-value {{
            color: {c['primary']} !important;
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }}

        .kpi-subtitle {{
            color: {c['text_muted']} !important;
            font-size: 0.82rem;
            font-weight: 500;
            line-height: 1.3;
        }}

        .kpi-tag {{
            display: inline-block;
            background-color: {c['accent_bg']} !important;
            color: {c['accent']} !important;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            border: 1px solid {c['accent']}40 !important;
        }}

        /* Info Card Styling */
        .info-card {{
            background: {c['card_bg']} !important;
            border: 1px solid {c['border']} !important;
            border-radius: 14px;
            padding: 1.8rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
            color: {c['text_main']} !important;
        }}

        .info-card h3 {{
            color: {c['primary']} !important;
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 0.75rem;
        }}

        .info-card p, .info-card li {{
            color: {c['text_main']} !important;
            font-size: 0.96rem;
            line-height: 1.65;
        }}

        /* Callout Box */
        .paradox-callout {{
            background-color: {c['callout_bg']} !important;
            border-left: 4px solid {c['primary']} !important;
            border-right: 1px solid {c['border']} !important;
            border-top: 1px solid {c['border']} !important;
            border-bottom: 1px solid {c['border']} !important;
            border-radius: 0 10px 10px 0;
            padding: 1.2rem 1.4rem;
            margin: 1.2rem 0;
            color: {c['text_main']} !important;
        }}

        .paradox-callout strong {{
            color: {c['primary']} !important;
        }}

        /* Dynamic override for inlined HTML background styles across all stages */
        .stApp div[style*="background-color: #FFFFFF"],
        .stApp div[style*="background-color: rgb(255, 255, 255)"],
        .stApp div[style*="background: #FFFFFF"],
        .stApp div[style*="background: rgb(255, 255, 255)"] {{
            background-color: {c['card_bg']} !important;
            border-color: {c['border']} !important;
            color: {c['text_main']} !important;
        }}

        .stApp div[style*="background-color: #F8FAFC"] {{
            background-color: {c['callout_bg']} !important;
            border-color: {c['border']} !important;
            color: {c['text_main']} !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {c['sidebar_bg']} !important;
            border-right: 1px solid {c['border']} !important;
        }}

        section[data-testid="stSidebar"] .sidebar-branding {{
            padding: 0.8rem 0.5rem 1rem 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid {c['border']};
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {{
            color: {c['primary']} !important;
            font-weight: 800 !important;
            font-size: 0.92rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }}

        section[data-testid="stSidebar"] hr {{
            border-color: {c['border']} !important;
            margin: 1rem 0 !important;
        }}

        /* Inactive Navigation Radio Labels */
        div[data-testid="stRadio"] div[role="radiogroup"] > label {{
            background-color: {c['nav_inactive_bg']} !important;
            border: 1px solid {c['border']} !important;
            border-radius: 8px !important;
            padding: 0.6rem 0.85rem !important;
            cursor: pointer !important;
            transition: all 0.18s ease-in-out !important;
            margin: 0 !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
            background-color: {c['nav_inactive_hover']} !important;
            border-color: {c['accent']} !important;
            transform: translateX(2px) !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label p,
        div[data-testid="stRadio"] div[role="radiogroup"] > label span {{
            color: {c['nav_inactive_text']} !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
        }}

        /* Active Navigation Radio Label */
        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
            background-color: {c['nav_active_bg']} !important;
            border-color: {c['nav_active_border']} !important;
            border-left: 4px solid {c['nav_active_border']} !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {{
            color: {c['nav_active_text']} !important;
            font-weight: 700 !important;
        }}

        /* Radio indicator circle */
        div[data-testid="stRadio"] div[role="radiogroup"] input:checked + div {{
            border-color: {c['accent']} !important;
            background-color: {c['accent']} !important;
        }}

        /* Form widgets (selectboxes, inputs, multiselects) in dark mode */
        {f'''
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input {{
            background-color: {c['card_bg']} !important;
            color: {c['text_main']} !important;
            border-color: {c['border']} !important;
        }}

        .stMultiSelect [data-baseweb="tag"] {{
            background-color: #334155 !important;
            color: #F1F5F9 !important;
            border: 1px solid #475569 !important;
        }}

        .stButton > button {{
            background-color: {c['card_bg']} !important;
            color: {c['text_main']} !important;
            border-color: {c['border']} !important;
            transition: all 0.15s ease-in-out !important;
        }}

        .stButton > button:hover {{
            border-color: {c['accent']} !important;
            color: {c['accent']} !important;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {c['border']} !important;
            border-radius: 8px !important;
        }}
        ''' if theme == 'dark' else ''}
    </style>
    """
    return css

