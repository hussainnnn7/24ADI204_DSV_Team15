"""
Stage 6: Paradox Country Comparator
===================================
Module containing:
- Multi-Country Selection (2 to 4 countries with presets and reset)
- Part 1: Head-to-Head KPI Comparison Cards (Life Expectancy, GDP, Infant Mort, Internet)
- Part 2: Wealth vs. Health Scatter with Global Background & Residual Connector Lines
- Part 3: Health & Demographic Grouped Bar Chart (Infant Mortality, Fertility, Homicide, Urbanization)
- Part 4: Gender & Education Grouped Bar Chart (Male vs Female Life, Gap, Enrollment)
- Part 5: Multidimensional Radar Chart (0-100 Min-Max Normalized against full 192 nations)
- Part 6: Empirical Paradox Detection & Residual Analysis Table
- CSV Download Feature for Comparative Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional

from utils import (
    NAVY_PRIMARY,
    NAVY_LIGHT,
    TEAL_ACCENT,
    TEAL_LIGHT,
    TEAL_BG,
    BORDER_COLOR,
    CARD_BG,
    render_kpi,
)

# Color palette for up to 4 compared countries
COMPARATOR_PALETTE = [
    "#0F2B48",  # Country 1: Deep Navy Blue
    "#0D9488",  # Country 2: Primary Teal
    "#E11D48",  # Country 3: Vivid Rose / Crimson
    "#7C3AED",  # Country 4: Deep Purple / Violet
]

# Preset comparative scenarios
PRESET_SCENARIOS: Dict[str, List[str]] = {
    "The Mid-Income Paradox: Cuba vs. Equatorial Guinea": [
        "Cuba", "Equatorial Guinea"
    ],
    "High-Income Healthcare Efficiency: Japan vs. United States": [
        "Japan", "United States"
    ],
    "Emerging Overperformers & Underperformers: Sri Lanka vs. Costa Rica vs. Nigeria": [
        "Sri Lanka", "Costa Rica", "Nigeria"
    ],
    "Nordic Longevity Benchmarks: Denmark vs. Sweden vs. Finland": [
        "Denmark", "Sweden", "Finland"
    ]
}

# Radar chart dimensions and normalization rules
RADAR_DIMENSIONS = [
    {
        "key": "avg_life_expectancy",
        "label": "Longevity",
        "reverse": False,
        "unit": "yrs",
        "desc": "Combined life expectancy at birth (higher is better)"
    },
    {
        "key": "log_gdp",
        "label": "Economic Wealth",
        "reverse": False,
        "unit": "log10($)",
        "desc": "Log10 GDP per capita (higher is better)"
    },
    {
        "key": "infant_mortality",
        "label": "Child Survival",
        "reverse": True,
        "unit": "per 1,000",
        "desc": "Infant survival rate (reversed: lower mortality yields higher score)"
    },
    {
        "key": "homicide_rate",
        "label": "Public Safety",
        "reverse": True,
        "unit": "per 100k",
        "desc": "Public safety score (reversed: lower homicide yields higher score)"
    },
    {
        "key": "urban_population",
        "label": "Urban Modernization",
        "reverse": False,
        "unit": "%",
        "desc": "Urban population share (higher is better)"
    },
    {
        "key": "internet_users",
        "label": "Digital Connectivity",
        "reverse": False,
        "unit": "%",
        "desc": "Internet usage penetration % (higher is better)"
    },
    {
        "key": "secondary_school_enrollment_female",
        "label": "Female Education",
        "reverse": False,
        "unit": "%",
        "desc": "Female secondary school enrollment rate (higher is better)"
    }
]


def fit_global_gdp_model(
    df: pd.DataFrame,
    log_scale: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits an OLS regression model of Life Expectancy as a function of GDP per capita across the full 192 nations.
    Returns:
    - Augmented dataframe with predicted_life_exp and residual
    - Dictionary with model statistics (slope, intercept, r2, pearson_r)
    """
    df_mod = df.copy()

    if "log_gdp" not in df_mod.columns:
        df_mod["log_gdp"] = np.log10(df_mod["gdp_per_capita"].clip(lower=1.0))

    x_col = "log_gdp" if log_scale else "gdp_per_capita"
    x = df_mod[x_col].astype(float)
    y = df_mod["avg_life_expectancy"].astype(float)

    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    r2 = r_val ** 2

    df_mod["predicted_life_exp"] = intercept + slope * x
    df_mod["residual"] = df_mod["avg_life_expectancy"] - df_mod["predicted_life_exp"]

    # Classify paradox behavior
    def classify_residual(res: float) -> str:
        if res >= 3.0:
            return "High-Efficiency Overperformer"
        elif res <= -3.0:
            return "Longevity Penalty Underperformer"
        else:
            return "Expected Alignment"

    df_mod["paradox_classification"] = df_mod["residual"].apply(classify_residual)

    model_stats = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "pearson_r": float(r_val),
        "p_val": float(p_val),
        "std_err": float(std_err),
        "log_scale": log_scale
    }

    return df_mod, model_stats


def build_comparator_scatter_figure(
    df_augmented: pd.DataFrame,
    selected_countries: List[str],
    log_scale: bool = True,
    model_stats: Dict[str, Any] = None
) -> go.Figure:
    """
    Part 2: Scatter plot of Wealth vs. Health showing:
    - All 192 countries as muted background points
    - Global OLS regression curve
    - Selected 2-4 countries prominently highlighted
    - Vertical residual connector lines illustrating positive/negative paradox gaps
    """
    if df_augmented.empty or not selected_countries:
        fig = go.Figure()
        fig.update_layout(title="No countries selected for comparison.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    fig = go.Figure()

    # 1. Background trace: All 192 countries
    bg_df = df_augmented[~df_augmented["name"].isin(selected_countries)]
    fig.add_trace(
        go.Scatter(
            x=bg_df["gdp_per_capita"],
            y=bg_df["avg_life_expectancy"],
            mode="markers",
            marker=dict(size=7, color="#CBD5E1", opacity=0.45),
            name="Global Baseline (192 Nations)",
            text=bg_df["name"],
            customdata=np.stack([bg_df["region"], bg_df["residual"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b> (%{customdata[0]})<br>"
                "• GDP per Capita: <b>$%{x:,.0f}</b><br>"
                "• Life Expectancy: <b>%{y:.1f} yrs</b><br>"
                "• GDP-Model Residual: <b>%{customdata[1]:+.2f} yrs</b><br>"
                "<extra></extra>"
            )
        )
    )

    # 2. Global OLS Regression Curve
    if model_stats:
        if log_scale:
            min_gdp = max(df_augmented["gdp_per_capita"].min(), 100.0)
            max_gdp = df_augmented["gdp_per_capita"].max()
            x_curve = np.logspace(np.log10(min_gdp), np.log10(max_gdp), 150)
            y_curve = model_stats["intercept"] + model_stats["slope"] * np.log10(x_curve)
        else:
            min_gdp = df_augmented["gdp_per_capita"].min()
            max_gdp = df_augmented["gdp_per_capita"].max()
            x_curve = np.linspace(min_gdp, max_gdp, 150)
            y_curve = model_stats["intercept"] + model_stats["slope"] * x_curve

        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                name=f"Global Regression Benchmark (R² = {model_stats['r2']:.2f})",
                line=dict(color=NAVY_PRIMARY, width=2.4, dash="solid"),
                hoverinfo="skip"
            )
        )

    # 3. Selected countries highlighted with residual connector lines
    sub_df = df_augmented[df_augmented["name"].isin(selected_countries)].copy()

    for idx, (_, row) in enumerate(sub_df.iterrows()):
        color = COMPARATOR_PALETTE[idx % len(COMPARATOR_PALETTE)]

        # Vertical connector line from predicted to actual
        fig.add_trace(
            go.Scatter(
                x=[row["gdp_per_capita"], row["gdp_per_capita"]],
                y=[row["predicted_life_exp"], row["avg_life_expectancy"]],
                mode="lines",
                line=dict(color=color, width=2.5, dash="dot"),
                name=f"{row['name']} Residual ({row['residual']:+.1f}y)",
                hoverinfo="skip",
                showlegend=False
            )
        )

        # Predicted reference marker
        fig.add_trace(
            go.Scatter(
                x=[row["gdp_per_capita"]],
                y=[row["predicted_life_exp"]],
                mode="markers",
                marker=dict(symbol="x", size=10, color=color, line=dict(width=2)),
                name=f"{row['name']} Predicted",
                hoverinfo="skip",
                showlegend=False
            )
        )

        # Actual country bubble
        res_sign = "+" if row["residual"] >= 0 else ""
        label_text = f"<b>{row['name']}</b> ({res_sign}{row['residual']:.1f}y)"
        fig.add_trace(
            go.Scatter(
                x=[row["gdp_per_capita"]],
                y=[row["avg_life_expectancy"]],
                mode="markers+text",
                text=[label_text],
                textposition="top center",
                textfont=dict(size=11, color=color, family="Plus Jakarta Sans"),
                marker=dict(
                    size=22,
                    color=color,
                    opacity=0.92,
                    line=dict(color="#FFFFFF", width=2.5)
                ),
                name=f"{row['name']} (Actual)",
                customdata=np.stack([[
                    row["region"],
                    row["predicted_life_exp"],
                    row["residual"],
                    row["paradox_classification"]
                ]], axis=0),
                hovertemplate=(
                    f"<b>{row['name']}</b> (%{{customdata[0]}})<br>"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
                    "• Actual Life Expectancy: <b>%{y:.1f} yrs</b><br>"
                    "• Predicted from GDP: <b>%{customdata[1]:.1f} yrs</b><br>"
                    "• <b>Paradox Residual: %{customdata[2]:+.2f} yrs</b><br>"
                    "• Classification: <b>%{customdata[3]}</b><br>"
                    "• GDP per Capita: <b>$%{x:,.0f}</b><br>"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Wealth vs. Health: Selected Nations Against Global Baseline</b> (Residual Divergence)",
            font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title="GDP per Capita (Nominal US$)",
            type="log" if log_scale else "linear",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickprefix="$"
        ),
        yaxis=dict(
            title="Average Life Expectancy at Birth (Years)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            ticksuffix=" yrs"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=530,
        margin=dict(l=60, r=40, t=55, b=65),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#334155")
        )
    )

    return fig


def build_health_demographics_comparison_figure(
    selected_df: pd.DataFrame
) -> go.Figure:
    """
    Part 3: Grouped bar chart comparing the 2 to 4 selected countries on:
    - Infant mortality (deaths / 1k)
    - Fertility (births / woman)
    - Homicide rate (per 100k)
    - Urban population (%)
    """
    if selected_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    categories = [
        "Infant Mortality<br>(per 1,000)",
        "Fertility Rate<br>(births/woman)",
        "Homicide Rate<br>(per 100k)",
        "Urban Population<br>(% of total)"
    ]

    fig = go.Figure()

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        color = COMPARATOR_PALETTE[idx % len(COMPARATOR_PALETTE)]
        vals = [
            row["infant_mortality"],
            row["fertility"],
            row["homicide_rate"],
            row["urban_population"]
        ]
        text_labels = [
            f"{row['infant_mortality']:.1f}",
            f"{row['fertility']:.2f}",
            f"{row['homicide_rate']:.1f}",
            f"{row['urban_population']:.1f}%"
        ]

        fig.add_trace(
            go.Bar(
                name=row["name"],
                x=categories,
                y=vals,
                marker_color=color,
                text=text_labels,
                textposition="outside",
                textfont=dict(size=10.5),
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    "• Indicator: %{x}<br>"
                    "• Value: <b>%{y:.2f}</b><br>"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Health & Demographic Drivers: Head-to-Head Comparison</b>",
            font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(title="", tickfont=dict(size=11, color="#334155")),
        yaxis=dict(title="Metric Value (see category units)", showgrid=True, gridcolor="#F1F5F9"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=480,
        margin=dict(l=50, r=30, t=55, b=65),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#334155")
        )
    )

    return fig


def build_gender_education_comparison_figure(
    selected_df: pd.DataFrame
) -> go.Figure:
    """
    Part 4: Grouped bar chart comparing the 2 to 4 selected countries on:
    - Male Life Expectancy
    - Female Life Expectancy
    - Longevity Gender Gap (F - M)
    - Male Secondary School Enrollment Rate
    - Female Secondary School Enrollment Rate
    """
    if selected_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    categories = [
        "Male Life Exp<br>(Years)",
        "Female Life Exp<br>(Years)",
        "Longevity Gap<br>(F - M yrs)",
        "Male Sec. Edu<br>(Enrollment %)",
        "Female Sec. Edu<br>(Enrollment %)"
    ]

    fig = go.Figure()

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        color = COMPARATOR_PALETTE[idx % len(COMPARATOR_PALETTE)]
        vals = [
            row["life_expectancy_male"],
            row["life_expectancy_female"],
            row["female_male_gap"],
            row["secondary_school_enrollment_male"],
            row["secondary_school_enrollment_female"]
        ]
        text_labels = [
            f"{row['life_expectancy_male']:.1f}y",
            f"{row['life_expectancy_female']:.1f}y",
            f"+{row['female_male_gap']:.2f}y",
            f"{row['secondary_school_enrollment_male']:.1f}%",
            f"{row['secondary_school_enrollment_female']:.1f}%"
        ]

        fig.add_trace(
            go.Bar(
                name=row["name"],
                x=categories,
                y=vals,
                marker_color=color,
                text=text_labels,
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    "• Indicator: %{x}<br>"
                    "• Value: <b>%{y:.2f}</b><br>"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Gender & Educational Disparities: Head-to-Head Comparison</b>",
            font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(title="", tickfont=dict(size=10.5, color="#334155")),
        yaxis=dict(title="Value (Years / %)", showgrid=True, gridcolor="#F1F5F9"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=480,
        margin=dict(l=50, r=30, t=55, b=65),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#334155")
        )
    )

    return fig


def build_multidimensional_radar_figure(
    full_df: pd.DataFrame,
    selected_df: pd.DataFrame
) -> go.Figure:
    """
    Part 5: Multidimensional Plotly Radar Chart comparing selected countries.
    Normalizes indicators to a 0-100 scale using the FULL dataset's min-max values.
    Reverses indicators where lower is better (Infant Mortality, Homicide Rate).
    Defensively handles constant columns.
    """
    if selected_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # Ensure log_gdp is calculated on full_df
    df_calc = full_df.copy()
    if "log_gdp" not in df_calc.columns:
        df_calc["log_gdp"] = np.log10(df_calc["gdp_per_capita"].clip(lower=1.0))

    # Precalculate min-max bounds across the full 192 nations
    bounds: Dict[str, Tuple[float, float]] = {}
    for dim in RADAR_DIMENSIONS:
        k = dim["key"]
        c_min = float(df_calc[k].min())
        c_max = float(df_calc[k].max())
        bounds[k] = (c_min, c_max)

    fig = go.Figure()
    dimension_labels = [dim["label"] for dim in RADAR_DIMENSIONS]

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        color = COMPARATOR_PALETTE[idx % len(COMPARATOR_PALETTE)]
        scores = []

        for dim in RADAR_DIMENSIONS:
            k = dim["key"]
            val = float(np.log10(max(row["gdp_per_capita"], 1.0))) if k == "log_gdp" else float(row[k])
            c_min, c_max = bounds[k]

            # Defensive check for constant columns
            if c_max == c_min:
                score = 50.0
            else:
                if dim["reverse"]:
                    # Lower value is better -> 100 corresponds to min value
                    score = 100.0 * (c_max - val) / (c_max - c_min)
                else:
                    # Higher value is better -> 100 corresponds to max value
                    score = 100.0 * (val - c_min) / (c_max - c_min)

            scores.append(float(np.clip(score, 0.0, 100.0)))

        # Close the radar polygon
        r_vals = scores + [scores[0]]
        theta_vals = dimension_labels + [dimension_labels[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                name=row["name"],
                line=dict(color=color, width=2.4),
                opacity=0.6,
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    "• Dimension: %{theta}<br>"
                    "• Normalized Score: <b>%{r:.1f} / 100</b><br>"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=9.5, color="#64748B"),
                gridcolor="#E2E8F0"
            ),
            angularaxis=dict(
                direction="clockwise",
                tickfont=dict(size=11, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                gridcolor="#E2E8F0"
            )
        ),
        title=dict(
            text="<b>Multidimensional Development Radar (Normalized 0–100 Scale)</b>",
            font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=520,
        margin=dict(l=60, r=60, t=55, b=65),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#334155")
        )
    )

    return fig


def render_stage6_page(df: pd.DataFrame):
    """
    Main renderer for Stage 6: Paradox Country Comparator.
    """
    # -------------------------------------------------------------------------
    # HERO HEADER
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <span class="badge" style="background-color: rgba(13, 148, 136, 0.15); color: {TEAL_ACCENT}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(13, 148, 136, 0.3); text-transform: uppercase;">
                STAGE 6 DELIVERABLE • PARADOX COUNTRY COMPARATOR
            </span>
            <h1 style="color: {NAVY_PRIMARY}; font-size: 2.1rem; font-weight: 800; margin: 0.5rem 0 0.3rem 0; letter-spacing: -0.02em;">
                Head-to-Head Paradox Country Comparator
            </h1>
            <p style="color: #475569; font-size: 1.05rem; margin: 0; line-height: 1.5;">
                Select 2 to 4 sovereign nations to directly interrogate the Longevity Paradox: 
                compare health returns on economic wealth, empirical model residuals, and multidimensional development profiles.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Fit global regression model dynamically across the full 192 nations
    df_augmented, model_stats = fit_global_gdp_model(df, log_scale=True)

    all_country_names = sorted(df["name"].unique().tolist())

    # -------------------------------------------------------------------------
    # 1. INTERACTIVE COMPARATOR CONTROLS
    # -------------------------------------------------------------------------
    if "stage6_selected_countries" not in st.session_state:
        st.session_state["stage6_selected_countries"] = ["Cuba", "Equatorial Guinea", "United States"]
    if "stage6_log_scale" not in st.session_state:
        st.session_state["stage6_log_scale"] = True

    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                <strong style="color: {NAVY_PRIMARY}; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em;">
                    🎛️ Country Selection & Comparator Presets
                </strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        c_col1, c_col2, c_col3, c_col4 = st.columns([3, 2.5, 1.3, 1])

        with c_col1:
            selected_countries = st.multiselect(
                "Select 2 to 4 Nations to Compare:",
                options=all_country_names,
                default=st.session_state["stage6_selected_countries"],
                max_selections=4,
                key="stage6_countries_widget"
            )

        with c_col2:
            preset_choice = st.selectbox(
                "Quick Benchmark Presets:",
                options=["(Custom Selection)"] + list(PRESET_SCENARIOS.keys()),
                index=0,
                key="stage6_preset_widget"
            )
            # Apply preset if changed
            if preset_choice != "(Custom Selection)":
                preset_list = PRESET_SCENARIOS[preset_choice]
                if preset_list != selected_countries:
                    st.session_state["stage6_selected_countries"] = preset_list
                    st.rerun()

        with c_col3:
            log_scale_toggle = st.checkbox(
                "Log10 GDP Scale",
                value=st.session_state["stage6_log_scale"],
                key="stage6_log_widget"
            )

        with c_col4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Presets", use_container_width=True):
                st.session_state["stage6_selected_countries"] = ["Cuba", "Equatorial Guinea", "United States"]
                st.session_state["stage6_log_scale"] = True
                st.rerun()

    # Update state
    st.session_state["stage6_selected_countries"] = selected_countries
    st.session_state["stage6_log_scale"] = log_scale_toggle

    # Validate 2 to 4 selection count
    n_sel = len(selected_countries)
    if n_sel < 2:
        st.warning("⚠️ **Selection Notice:** Please select at least **2 countries** (up to 4) to enable head-to-head comparative analytics.")
        return
    elif n_sel > 4:
        st.error("⚠️ **Maximum Exceeded:** Please select at most **4 countries** for visual legibility.")
        selected_countries = selected_countries[:4]

    # Filter augmented dataframe for selected countries
    selected_df = df_augmented[df_augmented["name"].isin(selected_countries)].copy()

    # Re-order according to user selection order
    selected_df["order"] = selected_df["name"].apply(lambda n: selected_countries.index(n))
    selected_df = selected_df.sort_values("order").drop(columns=["order"])

    # -------------------------------------------------------------------------
    # PART 1: SIDE-BY-SIDE KPI CARDS
    # -------------------------------------------------------------------------
    st.markdown("### Head-to-Head Country Profile Cards")
    kpi_cols = st.columns(n_sel)

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        color = COMPARATOR_PALETTE[idx % len(COMPARATOR_PALETTE)]
        with kpi_cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-top-color: {color};">
                    <div class="kpi-title">
                        <span style="font-size: 1rem; color: {color}; font-weight: 800;">{row['name']}</span>
                        <span class="kpi-tag" style="background-color: {color}15; color: {color}; border-color: {color}30;">
                            {row['region']}
                        </span>
                    </div>
                    <div class="kpi-value" style="color: {color}; font-size: 1.9rem; margin-top: 0.5rem;">
                        {row['avg_life_expectancy']:.1f} <span style="font-size: 1rem; font-weight: 600;">yrs</span>
                    </div>
                    <div class="kpi-subtitle" style="margin-top: 0.6rem; line-height: 1.6;">
                        • <strong>GDP per Capita:</strong> ${row['gdp_per_capita']:,.0f}<br>
                        • <strong>Infant Mortality:</strong> {row['infant_mortality']:.1f} per 1k<br>
                        • <strong>Internet Penetration:</strong> {row['internet_users']:.1f}%<br>
                        • <strong>Gender Gap (F - M):</strong> +{row['female_male_gap']:.2f} yrs
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-bottom: 1.8rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PART 6: PARADOX DETECTION & EMPIRICAL RESIDUAL TABLE
    # -------------------------------------------------------------------------
    st.markdown("### Empirical Paradox Detection & Wealth-Model Residuals")
    st.caption("Fitted against global log-GDP model: Life Expectancy = 36.79 + 9.37 × log10(GDP). R² = 64.1%.")

    paradox_cols = st.columns(n_sel)
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        color = COMPARATOR_PALETTE[idx % len(COMPARATOR_PALETTE)]
        res = row["residual"]
        res_sign = "+" if res >= 0 else ""

        if res >= 3.0:
            status_badge = "🟢 High Health Efficiency (Positive Paradox)"
            status_desc = f"Achieves <strong>+{res:.1f} years</strong> longer life than predicted solely by national income."
        elif res <= -3.0:
            status_badge = "🔴 Longevity Penalty (Negative Paradox)"
            status_desc = f"Lags by <strong>{res:.1f} years</strong> behind the lifespan predicted by its per-capita GDP."
        else:
            status_badge = "🔵 Expected Economic Alignment"
            status_desc = f"Lifespan matches the expected global economic trajectory within ±3 years ({res_sign}{res:.1f} yrs)."

        with paradox_cols[idx]:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 10px; padding: 1.1rem; border-left: 4px solid {color}; height: 100%;">
                    <strong style="color: {color}; font-size: 0.95rem;">{row['name']}</strong><br>
                    <small style="color: #64748B;">{status_badge}</small>
                    <div style="margin: 0.6rem 0; font-size: 0.88rem; color: #334155;">
                        • <strong>Actual:</strong> {row['avg_life_expectancy']:.1f} yrs<br>
                        • <strong>Predicted:</strong> {row['predicted_life_exp']:.1f} yrs<br>
                        • <strong>Residual Gap:</strong> <span style="font-weight: 800; color: {color};">{res_sign}{res:.2f} yrs</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #475569; line-height: 1.4;">
                        {status_desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MULTI-VIEW COMPARATOR VISUALIZATIONS
    # -------------------------------------------------------------------------
    st.markdown("### Comparative Analytical Visualizations")

    view_tabs = [
        "📈 Part 2: Wealth vs. Health Residuals",
        "🩺 Part 3: Health & Demographics",
        "🚻 Part 4: Gender & Educational Parity",
        "🕸️ Part 5: Multidimensional Radar Benchmark"
    ]

    selected_view = st.radio(
        "Select Comparative Visual Dimension:",
        view_tabs,
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # VIEW 1: PART 2 - SCATTER WITH RESIDUALS
    if selected_view == "📈 Part 2: Wealth vs. Health Residuals":
        col_plot2, col_info2 = st.columns([3, 2])
        with col_plot2:
            fig_scatter = build_comparator_scatter_figure(
                df_augmented,
                selected_countries,
                log_scale=log_scale_toggle,
                model_stats=model_stats
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_info2:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>📈 Interpreting Empirical Residuals</h3>
                    <p>
                        The solid navy line displays the <strong>Global Preston Curve</strong> fitted to all 192 sovereign nations. 
                        The vertical dotted lines connect each selected nation's actual lifespan to its GDP-predicted benchmark:
                    </p>
                    <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                        <li>
                            <strong>Positive Residual (Above Line):</strong> Represents <em>health efficiency</em>. 
                            The society converts modest wealth into high longevity, typical of nations investing in universal 
                            primary care (e.g. Cuba, Costa Rica, Sri Lanka).
                        </li>
                        <li>
                            <strong>Negative Residual (Below Line):</strong> Represents a <em>longevity penalty</em>. 
                            Despite substantial per-capita income, life expectancy remains truncated by high infant mortality, 
                            social inequality, or localized disease burdens (e.g. Equatorial Guinea).
                        </li>
                    </ul>
                    <div style="background-color: #FEF2F2; border-left: 3.5px solid #EF4444; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.84rem; color: #991B1B;">
                        <strong>Academic Non-Causal Caveat:</strong> Residuals do not prove that wealth or any single factor 'causes' longevity. 
                        Rather, they highlight the substantial portion of human lifespan variation left unexplained by national income alone.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # VIEW 2: PART 3 - HEALTH & DEMOGRAPHICS
    elif selected_view == "🩺 Part 3: Health & Demographics":
        col_plot3, col_info3 = st.columns([3, 2])
        with col_plot3:
            fig_health = build_health_demographics_comparison_figure(selected_df)
            st.plotly_chart(fig_health, use_container_width=True)

        with col_info3:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🩺 Underlying Health Infrastructure</h3>
                    <p>
                        Differences in infant mortality and public safety explain the vast majority of the observed longevity divergence:
                    </p>
                    <p>
                        When infant mortality is high (e.g., above 40 per 1,000 live births), overall life expectancy at birth drops 
                        precipitously, because neonatal deaths mathematically pull down the national average more severely than deaths at older ages.
                    </p>
                    <div class="paradox-callout">
                        <strong>The Urban Advantage vs. Homicide Penalty:</strong><br>
                        While urbanization provides proximity to emergency medical networks, high homicide rates can create localized 
                        premature mortality drags that reduce the potential years of life lost among young working-age cohorts.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # VIEW 3: PART 4 - GENDER & EDUCATION
    elif selected_view == "🚻 Part 4: Gender & Educational Parity":
        col_plot4, col_info4 = st.columns([3, 2])
        with col_plot4:
            fig_gender = build_gender_education_comparison_figure(selected_df)
            st.plotly_chart(fig_gender, use_container_width=True)

        with col_info4:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🚻 Gender Disparities & Schooling Parity</h3>
                    <p>
                        Comparing sex-disaggregated lifespans alongside secondary education reveals the institutional foundations of welfare:
                    </p>
                    <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                        <li>
                            <strong>Female Longevity Advantage:</strong> Present across all selected nations, but the magnitude 
                            reflects male behavioral/occupational risks vs. maternal mortality protection.
                        </li>
                        <li>
                            <strong>Educational Parity:</strong> Secondary school enrollment for girls acts as a powerful demographic catalyst, 
                            closely correlated with reduced infant mortality, smaller family sizes, and superior maternal health literacy.
                        </li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    # VIEW 4: PART 5 - RADAR CHART
    elif selected_view == "🕸️ Part 5: Multidimensional Radar Benchmark":
        col_plot5, col_info5 = st.columns([3, 2])
        with col_plot5:
            fig_radar = build_multidimensional_radar_figure(df_augmented, selected_df)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_info5:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🕸️ Normalization & Radar Methodology</h3>
                    <p>
                        To enable direct multidimensional comparison across wildly divergent units (years, dollars, percentages), 
                        each axis is mapped to an empirical <strong>0 to 100 scale</strong> benchmarked against the 
                        entire 192-nation dataset:
                    </p>
                    <ul style="color: #334155; font-size: 0.88rem; line-height: 1.6;">
                        <li><strong>Score 100:</strong> Best observed performance globally.</li>
                        <li><strong>Score 0:</strong> Lowest observed performance globally.</li>
                        <li>
                            <strong>Inversion for Negative Indicators:</strong> For <em>Infant Mortality</em> and <em>Homicide Rate</em>, 
                            the scale is inverted: lower deaths yield higher scores, ensuring a larger polygon consistently denotes superior societal health.
                        </li>
                    </ul>
                    <div class="paradox-callout">
                        <strong>Balanced Development:</strong> A country with a wide, symmetrical polygon demonstrates balanced human development, 
                        whereas an asymmetrical polygon highlights specific development bottlenecks (e.g., high GDP with poor child survival).
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # PART 4: EXPORT / DOWNLOAD FEATURE
    # -------------------------------------------------------------------------
    st.markdown("### Export Comparative Dataset")
    st.caption("Download the complete empirical record for the selected nations, including calculated gender gaps and model residuals.")

    # Prepare export dataframe
    export_cols = [
        "name", "region", "iso2", "iso3", "population",
        "avg_life_expectancy", "life_expectancy_female", "life_expectancy_male", "female_male_gap",
        "gdp_per_capita", "predicted_life_exp", "residual", "paradox_classification",
        "infant_mortality", "fertility", "homicide_rate", "urban_population", "internet_users",
        "secondary_school_enrollment_female", "secondary_school_enrollment_male"
    ]
    export_df = selected_df[[c for c in export_cols if c in selected_df.columns]].copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Selected Countries Comparison (CSV)",
        data=csv_bytes,
        file_name=f"paradox_comparison_{'_'.join([c.replace(' ', '_') for c in selected_countries])}.csv",
        mime="text/csv",
        use_container_width=False
    )
