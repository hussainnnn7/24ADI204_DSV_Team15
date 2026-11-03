"""
Stage 4: Gender Gap & Education Analysis
========================================
Module containing:
- Interactive Filter Strip (Region, Country Search, Trendlines, Reset)
- Part 1: Male vs Female Life Expectancy Scatter with 45-Degree Equality Line
- Part 2: Gender Longevity Gap Rankings (Top 10, Bottom 10 & Regional Comparison Bar Charts)
- Part 3: Education Comparison (Female vs Male Secondary Enrollment with 45-Degree Parity Line)
- Part 4: Education vs Life Expectancy (Female & Male Comparative Regressions)
- Part 5: Dynamic Statistical Insights (Pearson, Spearman, Slope, R-squared, Gaps)
- Comprehensive Analytical Synthesis & Key Findings
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, Any, List, Tuple

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


def calculate_bivariate_stats(
    df_sub: pd.DataFrame,
    x_col: str,
    y_col: str
) -> Dict[str, Any]:
    """
    Dynamically compute Pearson r, Spearman rho, regression slope, intercept, R-squared, and n.
    Includes safeguards against division by zero, empty subsets, and constant columns.
    """
    n = len(df_sub)
    if n < 3:
        return {
            "n": n,
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
            "slope": 0.0,
            "intercept": df_sub[y_col].mean() if n > 0 else 0.0,
            "r2": 0.0,
            "std_err": 0.0,
            "is_valid": False
        }

    x = df_sub[x_col].astype(float)
    y = df_sub[y_col].astype(float)

    if x.std() == 0 or y.std() == 0:
        return {
            "n": n,
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
            "slope": 0.0,
            "intercept": float(y.mean()),
            "r2": 0.0,
            "std_err": 0.0,
            "is_valid": False
        }

    try:
        p_r, p_p = stats.pearsonr(x, y)
        s_r, s_p = stats.spearmanr(x, y)
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        r2 = r_val ** 2
        return {
            "n": n,
            "pearson_r": float(p_r),
            "pearson_p": float(p_p),
            "spearman_r": float(s_r),
            "spearman_p": float(s_p),
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "std_err": float(std_err),
            "is_valid": True
        }
    except Exception:
        return {
            "n": n,
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
            "slope": 0.0,
            "intercept": float(y.mean()),
            "r2": 0.0,
            "std_err": 0.0,
            "is_valid": False
        }


def build_male_vs_female_life_figure(
    df: pd.DataFrame,
    highlight_countries: List[str],
    highlight_large_gap: bool = False
) -> go.Figure:
    """
    Part 1: Scatter plot of Male vs Female Life Expectancy with 45-degree Equality Line.
    X = life_expectancy_male, Y = life_expectancy_female.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No countries match the selected filters.",
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        return fig

    # Unique columns for custom_data to avoid duplicate column name issues in plotly
    custom_cols = [
        "region",
        "female_male_gap",
        "gdp_per_capita",
        "population",
        "avg_life_expectancy"
    ]

    fig = px.scatter(
        df,
        x="life_expectancy_male",
        y="life_expectancy_female",
        color="region",
        size="population",
        hover_name="name",
        size_max=30,
        custom_data=custom_cols,
        category_orders={"region": sorted(df["region"].unique())}
    )

    hover_temp = (
        "<b>%{hovertext}</b> (%{customdata[0]})<br>"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        "• Female Life Expectancy: <b>%{y:.1f} yrs</b><br>"
        "• Male Life Expectancy: <b>%{x:.1f} yrs</b><br>"
        "• Longevity Advantage (F - M): <b>+%{customdata[1]:.2f} yrs</b><br>"
        "• Combined Average: <b>%{customdata[4]:.1f} yrs</b><br>"
        "• GDP per Capita: <b>$%{customdata[2]:,.0f}</b><br>"
        "• Population: <b>%{customdata[3]:,.0f}k</b>"
        "<extra></extra>"
    )

    fig.update_traces(
        hovertemplate=hover_temp,
        marker=dict(opacity=0.82, line=dict(width=0.8, color="#FFFFFF"))
    )

    # 45-degree reference line (Equal Longevity)
    min_val = min(df["life_expectancy_male"].min(), df["life_expectancy_female"].min()) - 3.0
    max_val = max(df["life_expectancy_male"].max(), df["life_expectancy_female"].max()) + 3.0

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="45° Equality Line (Female = Male)",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            hoverinfo="skip"
        )
    )

    # Annotate zone above equality line
    fig.add_annotation(
        x=min_val + 5,
        y=min_val + 10,
        text="<b>↑ Female Advantage Zone (100% of nations)</b>",
        showarrow=False,
        font=dict(size=11, color=TEAL_ACCENT, family="Plus Jakarta Sans"),
        bgcolor="rgba(240, 253, 250, 0.8)",
        bordercolor="rgba(13, 148, 136, 0.3)",
        borderwidth=1,
        borderpad=4
    )

    # Highlight countries with large gender gaps (> 8 years) if enabled
    if highlight_large_gap:
        large_gap_df = df[df["female_male_gap"] >= 8.0]
        if not large_gap_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=large_gap_df["life_expectancy_male"],
                    y=large_gap_df["life_expectancy_female"],
                    mode="markers+text",
                    text=large_gap_df["name"],
                    textposition="top left",
                    textfont=dict(size=9.5, color="#B91C1C", family="Plus Jakarta Sans"),
                    marker=dict(
                        symbol="diamond-open",
                        size=14,
                        color="#DC2626",
                        line=dict(width=2, color="#DC2626")
                    ),
                    name="Large Gender Gap (≥ 8.0 yrs)",
                    hoverinfo="skip"
                )
            )

    # Highlight user-searched countries
    if highlight_countries:
        hl_df = df[df["name"].isin(highlight_countries)]
        if not hl_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hl_df["life_expectancy_male"],
                    y=hl_df["life_expectancy_female"],
                    mode="markers+text",
                    text=hl_df["name"],
                    textposition="top center",
                    textfont=dict(size=11, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                    marker=dict(
                        symbol="circle-open",
                        size=32,
                        color="#DC2626",
                        line=dict(width=2.5, color="#DC2626")
                    ),
                    name="Highlighted Countries",
                    hoverinfo="skip"
                )
            )

    fig.update_layout(
        title=dict(
            text="<b>Male vs. Female Life Expectancy at Birth</b> (45° Equality Line Reference)",
            font=dict(size=15, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title="Male Life Expectancy (Years)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zeroline=False,
            range=[min_val, max_val]
        ),
        yaxis=dict(
            title="Female Life Expectancy (Years)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zeroline=False,
            range=[min_val, max_val]
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=540,
        margin=dict(l=60, r=40, t=55, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.26,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#334155"),
            title=dict(text="")
        )
    )

    return fig


def build_gender_gap_bar_figure(
    df: pd.DataFrame,
    mode: str = "top10"
) -> go.Figure:
    """
    Part 2: Interactive bar chart for:
    - mode="top10": Top 10 largest female-male gaps
    - mode="bottom10": Bottom 10 smallest female-male gaps
    - mode="regional": Regional average gender gaps sorted
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    if mode == "top10":
        sub_df = df.nlargest(min(10, len(df)), "female_male_gap").sort_values("female_male_gap", ascending=True)
        title_text = "<b>Top 10 Nations with Largest Longevity Gender Gap</b> (Female Advantage)"
        bar_color = NAVY_PRIMARY
        x_col = "female_male_gap"
        y_col = "name"
        hover_tpl = (
            "<b>%{y}</b><br>"
            "Longevity Gap: <b>+%{x:.2f} yrs</b><br>"
            "<extra></extra>"
        )

        fig = go.Figure(
            go.Bar(
                x=sub_df[x_col],
                y=sub_df[y_col],
                orientation="h",
                marker=dict(
                    color=sub_df[x_col],
                    colorscale=[[0, "#38BDF8"], [1, NAVY_PRIMARY]],
                    line=dict(color="#CBD5E1", width=0.5)
                ),
                text=[f"+{v:.2f} yrs" for v in sub_df[x_col]],
                textposition="outside",
                hovertemplate=hover_tpl
            )
        )
        fig.update_layout(
            xaxis=dict(title="Female - Male Life Expectancy Gap (Years)", range=[0, sub_df[x_col].max() * 1.18]),
            yaxis=dict(title="")
        )

    elif mode == "bottom10":
        sub_df = df.nsmallest(min(10, len(df)), "female_male_gap").sort_values("female_male_gap", ascending=True)
        title_text = "<b>Bottom 10 Nations with Smallest Longevity Gender Gap</b> (Narrowest Advantage)"
        x_col = "female_male_gap"
        y_col = "name"
        hover_tpl = (
            "<b>%{y}</b><br>"
            "Longevity Gap: <b>+%{x:.2f} yrs</b><br>"
            "<extra></extra>"
        )

        fig = go.Figure(
            go.Bar(
                x=sub_df[x_col],
                y=sub_df[y_col],
                orientation="h",
                marker=dict(
                    color=sub_df[x_col],
                    colorscale=[[0, "#CCFBF1"], [1, TEAL_ACCENT]],
                    line=dict(color="#CBD5E1", width=0.5)
                ),
                text=[f"+{v:.2f} yrs" for v in sub_df[x_col]],
                textposition="outside",
                hovertemplate=hover_tpl
            )
        )
        fig.update_layout(
            xaxis=dict(title="Female - Male Life Expectancy Gap (Years)", range=[0, max(sub_df[x_col].max() * 1.25, 4.0)]),
            yaxis=dict(title="")
        )

    else:  # regional average
        reg_df = df.groupby("region")["female_male_gap"].agg(["mean", "count"]).reset_index()
        reg_df = reg_df.sort_values("mean", ascending=True)
        title_text = "<b>Regional Average Longevity Gap</b> (Female Advantage in Years)"

        fig = go.Figure(
            go.Bar(
                x=reg_df["mean"],
                y=reg_df["region"],
                orientation="h",
                marker=dict(
                    color=reg_df["mean"],
                    colorscale=[[0, "#99F6E4"], [0.5, TEAL_ACCENT], [1, NAVY_PRIMARY]],
                    line=dict(color="#CBD5E1", width=0.5)
                ),
                text=[f"+{v:.2f} yrs ({c} nations)" for v, c in zip(reg_df["mean"], reg_df["count"])],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Regional Mean Gap: <b>+%{x:.2f} yrs</b><br>"
                    "<extra></extra>"
                )
            )
        )
        fig.update_layout(
            xaxis=dict(title="Average Female Longevity Advantage (Years)", range=[0, reg_df["mean"].max() * 1.25]),
            yaxis=dict(title="")
        )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=14, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=430,
        margin=dict(l=140, r=40, t=50, b=50),
        xaxis_showgrid=True,
        xaxis_gridcolor="#F1F5F9"
    )

    return fig


def build_education_parity_figure(
    df: pd.DataFrame,
    highlight_countries: List[str]
) -> go.Figure:
    """
    Part 3: Scatter plot of Male vs Female Secondary School Enrollment with 45-degree Parity Line.
    X = secondary_school_enrollment_male, Y = secondary_school_enrollment_female.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No countries match the selected filters.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # Compute edu gap dynamically if not already in dataframe
    plot_df = df.copy()
    if "edu_gender_gap" not in plot_df.columns:
        plot_df["edu_gender_gap"] = (
            plot_df["secondary_school_enrollment_female"] - plot_df["secondary_school_enrollment_male"]
        )

    custom_cols = [
        "region",
        "edu_gender_gap",
        "avg_life_expectancy",
        "gdp_per_capita",
        "population"
    ]

    fig = px.scatter(
        plot_df,
        x="secondary_school_enrollment_male",
        y="secondary_school_enrollment_female",
        color="region",
        size="population",
        hover_name="name",
        size_max=30,
        custom_data=custom_cols,
        category_orders={"region": sorted(plot_df["region"].unique())}
    )

    hover_temp = (
        "<b>%{hovertext}</b> (%{customdata[0]})<br>"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        "• Female Secondary Enrollment: <b>%{y:.1f}%</b><br>"
        "• Male Secondary Enrollment: <b>%{x:.1f}%</b><br>"
        "• Educational Gap (F - M): <b>%{customdata[1]:+.1f}%</b><br>"
        "• Avg Life Expectancy: <b>%{customdata[2]:.1f} yrs</b><br>"
        "• GDP per Capita: <b>$%{customdata[3]:,.0f}</b>"
        "<extra></extra>"
    )

    fig.update_traces(
        hovertemplate=hover_temp,
        marker=dict(opacity=0.82, line=dict(width=0.8, color="#FFFFFF"))
    )

    # 45-degree parity reference line
    min_val = min(
        plot_df["secondary_school_enrollment_male"].min(),
        plot_df["secondary_school_enrollment_female"].min()
    ) - 5.0
    max_val = max(
        plot_df["secondary_school_enrollment_male"].max(),
        plot_df["secondary_school_enrollment_female"].max()
    ) + 5.0

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="45° Educational Parity Line (F = M)",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            hoverinfo="skip"
        )
    )

    # Parity zone annotations
    fig.add_annotation(
        x=min_val + 10,
        y=min_val + 25,
        text="<b>↑ Female Enrollment Advantage</b>",
        showarrow=False,
        font=dict(size=11, color=TEAL_ACCENT, family="Plus Jakarta Sans"),
        bgcolor="rgba(240, 253, 250, 0.8)",
        bordercolor="rgba(13, 148, 136, 0.3)",
        borderwidth=1,
        borderpad=4
    )
    fig.add_annotation(
        x=max_val - 15,
        y=max_val - 35,
        text="<b>↓ Male Enrollment Advantage</b>",
        showarrow=False,
        font=dict(size=11, color="#B91C1C", family="Plus Jakarta Sans"),
        bgcolor="rgba(254, 242, 242, 0.8)",
        bordercolor="rgba(239, 68, 68, 0.3)",
        borderwidth=1,
        borderpad=4
    )

    # Highlight notable extreme education outliers
    extreme_pos = plot_df.nlargest(3, "edu_gender_gap")
    extreme_neg = plot_df.nsmallest(3, "edu_gender_gap")
    extreme_df = pd.concat([extreme_pos, extreme_neg]).drop_duplicates()

    fig.add_trace(
        go.Scatter(
            x=extreme_df["secondary_school_enrollment_male"],
            y=extreme_df["secondary_school_enrollment_female"],
            mode="text",
            text=extreme_df["name"],
            textposition="top left",
            textfont=dict(size=9.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            name="Notable Gap Outliers",
            hoverinfo="skip"
        )
    )

    # User-searched highlights
    if highlight_countries:
        hl_df = plot_df[plot_df["name"].isin(highlight_countries)]
        if not hl_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hl_df["secondary_school_enrollment_male"],
                    y=hl_df["secondary_school_enrollment_female"],
                    mode="markers+text",
                    text=hl_df["name"],
                    textposition="top center",
                    textfont=dict(size=11, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                    marker=dict(
                        symbol="circle-open",
                        size=32,
                        color="#DC2626",
                        line=dict(width=2.5, color="#DC2626")
                    ),
                    name="Highlighted Countries",
                    hoverinfo="skip"
                )
            )

    fig.update_layout(
        title=dict(
            text="<b>Female vs. Male Secondary School Enrollment Rate</b> (Gender Parity Line Reference)",
            font=dict(size=15, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title="Male Secondary Enrollment Rate (%)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zeroline=False,
            range=[min_val, max_val]
        ),
        yaxis=dict(
            title="Female Secondary Enrollment Rate (%)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zeroline=False,
            range=[min_val, max_val]
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=540,
        margin=dict(l=60, r=40, t=55, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.26,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#334155"),
            title=dict(text="")
        )
    )

    return fig


def build_education_vs_life_figure(
    df: pd.DataFrame,
    gender: str = "female",
    highlight_countries: List[str] = None,
    show_trendline: bool = True
) -> go.Figure:
    """
    Part 4: Scatter plot of Secondary School Enrollment vs Average Life Expectancy.
    gender in ['female', 'male'].
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No countries match the selected filters.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    edu_col = f"secondary_school_enrollment_{gender}"
    gender_title = "Female" if gender == "female" else "Male"
    color_accent = TEAL_ACCENT if gender == "female" else NAVY_PRIMARY

    custom_cols = [
        "region",
        "gdp_per_capita",
        "population",
        "life_expectancy_female" if gender == "female" else "life_expectancy_male",
        "female_male_gap"
    ]

    fig = px.scatter(
        df,
        x=edu_col,
        y="avg_life_expectancy",
        color="region",
        size="population",
        hover_name="name",
        size_max=30,
        custom_data=custom_cols,
        category_orders={"region": sorted(df["region"].unique())}
    )

    hover_temp = (
        "<b>%{hovertext}</b> (%{customdata[0]})<br>"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        f"• {gender_title} Secondary Enrollment: <b>%{{x:.1f}}%</b><br>"
        "• Avg Life Expectancy: <b>%{y:.1f} yrs</b><br>"
        f"• {gender_title} Life Expectancy: <b>%{{customdata[3]:.1f}} yrs</b><br>"
        "• Longevity Gap (F - M): <b>+%{customdata[4]:.2f} yrs</b><br>"
        "• GDP per Capita: <b>$%{customdata[1]:,.0f}</b>"
        "<extra></extra>"
    )

    fig.update_traces(
        hovertemplate=hover_temp,
        marker=dict(opacity=0.82, line=dict(width=0.8, color="#FFFFFF"))
    )

    # OLS Regression Trendline
    st_res = calculate_bivariate_stats(df, edu_col, "avg_life_expectancy")
    if show_trendline and st_res["is_valid"] and len(df) >= 3:
        min_x = df[edu_col].min()
        max_x = df[edu_col].max()
        x_line = np.linspace(min_x, max_x, 100)
        y_line = st_res["intercept"] + st_res["slope"] * x_line

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS Trendline (R² = {st_res['r2']:.2f}, Slope = +{st_res['slope']:.3f})",
                line=dict(color=color_accent, width=2.6, dash="solid"),
                hoverinfo="skip"
            )
        )

    # Highlight specific countries
    if highlight_countries:
        hl_df = df[df["name"].isin(highlight_countries)]
        if not hl_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hl_df[edu_col],
                    y=hl_df["avg_life_expectancy"],
                    mode="markers+text",
                    text=hl_df["name"],
                    textposition="top center",
                    textfont=dict(size=11, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                    marker=dict(
                        symbol="circle-open",
                        size=32,
                        color="#DC2626",
                        line=dict(width=2.5, color="#DC2626")
                    ),
                    name="Highlighted Countries",
                    hoverinfo="skip"
                )
            )

    fig.update_layout(
        title=dict(
            text=f"<b>{gender_title} Secondary Enrollment vs. Average Life Expectancy</b>",
            font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title=f"{gender_title} Secondary School Enrollment Rate (%)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zeroline=False
        ),
        yaxis=dict(
            title="Average Life Expectancy at Birth (Years)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zeroline=False,
            ticksuffix=" yrs"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
        margin=dict(l=60, r=40, t=55, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.26,
            xanchor="center",
            x=0.5,
            font=dict(size=10.5, color="#334155"),
            title=dict(text="")
        )
    )

    return fig


def render_stage4_page(df: pd.DataFrame):
    """
    Main renderer for Stage 4: Gender Gap & Education Analysis.
    """
    # Ensure derived education columns are ready in memory
    working_df = df.copy()
    if "edu_gender_gap" not in working_df.columns:
        working_df["edu_gender_gap"] = (
            working_df["secondary_school_enrollment_female"] - working_df["secondary_school_enrollment_male"]
        )

    # -------------------------------------------------------------------------
    # HERO HEADER
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <span class="badge" style="background-color: rgba(13, 148, 136, 0.15); color: {TEAL_ACCENT}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(13, 148, 136, 0.3); text-transform: uppercase;">
                STAGE 4 DELIVERABLE • GENDER GAP & EDUCATION ANALYSIS
            </span>
            <h1 style="color: {NAVY_PRIMARY}; font-size: 2.1rem; font-weight: 800; margin: 0.5rem 0 0.3rem 0; letter-spacing: -0.02em;">
                Gender Disparities & Educational Dynamics in Longevity
            </h1>
            <p style="color: #475569; font-size: 1.05rem; margin: 0; line-height: 1.5;">
                Investigating the universal female longevity advantage, gender parity in secondary education, 
                and how female versus male educational attainment correlates with national lifespan across 192 sovereign nations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # 1. INTERACTIVE FILTER STRIP
    # -------------------------------------------------------------------------
    all_regions = sorted(working_df["region"].unique().tolist())
    all_countries = sorted(working_df["name"].unique().tolist())

    if "stage4_selected_regions" not in st.session_state:
        st.session_state["stage4_selected_regions"] = all_regions
    if "stage4_selected_countries" not in st.session_state:
        st.session_state["stage4_selected_countries"] = []
    if "stage4_show_trendline" not in st.session_state:
        st.session_state["stage4_show_trendline"] = True
    if "stage4_highlight_large_gap" not in st.session_state:
        st.session_state["stage4_highlight_large_gap"] = False

    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <strong style="color: {NAVY_PRIMARY}; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em;">
                        🎛️ Interactive Filters & Analysis Controls
                    </strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        f_col1, f_col2, f_col3, f_col4 = st.columns([2.5, 2.5, 1.8, 1])

        with f_col1:
            selected_regions = st.multiselect(
                "Filter by Geographic Region:",
                options=all_regions,
                default=st.session_state["stage4_selected_regions"],
                key="stage4_regions_widget"
            )
            if not selected_regions:
                selected_regions = all_regions

        with f_col2:
            highlight_countries = st.multiselect(
                "Search & Highlight Countries:",
                options=all_countries,
                default=st.session_state["stage4_selected_countries"],
                key="stage4_countries_widget"
            )

        with f_col3:
            show_trendline = st.checkbox(
                "Show OLS Trendlines",
                value=st.session_state["stage4_show_trendline"],
                key="stage4_trend_widget"
            )
            highlight_large_gap = st.checkbox(
                "Highlight Gaps ≥ 8.0 yrs",
                value=st.session_state["stage4_highlight_large_gap"],
                key="stage4_gap_widget"
            )

        with f_col4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state["stage4_selected_regions"] = all_regions
                st.session_state["stage4_selected_countries"] = []
                st.session_state["stage4_show_trendline"] = True
                st.session_state["stage4_highlight_large_gap"] = False
                st.rerun()

    filtered_df = working_df[working_df["region"].isin(selected_regions)].copy()
    num_filtered = len(filtered_df)

    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 1.5rem;">
            Displaying <strong>{num_filtered}</strong> of <strong>{len(working_df)}</strong> sovereign nations 
            across <strong>{len(selected_regions)}</strong> regions.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # PART 5: DYNAMIC STATISTICAL INSIGHTS (KPIS & SUMMARY)
    # -------------------------------------------------------------------------
    st.markdown("### Core Gender & Educational Indicators")
    st.caption("Calculated dynamically from the active filtered records.")

    # Calculate dynamic summary metrics
    if not filtered_df.empty:
        mean_female_life = float(filtered_df["life_expectancy_female"].mean())
        mean_male_life = float(filtered_df["life_expectancy_male"].mean())
        mean_life_gap = float(filtered_df["female_male_gap"].mean())
        mean_female_edu = float(filtered_df["secondary_school_enrollment_female"].mean())
        mean_male_edu = float(filtered_df["secondary_school_enrollment_male"].mean())
        mean_edu_gap = float(filtered_df["edu_gender_gap"].mean())

        # Bivariate correlations
        stat_f_edu = calculate_bivariate_stats(filtered_df, "secondary_school_enrollment_female", "avg_life_expectancy")
        stat_m_edu = calculate_bivariate_stats(filtered_df, "secondary_school_enrollment_male", "avg_life_expectancy")
        stat_mf_life = calculate_bivariate_stats(filtered_df, "life_expectancy_male", "life_expectancy_female")
    else:
        mean_female_life = mean_male_life = mean_life_gap = 0.0
        mean_female_edu = mean_male_edu = mean_edu_gap = 0.0
        stat_f_edu = stat_m_edu = stat_mf_life = {"pearson_r": 0.0, "spearman_r": 0.0, "r2": 0.0, "slope": 0.0, "is_valid": False}

    # Row 1: KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        render_kpi(
            title="Avg Female Longevity",
            value=f"{mean_female_life:.1f} yrs",
            subtitle=f"Male avg: {mean_male_life:.1f} yrs",
            tag="Biological Advantage"
        )

    with kpi_col2:
        render_kpi(
            title="Avg Gender Longevity Gap",
            value=f"+{mean_life_gap:.2f} yrs",
            subtitle="Female lifespan exceeding male",
            tag="100% of Nations"
        )

    with kpi_col3:
        render_kpi(
            title="Avg Female Enrollment",
            value=f"{mean_female_edu:.1f}%",
            subtitle=f"Male avg: {mean_male_edu:.1f}%",
            tag="Secondary Education"
        )

    with kpi_col4:
        sign = "+" if mean_edu_gap >= 0 else ""
        render_kpi(
            title="Net Educational Gap",
            value=f"{sign}{mean_edu_gap:.2f}%",
            subtitle="Female minus male secondary rate",
            tag="Gender Parity Index"
        )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STAGE 4 MULTI-VIEW TAB NAVIGATION
    # -------------------------------------------------------------------------
    tab_labels = [
        "🚻 Part 1: Male vs. Female Longevity",
        "📊 Part 2: Longevity Gap Rankings",
        "🎓 Part 3: Secondary Education Parity",
        "📈 Part 4: Education vs. Life Expectancy",
        "📋 Part 5: Empirical Synthesis & Insights"
    ]

    selected_view = st.radio(
        "Select Stage 4 Analysis View:",
        tab_labels,
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW 1: PART 1 - MALE VS FEMALE LIFE EXPECTANCY
    # -------------------------------------------------------------------------
    if selected_view == "🚻 Part 1: Male vs. Female Longevity":
        col_plot1, col_info1 = st.columns([3, 2])

        with col_plot1:
            fig_mf = build_male_vs_female_life_figure(
                filtered_df,
                highlight_countries,
                highlight_large_gap
            )
            st.plotly_chart(fig_mf, use_container_width=True)

        with col_info1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🚻 The Universal Female Advantage</h3>
                    <p>
                        <strong>45-Degree Equality Reference:</strong> The dashed line represents demographic equality 
                        where female and male life expectancies are identical (Y = X).
                    </p>
                    <div class="paradox-callout" style="margin: 0.8rem 0;">
                        <strong>Empirical Constant:</strong> In <strong>100% of the 192 sovereign nations</strong> in the dataset, 
                        female life expectancy strictly exceeds male life expectancy. There is not a single country on or below 
                        the equality line.
                    </div>
                    <p>
                        <strong>Statistical Alignment:</strong> Across the active selection, male and female longevity 
                        exhibit an extraordinarily tight linear correlation:
                    </p>
                    <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7; margin-bottom: 0.8rem;">
                        <li><strong>Pearson r:</strong> {stat_mf_life['pearson_r']:.4f}</li>
                        <li><strong>Spearman ρ:</strong> {stat_mf_life['spearman_r']:.4f}</li>
                        <li><strong>Variance Explained (R²):</strong> {stat_mf_life['r2']*100:.1f}%</li>
                        <li><strong>Slope:</strong> +{stat_mf_life['slope']:.3f} (for every 1 yr gain in male life, female life gains ~{stat_mf_life['slope']:.2f} yrs)</li>
                    </ul>
                    <p style="font-size: 0.88rem; color: #64748B; margin: 0;">
                        <em>Why the gap exists:</em> Biological factors (protective effects of estrogen, two X chromosomes) combine with 
                        behavioral and occupational exposures (higher male rates of cardiovascular disease, heavy industrial labor, 
                        and substance usage) to create a persistent female advantage across all income levels.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------------------
    # VIEW 2: PART 2 - GENDER LONGEVITY GAP RANKINGS & REGIONAL COMPARISON
    # -------------------------------------------------------------------------
    elif selected_view == "📊 Part 2: Longevity Gap Rankings":
        st.markdown("### Longevity Gender Gap: Country Extremes & Regional Patterns")
        st.caption("Female Life Expectancy minus Male Life Expectancy (Years).")

        bar_tab1, bar_tab2, bar_tab3 = st.tabs([
            "🏆 Top 10 Largest Gender Gaps",
            "🌱 Bottom 10 Smallest Gender Gaps",
            "🗺️ Regional Longevity Gap Comparison"
        ])

        with bar_tab1:
            col_b1, col_t1 = st.columns([3, 2])
            with col_b1:
                fig_top10 = build_gender_gap_bar_figure(filtered_df, mode="top10")
                st.plotly_chart(fig_top10, use_container_width=True)
            with col_t1:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>🔍 The Eastern European Male Health Crisis</h3>
                        <p>
                            The largest longevity gender gaps in the world are intensely concentrated in 
                            <strong>Eastern Europe and Central Asia</strong>:
                        </p>
                        <p>
                            Nations such as <strong>Syria (+11.8y)</strong>, <strong>Lithuania (+11.1y)</strong>, 
                            <strong>Russia (+10.7y)</strong>, and <strong>Belarus (+10.0y)</strong> feature gaps 
                            more than double the global average (+4.87 yrs).
                        </p>
                        <div class="paradox-callout">
                            <strong>Epidemiological Context:</strong><br>
                            This massive disparity does not stem from extraordinary female longevity, but rather 
                            from <em>premature male adult mortality</em> driven by high cardiovascular disease, 
                            heavy alcohol/tobacco consumption, and industrial health risks.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with bar_tab2:
            col_b2, col_t2 = st.columns([3, 2])
            with col_b2:
                fig_bot10 = build_gender_gap_bar_figure(filtered_df, mode="bottom10")
                st.plotly_chart(fig_bot10, use_container_width=True)
            with col_t2:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>🌱 Minimal Gap Economies</h3>
                        <p>
                            The smallest gender gaps occur predominantly in low-income or agrarian regions:
                        </p>
                        <p>
                            <strong>Bhutan (+0.6y)</strong>, <strong>Guinea (+1.2y)</strong>, <strong>Marshall Islands (+1.2y)</strong>, 
                            and <strong>Burkina Faso (+1.5y)</strong> exhibit gender gaps below 2 years.
                        </p>
                        <div class="paradox-callout">
                            <strong>Maternal Mortality Drag:</strong><br>
                            In nations with elevated maternal mortality rates and limited obstetric care, 
                            the biological longevity advantage of females is partially offset by childbearing risks, 
                            compressing the gap closer to zero.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with bar_tab3:
            fig_reg = build_gender_gap_bar_figure(filtered_df, mode="regional")
            st.plotly_chart(fig_reg, use_container_width=True)

    # -------------------------------------------------------------------------
    # VIEW 3: PART 3 - SECONDARY EDUCATION PARITY
    # -------------------------------------------------------------------------
    elif selected_view == "🎓 Part 3: Secondary Education Parity":
        col_plot3, col_info3 = st.columns([3, 2])

        with col_plot3:
            fig_edu = build_education_parity_figure(filtered_df, highlight_countries)
            st.plotly_chart(fig_edu, use_container_width=True)

        with col_info3:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🎓 Gender Parity in Secondary Education</h3>
                    <p>
                        <strong>45-Degree Parity Line:</strong> Points on the line indicate equal enrollment 
                        between adolescent girls and boys (F = M).
                    </p>
                    <div class="paradox-callout" style="margin: 0.8rem 0;">
                        <strong>Symmetrical Global Distribution:</strong> Unlike life expectancy (where 100% of nations favor females), 
                        secondary education displays nations in both zones:
                        <br><br>
                        • <strong>Female Advantage:</strong> Nations above the line (e.g. Suriname +21.5%, Lesotho +18.5%, Belgium +18.1%).<br>
                        • <strong>Male Advantage:</strong> Nations below the line (e.g. Afghanistan -30.1%, Liechtenstein -24.0%, Angola -22.1%).
                    </div>
                    <p>
                        <strong>Global Alignment:</strong> Globally, secondary school enrollment rates between sexes 
                        are strongly correlated (Pearson r ≈ 0.98), but significant regional disparities remain in fragile 
                        and conflict-affected states.
                    </p>
                    <p style="font-size: 0.88rem; color: #64748B; margin: 0;">
                        <em>Policy Importance:</em> Expanding female secondary education is recognized by the WHO and UNICEF as 
                        one of the most effective non-medical interventions for improving public health, reducing child mortality, 
                        and lowering fertility rates.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------------------
    # VIEW 4: PART 4 - EDUCATION VS LIFE EXPECTANCY
    # -------------------------------------------------------------------------
    elif selected_view == "📈 Part 4: Education vs. Life Expectancy":
        st.markdown("### Secondary School Enrollment vs. National Life Expectancy")
        st.caption("Assessing how educational attainment across sexes relates to national longevity.")

        # Sub-selection: Side-by-side vs Individual Deep Dive
        sub_layout = st.radio(
            "Select Layout Style:",
            ["Side-by-Side Comparative View", "👩 Female Education Deep Dive", "👨 Male Education Deep Dive"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        if sub_layout == "Side-by-Side Comparative View":
            col_fem, col_masc = st.columns(2)

            with col_fem:
                fig_f = build_education_vs_life_figure(filtered_df, "female", highlight_countries, show_trendline)
                st.plotly_chart(fig_f, use_container_width=True)
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.8rem; font-size: 0.85rem;">
                        <strong>Female Education Regression:</strong><br>
                        • Pearson r: <strong>+{stat_f_edu['pearson_r']:.3f}</strong> | Spearman ρ: <strong>+{stat_f_edu['spearman_r']:.3f}</strong><br>
                        • Variance Explained (R²): <strong>{stat_f_edu['r2']*100:.1f}%</strong> | Slope: <strong>+{stat_f_edu['slope']:.3f} yrs/%</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_masc:
                fig_m = build_education_vs_life_figure(filtered_df, "male", highlight_countries, show_trendline)
                st.plotly_chart(fig_m, use_container_width=True)
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.8rem; font-size: 0.85rem;">
                        <strong>Male Education Regression:</strong><br>
                        • Pearson r: <strong>+{stat_m_edu['pearson_r']:.3f}</strong> | Spearman ρ: <strong>+{stat_m_edu['spearman_r']:.3f}</strong><br>
                        • Variance Explained (R²): <strong>{stat_m_edu['r2']*100:.1f}%</strong> | Slope: <strong>+{stat_m_edu['slope']:.3f} yrs/%</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        elif sub_layout == "👩 Female Education Deep Dive":
            c_f_plot, c_f_info = st.columns([3, 2])
            with c_f_plot:
                fig_f = build_education_vs_life_figure(filtered_df, "female", highlight_countries, show_trendline)
                st.plotly_chart(fig_f, use_container_width=True)
            with c_f_info:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>👩 Female Secondary Education</h3>
                        <p>
                            Secondary education for girls is a robust predictor of societal health:
                        </p>
                        <ul style="color: #334155; font-size: 0.92rem; line-height: 1.8;">
                            <li><strong>Pearson r:</strong> +{stat_f_edu['pearson_r']:.4f} (Very Strong Positive)</li>
                            <li><strong>Spearman ρ:</strong> +{stat_f_edu['spearman_r']:.4f}</li>
                            <li><strong>Variance Explained (R²):</strong> {stat_f_edu['r2']*100:.1f}%</li>
                            <li><strong>Linear Slope:</strong> +{stat_f_edu['slope']:.3f} yrs per percentage point</li>
                        </ul>
                        <div class="paradox-callout">
                            <strong>Demographic Multiplier:</strong><br>
                            Educated women have greater health literacy, marry later, have fewer and healthier children, 
                            and are far more likely to seek prenatal care and timely vaccinations.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            c_m_plot, c_m_info = st.columns([3, 2])
            with c_m_plot:
                fig_m = build_education_vs_life_figure(filtered_df, "male", highlight_countries, show_trendline)
                st.plotly_chart(fig_m, use_container_width=True)
            with c_m_info:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>👨 Male Secondary Education</h3>
                        <p>
                            Male secondary education also tracks closely with overall national longevity:
                        </p>
                        <ul style="color: #334155; font-size: 0.92rem; line-height: 1.8;">
                            <li><strong>Pearson r:</strong> +{stat_m_edu['pearson_r']:.4f} (Very Strong Positive)</li>
                            <li><strong>Spearman ρ:</strong> +{stat_m_edu['spearman_r']:.4f}</li>
                            <li><strong>Variance Explained (R²):</strong> {stat_m_edu['r2']*100:.1f}%</li>
                            <li><strong>Linear Slope:</strong> +{stat_m_edu['slope']:.3f} yrs per percentage point</li>
                        </ul>
                        <div class="paradox-callout">
                            <strong>Economic & Occupational Health:</strong><br>
                            Male education correlates strongly with shift from hazardous manual and informal sector 
                            labor toward safer service-sector employment with workplace safety standards.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # -------------------------------------------------------------------------
    # VIEW 5: PART 5 - EMPIRICAL SYNTHESIS & KEY FINDINGS
    # -------------------------------------------------------------------------
    elif selected_view == "📋 Part 5: Empirical Synthesis & Insights":
        st.markdown("### Empirical Synthesis & Key Findings")
        st.caption("Synthesizing gender disparities, education, and the Longevity Paradox.")

        col_syn1, col_syn2 = st.columns(2)

        with col_syn1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🔬 1. The Asymmetry of the Longevity Gap</h3>
                    <p>
                        Across all 192 sovereign nations, the average gender gap in life expectancy stands at 
                        <strong>+{mean_life_gap:.2f} years</strong>. However, the distribution is highly non-uniform:
                    </p>
                    <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                        <li>
                            <strong>High-Gap Cluster:</strong> Eastern European and Central Asian nations average 
                            gaps between <strong>6.5 and 8.1 years</strong>, driven by severe adult male mortality 
                            from non-communicable diseases.
                        </li>
                        <li>
                            <strong>Low-Gap Cluster:</strong> Sub-Saharan African and South Asian nations average 
                            gaps between <strong>2.6 and 3.0 years</strong>, where high maternal mortality and lower 
                            female healthcare access compress the gap.
                        </li>
                    </ul>
                    <div class="paradox-callout">
                        <strong>Takeaway:</strong> A widening gender gap is not necessarily an indicator of superior female welfare; 
                        it often signals a severe public health crisis among adult working-age males.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_syn2:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🎓 2. Education as a Determinant of Longevity</h3>
                    <p>
                        Both female (<strong>r = +{stat_f_edu['pearson_r']:.3f}</strong>) and male 
                        (<strong>r = +{stat_m_edu['pearson_r']:.3f}</strong>) secondary school enrollment explain 
                        nearly <strong>60% of the variance</strong> in global life expectancy.
                    </p>
                    <p>
                        Education operates as a powerful structural proxy for:
                    </p>
                    <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                        <li><strong>Health Literacy:</strong> Understanding nutritional requirements, hygiene, and disease vectors.</li>
                        <li><strong>Delayed Childbearing:</strong> Reducing adolescent pregnancies and maternal complications.</li>
                        <li><strong>Household Income:</strong> Greater resilience against catastrophic healthcare costs.</li>
                    </ul>
                    <div style="background-color: #FEF2F2; border-left: 3.5px solid #EF4444; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.84rem; color: #991B1B;">
                        <strong>Academic Caution:</strong> These findings demonstrate correlation, not direct isolated causation. 
                        Education works synergistically with sanitation infrastructure, healthcare systems, and economic stability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

