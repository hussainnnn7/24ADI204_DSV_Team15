"""
Stage 5: Regional Disparities Analysis
======================================
Module containing:
- Interactive Filter Strip (Region Multi-select, Country Search, Sorting Controls, Reset)
- Part 1: Regional Life Expectancy (Grouped Male vs. Female Bars & Longevity Gap Analysis)
- Part 2: Regional Wealth Comparison (GDP per Capita Box Plots, Mean vs. Median Divergence)
- Part 3: Health & Demographic Disparities (Infant Mortality, Fertility, Homicide, Urbanization)
- Part 4: Education & Digital Divide (Female & Male Secondary Schooling, Internet Penetration)
- Part 5: Regional Rankings & Statistical Insights (Interactive Indicator Selector & Summary Table)
- Methodological Integrity & Non-Causal Analytical Synthesis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

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

# Metadata dictionary for all selectable regional indicators
INDICATOR_META: Dict[str, Dict[str, Any]] = {
    "avg_life_expectancy": {
        "title": "Average Life Expectancy at Birth",
        "short_name": "Life Expectancy",
        "unit": "yrs",
        "fmt": ".1f",
        "desc": "Combined arithmetic mean of female and male life expectancy at birth."
    },
    "female_male_gap": {
        "title": "Female Longevity Advantage (F - M Gap)",
        "short_name": "Longevity Gap",
        "unit": "yrs",
        "fmt": ".2f",
        "desc": "Female life expectancy minus male life expectancy."
    },
    "gdp_per_capita": {
        "title": "GDP per Capita (Nominal US$)",
        "short_name": "GDP per Capita",
        "unit": "$",
        "fmt": ",.0f",
        "desc": "Gross domestic product divided by midyear national population."
    },
    "infant_mortality": {
        "title": "Infant Mortality Rate (per 1,000 live births)",
        "short_name": "Infant Mortality",
        "unit": "per 1,000",
        "fmt": ".1f",
        "desc": "Number of infant deaths before age one per 1,000 live births."
    },
    "fertility": {
        "title": "Total Fertility Rate (Births per woman)",
        "short_name": "Fertility Rate",
        "unit": "births/woman",
        "fmt": ".2f",
        "desc": "Average number of children born to a woman across reproductive years."
    },
    "urban_population": {
        "title": "Urban Population Share (% of total)",
        "short_name": "Urbanization",
        "unit": "%",
        "fmt": ".1f",
        "desc": "Percentage of total national population residing in urban agglomerations."
    },
    "homicide_rate": {
        "title": "Intentional Homicide Rate (per 100k)",
        "short_name": "Homicide Rate",
        "unit": "per 100k",
        "fmt": ".1f",
        "desc": "Unlawful intentional homicides per 100,000 population."
    },
    "internet_users": {
        "title": "Internet Usage Penetration (% of population)",
        "short_name": "Internet Users",
        "unit": "%",
        "fmt": ".1f",
        "desc": "Percentage of individuals who have used the internet in the past 3 months."
    },
    "secondary_school_enrollment_female": {
        "title": "Female Secondary School Enrollment Rate (%)",
        "short_name": "Female Secondary Edu",
        "unit": "%",
        "fmt": ".1f",
        "desc": "Gross enrollment ratio for females in secondary education institutions."
    },
    "secondary_school_enrollment_male": {
        "title": "Male Secondary School Enrollment Rate (%)",
        "short_name": "Male Secondary Edu",
        "unit": "%",
        "fmt": ".1f",
        "desc": "Gross enrollment ratio for males in secondary education institutions."
    }
}


def build_regional_life_expectancy_figure(
    df: pd.DataFrame,
    sort_order: str = "desc"
) -> go.Figure:
    """
    Part 1: Grouped bar chart comparing Male and Female Life Expectancy by Region,
    with hover tooltips displaying both sexes and the longevity gap.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for current selection.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # Aggregate regional averages
    reg_df = df.groupby("region").agg({
        "life_expectancy_male": "mean",
        "life_expectancy_female": "mean",
        "avg_life_expectancy": "mean",
        "female_male_gap": "mean",
        "name": "count"
    }).reset_index()

    # Sort
    ascending = (sort_order == "asc")
    reg_df = reg_df.sort_values("avg_life_expectancy", ascending=ascending)

    fig = go.Figure()

    # Male Bar Trace
    fig.add_trace(
        go.Bar(
            name="Male Life Expectancy",
            x=reg_df["region"],
            y=reg_df["life_expectancy_male"],
            marker_color=NAVY_PRIMARY,
            text=[f"{v:.1f}y" for v in reg_df["life_expectancy_male"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="white"),
            customdata=np.stack([
                reg_df["name"],
                reg_df["avg_life_expectancy"],
                reg_df["female_male_gap"]
            ], axis=-1),
            hovertemplate=(
                "<b>%{x}</b> (%{customdata[0]} nations)<br>"
                "• Male Life Expectancy: <b>%{y:.1f} yrs</b><br>"
                "• Combined Average: <b>%{customdata[1]:.1f} yrs</b><br>"
                "• Gender Gap (F - M): <b>+%{customdata[2]:.2f} yrs</b><br>"
                "<extra></extra>"
            )
        )
    )

    # Female Bar Trace
    fig.add_trace(
        go.Bar(
            name="Female Life Expectancy",
            x=reg_df["region"],
            y=reg_df["life_expectancy_female"],
            marker_color=TEAL_ACCENT,
            text=[f"{v:.1f}y" for v in reg_df["life_expectancy_female"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="white"),
            customdata=np.stack([
                reg_df["name"],
                reg_df["avg_life_expectancy"],
                reg_df["female_male_gap"]
            ], axis=-1),
            hovertemplate=(
                "<b>%{x}</b> (%{customdata[0]} nations)<br>"
                "• Female Life Expectancy: <b>%{y:.1f} yrs</b><br>"
                "• Combined Average: <b>%{customdata[1]:.1f} yrs</b><br>"
                "• Gender Gap (F - M): <b>+%{customdata[2]:.2f} yrs</b><br>"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Regional Life Expectancy by Sex (Male vs. Female)</b>",
            font=dict(size=15, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title="",
            tickangle=-40,
            tickfont=dict(size=10.5, color="#334155"),
            showgrid=False
        ),
        yaxis=dict(
            title="Life Expectancy at Birth (Years)",
            range=[40, 90],
            showgrid=True,
            gridcolor="#F1F5F9",
            ticksuffix=" yrs"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
        margin=dict(l=60, r=30, t=55, b=110),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.38,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#334155")
        )
    )

    return fig


def build_regional_wealth_figure(
    df: pd.DataFrame,
    chart_type: str = "box",
    highlight_countries: List[str] = None
) -> go.Figure:
    """
    Part 2: Visualizing Regional Wealth Comparison via Box Plot (showing country distributions
    and extreme outliers) or Bar Chart (comparing Mean vs. Median divergence).
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for current selection.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # Compute regional medians for sorting order
    reg_medians = df.groupby("region")["gdp_per_capita"].median().sort_values(ascending=False)
    sorted_regions = reg_medians.index.tolist()

    if chart_type == "box":
        # Interactive Box Plot of GDP per capita distribution per region
        fig = px.box(
            df,
            x="region",
            y="gdp_per_capita",
            category_orders={"region": sorted_regions},
            color_discrete_sequence=[NAVY_PRIMARY],
            points="all",
            hover_name="name",
            custom_data=["avg_life_expectancy", "population"]
        )

        fig.update_traces(
            marker=dict(size=5, opacity=0.7, color=TEAL_ACCENT),
            line=dict(color=NAVY_PRIMARY, width=1.5),
            hovertemplate=(
                "<b>%{hovertext}</b> (%{x})<br>"
                "• GDP per Capita: <b>$%{y:,.0f}</b><br>"
                "• Life Expectancy: <b>%{customdata[0]:.1f} yrs</b><br>"
                "• Population: <b>%{customdata[1]:,.0f}k</b><br>"
                "<extra></extra>"
            )
        )

        # Highlight user-searched countries if present
        if highlight_countries:
            hl_df = df[df["name"].isin(highlight_countries)]
            if not hl_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=hl_df["region"],
                        y=hl_df["gdp_per_capita"],
                        mode="markers+text",
                        text=hl_df["name"],
                        textposition="top center",
                        textfont=dict(size=11, color="#DC2626", family="Plus Jakarta Sans"),
                        marker=dict(
                            symbol="circle-open",
                            size=24,
                            color="#DC2626",
                            line=dict(width=2.5, color="#DC2626")
                        ),
                        name="Highlighted Countries",
                        hoverinfo="skip"
                    )
                )

        fig.update_layout(
            title=dict(
                text="<b>Regional GDP per Capita Distribution (Box Plots with Country Data Points)</b>",
                font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                x=0.01,
                y=0.98
            ),
            xaxis=dict(
                title="",
                tickangle=-40,
                tickfont=dict(size=10.5, color="#334155")
            ),
            yaxis=dict(
                title="GDP per Capita (Nominal US$)",
                showgrid=True,
                gridcolor="#F1F5F9",
                tickprefix="$"
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=520,
            margin=dict(l=70, r=30, t=55, b=110),
            showlegend=False
        )

    else:
        # Mean vs Median Bar Chart comparison
        reg_df = df.groupby("region")["gdp_per_capita"].agg(["mean", "median", "count"]).reset_index()
        reg_df = reg_df.sort_values("median", ascending=False)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Average (Mean) GDP",
                x=reg_df["region"],
                y=reg_df["mean"],
                marker_color=NAVY_PRIMARY,
                text=[f"${v:,.0f}" for v in reg_df["mean"]],
                textposition="outside",
                textfont=dict(size=9.5),
                hovertemplate="<b>%{x}</b><br>• Mean GDP: <b>$%{y:,.0f}</b><extra></extra>"
            )
        )
        fig.add_trace(
            go.Bar(
                name="Median GDP (50th percentile)",
                x=reg_df["region"],
                y=reg_df["median"],
                marker_color=TEAL_ACCENT,
                text=[f"${v:,.0f}" for v in reg_df["median"]],
                textposition="outside",
                textfont=dict(size=9.5),
                hovertemplate="<b>%{x}</b><br>• Median GDP: <b>$%{y:,.0f}</b><extra></extra>"
            )
        )

        fig.update_layout(
            barmode="group",
            title=dict(
                text="<b>Regional Wealth Comparison: Mean vs. Median GDP per Capita (US$)</b>",
                font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                x=0.01,
                y=0.98
            ),
            xaxis=dict(
                title="",
                tickangle=-40,
                tickfont=dict(size=10.5, color="#334155")
            ),
            yaxis=dict(
                title="GDP per Capita (Nominal US$)",
                showgrid=True,
                gridcolor="#F1F5F9",
                tickprefix="$",
                range=[0, reg_df["mean"].max() * 1.22]
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=520,
            margin=dict(l=70, r=30, t=55, b=110),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.38,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#334155")
            )
        )

    return fig


def build_health_demographics_figure(
    df: pd.DataFrame,
    indicator: str = "infant_mortality",
    sort_order: str = "desc"
) -> go.Figure:
    """
    Part 3: Comparative bar chart of Health and Demographic Disparities across regions
    (Infant Mortality, Fertility, Homicide, Urban Population).
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for current selection.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    meta = INDICATOR_META[indicator]
    ascending = (sort_order == "asc")

    reg_df = df.groupby("region").agg({
        indicator: ["mean", "median", "min", "max", "count"]
    })
    reg_df.columns = ["mean", "median", "min", "max", "count"]
    reg_df = reg_df.reset_index().sort_values("mean", ascending=ascending)

    val_fmt = meta["fmt"]
    unit = meta["unit"]

    fig = go.Figure(
        go.Bar(
            x=reg_df["region"],
            y=reg_df["mean"],
            marker=dict(
                color=reg_df["mean"],
                colorscale=[[0, "#99F6E4"], [0.5, TEAL_ACCENT], [1, NAVY_PRIMARY]],
                line=dict(color="#CBD5E1", width=0.5)
            ),
            text=[f"{v:{val_fmt}} {unit}" for v in reg_df["mean"]],
            textposition="outside",
            textfont=dict(size=9.5),
            customdata=np.stack([
                reg_df["count"],
                reg_df["median"],
                reg_df["min"],
                reg_df["max"]
            ], axis=-1),
            hovertemplate=(
                "<b>%{x}</b> (%{customdata[0]} nations)<br>"
                f"• Mean {meta['short_name']}: <b>%{{y:{val_fmt}}} {unit}</b><br>"
                f"• Median: <b>%{{customdata[1]:{val_fmt}}} {unit}</b><br>"
                f"• Range: %{{customdata[2]:{val_fmt}}} to %{{customdata[3]:{val_fmt}}} {unit}<br>"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Regional Comparison: {meta['title']}</b>",
            font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title="",
            tickangle=-40,
            tickfont=dict(size=10.5, color="#334155")
        ),
        yaxis=dict(
            title=f"{meta['short_name']} ({unit})",
            showgrid=True,
            gridcolor="#F1F5F9",
            range=[0, max(reg_df["mean"].max() * 1.25, 1.0)]
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
        margin=dict(l=60, r=30, t=55, b=110)
    )

    return fig


def build_education_digital_divide_figure(
    df: pd.DataFrame,
    metric_type: str = "education"
) -> go.Figure:
    """
    Part 4: Education and Digital Divide comparison across regions.
    - metric_type="education": Grouped bar chart of Female vs Male Secondary School Enrollment
    - metric_type="internet": Bar chart of Internet Users (% of population)
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for current selection.", paper_bgcolor="white", plot_bgcolor="white")
        return fig

    if metric_type == "education":
        reg_df = df.groupby("region").agg({
            "secondary_school_enrollment_female": "mean",
            "secondary_school_enrollment_male": "mean",
            "name": "count"
        }).reset_index()
        reg_df["gap"] = reg_df["secondary_school_enrollment_female"] - reg_df["secondary_school_enrollment_male"]
        reg_df = reg_df.sort_values("secondary_school_enrollment_female", ascending=False)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Female Secondary Enrollment",
                x=reg_df["region"],
                y=reg_df["secondary_school_enrollment_female"],
                marker_color=TEAL_ACCENT,
                text=[f"{v:.1f}%" for v in reg_df["secondary_school_enrollment_female"]],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=9.5, color="white"),
                customdata=np.stack([reg_df["name"], reg_df["gap"]], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b> (%{customdata[0]} nations)<br>"
                    "• Female Enrollment: <b>%{y:.1f}%</b><br>"
                    "• Gender Gap (F - M): <b>%{customdata[1]:+.1f}%</b><br>"
                    "<extra></extra>"
                )
            )
        )
        fig.add_trace(
            go.Bar(
                name="Male Secondary Enrollment",
                x=reg_df["region"],
                y=reg_df["secondary_school_enrollment_male"],
                marker_color=NAVY_PRIMARY,
                text=[f"{v:.1f}%" for v in reg_df["secondary_school_enrollment_male"]],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=9.5, color="white"),
                customdata=np.stack([reg_df["name"], reg_df["gap"]], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b> (%{customdata[0]} nations)<br>"
                    "• Male Enrollment: <b>%{y:.1f}%</b><br>"
                    "• Gender Gap (F - M): <b>%{customdata[1]:+.1f}%</b><br>"
                    "<extra></extra>"
                )
            )
        )

        fig.update_layout(
            barmode="group",
            title=dict(
                text="<b>Regional Secondary School Enrollment Rate: Female vs. Male (%)</b>",
                font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                x=0.01,
                y=0.98
            ),
            xaxis=dict(
                title="",
                tickangle=-40,
                tickfont=dict(size=10.5, color="#334155")
            ),
            yaxis=dict(
                title="Secondary School Enrollment Rate (%)",
                showgrid=True,
                gridcolor="#F1F5F9",
                ticksuffix="%"
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=500,
            margin=dict(l=60, r=30, t=55, b=110),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.38,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#334155")
            )
        )

    else:  # Digital divide / Internet users
        reg_df = df.groupby("region").agg({
            "internet_users": ["mean", "median", "count"]
        })
        reg_df.columns = ["mean", "median", "count"]
        reg_df = reg_df.reset_index().sort_values("mean", ascending=False)

        fig = go.Figure(
            go.Bar(
                x=reg_df["region"],
                y=reg_df["mean"],
                marker=dict(
                    color=reg_df["mean"],
                    colorscale=[[0, "#E0F2FE"], [0.5, "#0284C7"], [1, NAVY_PRIMARY]],
                    line=dict(color="#CBD5E1", width=0.5)
                ),
                text=[f"{v:.1f}%" for v in reg_df["mean"]],
                textposition="outside",
                textfont=dict(size=9.5),
                customdata=np.stack([reg_df["count"], reg_df["median"]], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b> (%{customdata[0]} nations)<br>"
                    "• Mean Internet Penetration: <b>%{y:.1f}%</b><br>"
                    "• Median: <b>%{customdata[1]:.1f}%</b><br>"
                    "<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title=dict(
                text="<b>Regional Digital Divide: Internet Users Penetration Rate (% of population)</b>",
                font=dict(size=14.5, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                x=0.01,
                y=0.98
            ),
            xaxis=dict(
                title="",
                tickangle=-40,
                tickfont=dict(size=10.5, color="#334155")
            ),
            yaxis=dict(
                title="Internet Penetration (% of population)",
                showgrid=True,
                gridcolor="#F1F5F9",
                ticksuffix="%",
                range=[0, 115]
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=500,
            margin=dict(l=60, r=30, t=55, b=110)
        )

    return fig


def build_regional_ranking_table(
    df: pd.DataFrame,
    indicator: str,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Part 5: Computes comprehensive regional ranking table with summary statistics:
    Country Count, Mean, Median, Min, Max, Std Dev.
    """
    if df.empty:
        return pd.DataFrame()

    meta = INDICATOR_META[indicator]

    reg_stats = df.groupby("region")[indicator].agg(
        Count="count",
        Mean="mean",
        Median="median",
        Min="min",
        Max="max",
        Std="std"
    ).reset_index()

    reg_stats = reg_stats.sort_values("Mean", ascending=ascending).reset_index(drop=True)
    reg_stats.index = reg_stats.index + 1  # 1-based rank
    reg_stats.index.name = "Rank"

    # Fill NaN std for single-country regions
    reg_stats["Std"] = reg_stats["Std"].fillna(0.0)

    return reg_stats


def render_stage5_page(df: pd.DataFrame):
    """
    Main renderer for Stage 5: Regional Disparities Analysis.
    """
    # -------------------------------------------------------------------------
    # HERO HEADER
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <span class="badge" style="background-color: rgba(13, 148, 136, 0.15); color: {TEAL_ACCENT}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(13, 148, 136, 0.3); text-transform: uppercase;">
                STAGE 5 DELIVERABLE • REGIONAL DISPARITIES ANALYSIS
            </span>
            <h1 style="color: {NAVY_PRIMARY}; font-size: 2.1rem; font-weight: 800; margin: 0.5rem 0 0.3rem 0; letter-spacing: -0.02em;">
                Regional Disparities: Geographic Divides in Longevity & Welfare
            </h1>
            <p style="color: #475569; font-size: 1.05rem; margin: 0; line-height: 1.5;">
                Benchmarking longevity, economic wealth distributions, primary health infrastructure, 
                and digital-educational divides across 22 geographic regions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # 1. INTERACTIVE FILTER STRIP
    # -------------------------------------------------------------------------
    all_regions = sorted(df["region"].unique().tolist())
    all_countries = sorted(df["name"].unique().tolist())

    if "stage5_selected_regions" not in st.session_state:
        st.session_state["stage5_selected_regions"] = all_regions
    if "stage5_selected_countries" not in st.session_state:
        st.session_state["stage5_selected_countries"] = []
    if "stage5_sort_order" not in st.session_state:
        st.session_state["stage5_sort_order"] = "desc"

    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <strong style="color: {NAVY_PRIMARY}; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em;">
                        🎛️ Regional Filters & Visual Sorting
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
                default=st.session_state["stage5_selected_regions"],
                key="stage5_regions_widget"
            )
            if not selected_regions:
                selected_regions = all_regions

        with f_col2:
            highlight_countries = st.multiselect(
                "Search & Highlight Countries:",
                options=all_countries,
                default=st.session_state["stage5_selected_countries"],
                key="stage5_countries_widget"
            )

        with f_col3:
            sort_order = st.selectbox(
                "Chart Sorting Order:",
                options=["desc", "asc"],
                format_func=lambda x: "Highest to Lowest (Descending)" if x == "desc" else "Lowest to Highest (Ascending)",
                key="stage5_sort_widget"
            )

        with f_col4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state["stage5_selected_regions"] = all_regions
                st.session_state["stage5_selected_countries"] = []
                st.session_state["stage5_sort_order"] = "desc"
                st.rerun()

    filtered_df = df[df["region"].isin(selected_regions)].copy()
    num_filtered_countries = len(filtered_df)
    num_filtered_regions = len(selected_regions)

    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 1.5rem;">
            Displaying <strong>{num_filtered_countries}</strong> nations across 
            <strong>{num_filtered_regions}</strong> of 22 global regions.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # DYNAMIC REGIONAL SUMMARY KPIS
    # -------------------------------------------------------------------------
    if not filtered_df.empty:
        reg_means = filtered_df.groupby("region")["avg_life_expectancy"].mean()
        highest_reg = reg_means.idxmax()
        highest_val = reg_means.max()
        lowest_reg = reg_means.idxmin()
        lowest_val = reg_means.min()
        span_val = highest_val - lowest_val

        reg_gdp_means = filtered_df.groupby("region")["gdp_per_capita"].mean()
        wealthiest_reg = reg_gdp_means.idxmax()
        wealthiest_val = reg_gdp_means.max()
    else:
        highest_reg = lowest_reg = wealthiest_reg = "N/A"
        highest_val = lowest_val = span_val = wealthiest_val = 0.0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        render_kpi(
            title="Regions Analyzed",
            value=f"{num_filtered_regions} / 22",
            subtitle=f"{num_filtered_countries} sovereign nations",
            tag="Global Coverage"
        )

    with kpi_col2:
        render_kpi(
            title="Highest Longevity Region",
            value=f"{highest_val:.1f} yrs",
            subtitle=f"{highest_reg}",
            tag="Regional Leader"
        )

    with kpi_col3:
        render_kpi(
            title="Lowest Longevity Region",
            value=f"{lowest_val:.1f} yrs",
            subtitle=f"{lowest_reg}",
            tag="Development Priority"
        )

    with kpi_col4:
        render_kpi(
            title="Regional Longevity Span",
            value=f"{span_val:.1f} yrs",
            subtitle=f"Gap between max & min regions",
            tag="Geographic Divide"
        )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW SELECTOR (RADIO GROUP NAVIGATION)
    # -------------------------------------------------------------------------
    tab_labels = [
        "🗺️ Part 1: Regional Life Expectancy",
        "💰 Part 2: Regional Wealth & Skewness",
        "🩺 Part 3: Health & Demographic Disparities",
        "🎓 Part 4: Education & Digital Divide",
        "📊 Part 5: Indicator Rankings & Summary Table"
    ]

    selected_view = st.radio(
        "Select Regional Analysis View:",
        tab_labels,
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW 1: PART 1 - REGIONAL LIFE EXPECTANCY (SEX-DISAGGREGATED)
    # -------------------------------------------------------------------------
    if selected_view == "🗺️ Part 1: Regional Life Expectancy":
        col_plot1, col_info1 = st.columns([3, 2])

        with col_plot1:
            fig_life = build_regional_life_expectancy_figure(filtered_df, sort_order)
            st.plotly_chart(fig_life, use_container_width=True)

        with col_info1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🗺️ The Geographic Longevity Gradient</h3>
                    <p>
                        Regional averages reveal dramatic geographic stratifications in average human lifespan:
                    </p>
                    <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                        <li>
                            <strong>Advanced Health Infrastructure Tier (79–83 yrs):</strong> 
                            Oceania, Northern America, Western Europe, and Northern Europe cluster at the top, 
                            supported by comprehensive primary and acute emergency systems.
                        </li>
                        <li>
                            <strong>Vulnerable Health Infrastructure Tier (60–64 yrs):</strong> 
                            Middle Africa, Western Africa, and Southern Africa report life expectancies 
                            over <strong>18 to 22 years lower</strong>, driven primarily by elevated childhood 
                            mortality and communicable diseases.
                        </li>
                    </ul>
                    <div class="paradox-callout">
                        <strong>Longevity Gender Gap Disparity:</strong><br>
                        While females live longer than males in every single region, the size of this female 
                        advantage varies from <strong>+2.7 yrs in Western Africa</strong> to over 
                        <strong>+8.1 yrs in Eastern Europe</strong>, reflecting severe regional adult male mortality crises.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------------------
    # VIEW 2: PART 2 - REGIONAL WEALTH & SKEWNESS
    # -------------------------------------------------------------------------
    elif selected_view == "💰 Part 2: Regional Wealth & Skewness":
        sub_wealth_type = st.radio(
            "Select Wealth Visualization Type:",
            ["📦 Box Plot Distributions (Intra-regional Spread & Outliers)", "📊 Mean vs. Median Comparison (Skewness Analysis)"],
            index=0,
            horizontal=True
        )

        col_w_plot, col_w_info = st.columns([3, 2])

        with col_w_plot:
            chart_mode = "box" if "Box Plot" in sub_wealth_type else "bar"
            fig_wealth = build_regional_wealth_figure(filtered_df, chart_mode, highlight_countries)
            st.plotly_chart(fig_wealth, use_container_width=True)

        with col_w_info:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>💰 Why Regional Averages & Medians Diverge</h3>
                    <p>
                        In economic analysis, relying solely on the <strong>arithmetic mean (average)</strong> 
                        frequently creates a distorted picture of typical living standards due to extreme right-skewness:
                    </p>
                    <div class="paradox-callout">
                        <strong>High-Income Hub Effect:</strong><br>
                        When a region contains one or two ultra-wealthy economies (e.g., Singapore in South-Eastern Asia, 
                        Qatar and the UAE in Western Asia, or Seychelles in Eastern Africa), the regional average GDP per capita 
                        is pulled dramatically upward, while the median remains modest.
                    </div>
                    <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                        <li><strong>South-Eastern Asia:</strong> Mean ($11,818) is nearly <strong>3.8× higher</strong> than median ($3,103).</li>
                        <li><strong>Eastern Africa:</strong> Mean ($1,976) is over <strong>2.5× higher</strong> than median ($773).</li>
                        <li><strong>Western Asia:</strong> Mean ($19,309) exceeds median ($9,368) by more than $10,000.</li>
                    </ul>
                    <p style="font-size: 0.88rem; color: #64748B; margin: 0;">
                        <em>Empirical Takeaway:</em> The median represents the true standard of living for the middle nation, 
                        whereas the mean captures aggregate capital concentration.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------------------
    # VIEW 3: PART 3 - HEALTH & DEMOGRAPHIC DISPARITIES
    # -------------------------------------------------------------------------
    elif selected_view == "🩺 Part 3: Health & Demographic Disparities":
        health_metric = st.selectbox(
            "Select Health / Demographic Indicator:",
            options=["infant_mortality", "fertility", "homicide_rate", "urban_population"],
            format_func=lambda x: INDICATOR_META[x]["title"],
            key="stage5_health_metric_select"
        )

        col_h_plot, col_h_info = st.columns([3, 2])

        with col_h_plot:
            fig_health = build_health_demographics_figure(filtered_df, health_metric, sort_order)
            st.plotly_chart(fig_health, use_container_width=True)

        with col_h_info:
            meta = INDICATOR_META[health_metric]
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>🩺 {meta['short_name']} Regional Landscape</h3>
                    <p><strong>Indicator Definition:</strong> {meta['desc']}</p>
                    <div class="paradox-callout" style="margin: 0.8rem 0;">
                        <strong>Regional Disparity Insight:</strong><br>
                        Examining <em>{meta['short_name'].lower()}</em> illustrates that health inequalities 
                        are geographically structured rather than randomly dispersed.
                    </div>
                    <p style="font-size: 0.88rem; color: #334155;">
                        Regions with high infant mortality and fertility rates consistently correspond to the lowest life 
                        expectancy clusters, whereas high urbanization regions enjoy superior emergency triage infrastructure.
                    </p>
                    <div style="background-color: #FEF2F2; border-left: 3.5px solid #EF4444; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.84rem; color: #991B1B;">
                        <strong>Academic Rigor:</strong> Regional associations do not demonstrate direct causation. Geographic factors 
                        interact with institutional histories, sanitation access, and governance stability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------------------
    # VIEW 4: PART 4 - EDUCATION & DIGITAL DIVIDE
    # -------------------------------------------------------------------------
    elif selected_view == "🎓 Part 4: Education & Digital Divide":
        divide_choice = st.radio(
            "Select Comparative Dimension:",
            ["📚 Secondary School Enrollment (Female vs. Male)", "🌐 Digital Divide (Internet Penetration % of Population)"],
            index=0,
            horizontal=True
        )

        col_d_plot, col_d_info = st.columns([3, 2])

        with col_d_plot:
            metric_type = "education" if "Enrollment" in divide_choice else "internet"
            fig_div = build_education_digital_divide_figure(filtered_df, metric_type)
            st.plotly_chart(fig_div, use_container_width=True)

        with col_d_info:
            if "Enrollment" in divide_choice:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>📚 The Educational Access Divide</h3>
                        <p>
                            Secondary education is recognized globally as a primary social determinant of health:
                        </p>
                        <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                            <li>
                                <strong>Universal Secondary Tiers (>100% gross enrollment):</strong> 
                                Northern, Western, and Southern Europe, along with Northern America and Oceania, 
                                have achieved near-universal secondary participation for both sexes.
                            </li>
                            <li>
                                <strong>Severe Access Barriers (<40% enrollment):</strong> 
                                Central Africa and Western Africa report average secondary enrollment below 45%, 
                                with girls facing acute disadvantages in rural districts.
                            </li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>🌐 The Digital Connectivity Divide</h3>
                        <p>
                            Internet connectivity serves as an empirical proxy for access to modern medical knowledge, 
                            telemedicine, and public health campaigns:
                        </p>
                        <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7;">
                            <li><strong>Connected Regions (>85%):</strong> Northern America, Northern Europe, Western Europe.</li>
                            <li><strong>Emerging Connectivity (20%–40%):</strong> Eastern, Middle, and Western Africa.</li>
                        </ul>
                        <div class="paradox-callout">
                            <strong>Information Equity:</strong> The digital divide directly mirrors the longevity divide, 
                            reflecting shared underlying infrastructure investments in electrical grids and telecommunications.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # -------------------------------------------------------------------------
    # VIEW 5: PART 5 - REGIONAL RANKINGS & SUMMARY TABLE
    # -------------------------------------------------------------------------
    elif selected_view == "📊 Part 5: Indicator Rankings & Summary Table":
        st.markdown("### Regional Indicator Rankings & Detailed Summary Statistics")
        st.caption("Rank regions dynamically by any development, economic, or demographic indicator.")

        r_col1, r_col2 = st.columns([3, 1])

        with r_col1:
            selected_indicator = st.selectbox(
                "Select Ranking Indicator:",
                options=list(INDICATOR_META.keys()),
                format_func=lambda x: INDICATOR_META[x]["title"],
                index=0,
                key="stage5_rank_indicator"
            )

        with r_col2:
            rank_direction = st.radio(
                "Sort Direction:",
                ["Descending (Highest First)", "Ascending (Lowest First)"],
                index=0
            )

        is_ascending = ("Ascending" in rank_direction)

        # Build ranking table
        ranking_df = build_regional_ranking_table(filtered_df, selected_indicator, ascending=is_ascending)

        if not ranking_df.empty:
            meta = INDICATOR_META[selected_indicator]
            val_fmt = meta["fmt"]
            unit = meta["unit"]

            # Visual Ranked Bar Chart
            fig_rank = px.bar(
                ranking_df.reset_index(),
                x="Mean",
                y="region",
                orientation="h",
                color="Mean",
                color_continuous_scale=[[0, "#99F6E4"], [0.5, TEAL_ACCENT], [1, NAVY_PRIMARY]],
                text=[f"{v:{val_fmt}} {unit}" for v in ranking_df["Mean"]],
                custom_data=["Count", "Median", "Min", "Max", "Std"]
            )

            fig_rank.update_traces(
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b> (Rank: %{pointNumber})<br>"
                    f"• Mean: <b>%{{x:{val_fmt}}} {unit}</b><br>"
                    f"• Median: <b>%{{customdata[1]:{val_fmt}}} {unit}</b><br>"
                    f"• Min - Max: %{{customdata[2]:{val_fmt}}} to %{{customdata[3]:{val_fmt}}}<br>"
                    "• Countries Analyzed: %{customdata[0]}<br>"
                    "<extra></extra>"
                )
            )

            fig_rank.update_layout(
                title=dict(
                    text=f"<b>Ranked Regions: {meta['title']}</b>",
                    font=dict(size=14, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                    x=0.01,
                    y=0.98
                ),
                xaxis=dict(title=f"Mean {meta['short_name']} ({unit})", showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(title="", autorange="reversed" if not is_ascending else True),
                paper_bgcolor="white",
                plot_bgcolor="white",
                height=520,
                margin=dict(l=140, r=40, t=50, b=50),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_rank, use_container_width=True)

            # Detailed Formatted Table
            st.markdown("#### Comprehensive Regional Statistical Breakdown")

            formatted_table = ranking_df.copy()
            formatted_table["Mean"] = formatted_table["Mean"].apply(lambda v: f"{v:{val_fmt}} {unit}")
            formatted_table["Median"] = formatted_table["Median"].apply(lambda v: f"{v:{val_fmt}} {unit}")
            formatted_table["Min"] = formatted_table["Min"].apply(lambda v: f"{v:{val_fmt}} {unit}")
            formatted_table["Max"] = formatted_table["Max"].apply(lambda v: f"{v:{val_fmt}} {unit}")
            formatted_table["Std"] = formatted_table["Std"].apply(lambda v: f"{v:{val_fmt}}")

            st.dataframe(
                formatted_table,
                use_container_width=True
            )
        else:
            st.warning("No records available to compute regional rankings.")

