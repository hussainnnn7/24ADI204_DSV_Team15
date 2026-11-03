"""
Stage 2: The Wealth vs Health Paradox
======================================
Module containing:
- Part 1: Interactive Preston Curve (GDP vs Life Expectancy Scatter)
- Part 2: Interactive World Choropleth Map (Multi-indicator Geospatial)
- Part 3: Empirical Key Insights (Outliers, Diminishing Returns, Regional Divide)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from utils import NAVY_PRIMARY, TEAL_ACCENT, BORDER_COLOR, CARD_BG


# Indicator metadata configuration for the World Choropleth Map
MAP_INDICATOR_CONFIG: Dict[str, Dict[str, Any]] = {
    "Average Life Expectancy": {
        "col": "avg_life_expectancy",
        "unit": "Years",
        "fmt": ".1f",
        "scale": "Tealgrn",
        "desc": "Average expected lifespan at birth in years (combined male and female)."
    },
    "GDP per Capita": {
        "col": "gdp_per_capita",
        "unit": "Current US$",
        "fmt": ",.0f",
        "scale": "Blues",
        "desc": "Gross Domestic Product per person in nominal US dollars."
    },
    "Infant Mortality Rate": {
        "col": "infant_mortality",
        "unit": "Per 1,000 live births",
        "fmt": ".1f",
        "scale": "YlOrRd",
        "desc": "Number of infant deaths before age 1 per 1,000 live births."
    },
    "Internet Usage": {
        "col": "internet_users",
        "unit": "% of population",
        "fmt": ".1f",
        "scale": "Purples",
        "desc": "Percentage of individuals using the internet."
    },
    "Female Education Enrollment": {
        "col": "secondary_school_enrollment_female",
        "unit": "Gross enrollment %",
        "fmt": ".1f",
        "scale": "Viridis",
        "desc": "Gross secondary school enrollment ratio for females (%)."
    }
}


def build_preston_curve_figure(
    df: pd.DataFrame,
    scale_type: str,
    selected_regions: List[str],
    highlight_countries: List[str],
    show_trendline: bool,
    filter_to_selected_countries: bool = False
) -> go.Figure:
    """
    Generate the interactive Preston Curve scatter plot using Plotly.
    """
    # 1. Filter by region
    filtered = df[df["region"].isin(selected_regions)].copy()
    
    # Optional: filter to highlight countries only if requested
    if filter_to_selected_countries and highlight_countries:
        filtered = filtered[filtered["name"].isin(highlight_countries)]
    
    if filtered.empty:
        # Graceful empty state
        fig = go.Figure()
        fig.update_layout(
            title="No countries match the selected filters.",
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        return fig

    # 2. Add highlight flag
    filtered["is_highlighted"] = filtered["name"].isin(highlight_countries) if highlight_countries else False
    
    is_log = (scale_type == "Logarithmic Scale (log10 US$)")
    
    # 3. Base scatter with Plotly Express
    fig = px.scatter(
        filtered,
        x="gdp_per_capita",
        y="avg_life_expectancy",
        size="population",
        color="region",
        hover_name="name",
        log_x=is_log,
        size_max=36,
        custom_data=[
            "region",
            "gdp_per_capita",
            "avg_life_expectancy",
            "infant_mortality",
            "secondary_school_enrollment_female",
            "secondary_school_enrollment_male",
            "population"
        ],
        category_orders={"region": sorted(filtered["region"].unique())}
    )

    # 4. Standardize hovertemplate for all region traces
    hover_template = (
        "<b>%{hovertext}</b> (%{customdata[0]})<br>"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        "• Avg Life Expectancy: <b>%{customdata[2]:.1f} yrs</b><br>"
        "• GDP per Capita: <b>$%{customdata[1]:,.0f}</b><br>"
        "• Infant Mortality: <b>%{customdata[3]:.1f} per 1k</b><br>"
        "• Female Secondary School: <b>%{customdata[4]:.1f}%</b><br>"
        "• Male Secondary School: <b>%{customdata[5]:.1f}%</b><br>"
        "• Population: <b>%{customdata[6]:,.0f}k</b>"
        "<extra></extra>"
    )
    fig.update_traces(
        hovertemplate=hover_template,
        marker=dict(opacity=0.82, line=dict(width=0.8, color="#FFFFFF"))
    )

    # 5. Highlight specific selected countries with a distinct halo ring
    if highlight_countries and not filter_to_selected_countries:
        hl_df = filtered[filtered["name"].isin(highlight_countries)]
        if not hl_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hl_df["gdp_per_capita"],
                    y=hl_df["avg_life_expectancy"],
                    mode="markers+text",
                    text=hl_df["name"],
                    textposition="top center",
                    textfont=dict(size=11, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
                    marker=dict(
                        symbol="circle-open",
                        size=34,
                        color="#DC2626",
                        line=dict(width=2.5, color="#DC2626")
                    ),
                    name="Highlighted Countries",
                    hoverinfo="skip",
                    showlegend=True
                )
            )

    # 6. Fit statistical Preston Curve (Logarithmic OLS: y = beta_0 + beta_1 * ln(x))
    if show_trendline and len(filtered) >= 5:
        valid_data = filtered[(filtered["gdp_per_capita"] > 0) & filtered["avg_life_expectancy"].notna()]
        if len(valid_data) >= 5:
            log_x = np.log(valid_data["gdp_per_capita"])
            y = valid_data["avg_life_expectancy"]
            
            # Linear fit on log scale: y = p[1] + p[0] * ln(x)
            slope, intercept = np.polyfit(log_x, y, 1)
            
            # Predict across continuous range
            min_x = valid_data["gdp_per_capita"].min()
            max_x = valid_data["gdp_per_capita"].max()
            x_line = np.geomspace(min_x, max_x, 150)
            y_line = intercept + slope * np.log(x_line)
            
            # Calculate R-squared
            r2 = np.corrcoef(log_x, y)[0, 1] ** 2
            
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Preston Curve Fit (R² = {r2:.2f})",
                    line=dict(color=NAVY_PRIMARY, width=2.8, dash="solid"),
                    hoverinfo="skip"
                )
            )

    # 7. Polish layout & axes
    x_title = "GDP per Capita (Current US$ - Logarithmic Scale)" if is_log else "GDP per Capita (Current US$ - Linear Scale)"
    
    fig.update_layout(
        title=dict(
            text="<b>The Preston Curve: GDP per Capita vs. Life Expectancy (192 Countries)</b>",
            font=dict(size=16, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            zeroline=False,
            tickprefix="$" if not is_log else None,
            tickformat=","
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
        height=620,
        margin=dict(l=60, r=40, t=65, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#334155"),
            title=dict(text="")
        )
    )

    return fig


def build_world_choropleth_figure(
    df: pd.DataFrame,
    indicator_key: str,
    map_projection: str = "natural earth"
) -> go.Figure:
    """
    Generate an interactive World Choropleth Map using Plotly.
    """
    config = MAP_INDICATOR_CONFIG[indicator_key]
    col_name = config["col"]
    scale_name = config["scale"]
    unit_label = config["unit"]
    fmt = config["fmt"]

    # Define unique custom data columns to prevent duplicate selection errors in Plotly
    custom_cols = [
        "region",
        "iso2",
        "avg_life_expectancy",
        "gdp_per_capita",
        "infant_mortality",
        "internet_users",
        "secondary_school_enrollment_female"
    ]
    col_idx = custom_cols.index(col_name)

    # Build figure
    fig = px.choropleth(
        df,
        locations="iso3",
        color=col_name,
        hover_name="name",
        color_continuous_scale=scale_name,
        projection=map_projection,
        custom_data=custom_cols
    )

    # Formatting hovertemplate
    if "gdp" in col_name.lower():
        val_str = f"$%{{customdata[{col_idx}]:,.0f}}"
    else:
        val_str = f"%{{customdata[{col_idx}]:{fmt}}} {unit_label}"

    hover_temp = (
        "<b>%{hovertext}</b> (%{customdata[0]})<br>"
        "ISO Code: %{customdata[1]} | %{location}<br>"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        f"• {indicator_key}: <b>{val_str}</b><br>"
        "• Avg Life Expectancy: <b>%{customdata[2]:.1f} yrs</b><br>"
        "• GDP per Capita: <b>$%{customdata[3]:,.0f}</b>"
        "<extra></extra>"
    )

    fig.update_traces(
        hovertemplate=hover_temp,
        marker_line_color="#CBD5E1",
        marker_line_width=0.6
    )

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#CBD5E1",
        showland=True,
        landcolor="#F8FAFC",
        showocean=True,
        oceancolor="#EDF2F7",
        showlakes=True,
        lakecolor="#EDF2F7",
        showcountries=True,
        countrycolor="#E2E8F0"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Global Geospatial Distribution: {indicator_key}</b>",
            font=dict(size=16, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text=unit_label,
                font=dict(size=12, color=NAVY_PRIMARY)
            ),
            thickness=15,
            len=0.75,
            x=1.02
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=580,
        margin=dict(l=0, r=0, t=50, b=10)
    )

    return fig


def render_stage2_page(df: pd.DataFrame):
    """
    Main renderer for Stage 2: The Wealth vs Health Paradox.
    """
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <span class="badge" style="background-color: rgba(13, 148, 136, 0.15); color: {TEAL_ACCENT}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(13, 148, 136, 0.3); text-transform: uppercase;">
                STAGE 2 DELIVERABLE • THE WEALTH VS HEALTH PARADOX
            </span>
            <h1 style="color: {NAVY_PRIMARY}; font-size: 2.1rem; font-weight: 800; margin: 0.5rem 0 0.3rem 0; letter-spacing: -0.02em;">
                The Wealth vs Health Paradox
            </h1>
            <p style="color: #475569; font-size: 1.05rem; margin: 0; line-height: 1.5;">
                Investigating the nonlinear relationship between economic prosperity (GDP per Capita) 
                and population longevity (Average Life Expectancy) across 192 sovereign nations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================================================================
    # PART 1: INTERACTIVE PRESTON CURVE
    # =========================================================================
    st.markdown("### Part 1: The Interactive Preston Curve")
    st.caption(
        "First formulated by demographer Samuel H. Preston (1975), the Preston Curve charts the "
        "cross-sectional relationship between national income per person and life expectancy at birth."
    )

    # Filter & Control Strip
    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                <strong style="color: {NAVY_PRIMARY}; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em;">
                    🎛️ Preston Curve Interactive Controls
                </strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 2.5, 2])

        with ctrl_col1:
            scale_type = st.radio(
                "GDP Scale Type:",
                ["Logarithmic Scale (log10 US$)", "Linear Scale (US$)"],
                index=0,
                help="Log scale normalizes extreme wealth disparities, linear scale visualizes the classic saturation curve."
            )
            show_trendline = st.checkbox(
                "Show Preston Curve Fit (Log OLS)",
                value=True,
                help="Overlays the logarithmic regression curve: Life Expectancy = β₀ + β₁ · ln(GDP)"
            )

        with ctrl_col2:
            all_regions = sorted(df["region"].unique().tolist())
            selected_regions = st.multiselect(
                "Filter by Geographic Region (UN):",
                options=all_regions,
                default=all_regions,
                help="Select one or more regions to isolate on the scatter plot."
            )
            if not selected_regions:
                selected_regions = all_regions

        with ctrl_col3:
            all_countries = sorted(df["name"].unique().tolist())
            highlight_countries = st.multiselect(
                "Search & Highlight Country:",
                options=all_countries,
                default=[],
                help="Pick countries to highlight with high-visibility markers and name labels."
            )
            filter_to_selected = st.checkbox(
                "Isolate highlighted countries only",
                value=False,
                disabled=len(highlight_countries) == 0,
                help="Only show the picked countries, filtering out all others."
            )

    # Build and display Preston Curve figure
    fig_preston = build_preston_curve_figure(
        df=df,
        scale_type=scale_type,
        selected_regions=selected_regions,
        highlight_countries=highlight_countries,
        show_trendline=show_trendline,
        filter_to_selected_countries=filter_to_selected
    )
    st.plotly_chart(fig_preston, use_container_width=True)

    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border-left: 4px solid {NAVY_PRIMARY}; border-radius: 4px; padding: 0.85rem 1.2rem; margin-top: -0.5rem; margin-bottom: 2rem; font-size: 0.9rem; color: #334155;">
            <strong>How to read this plot:</strong> Each bubble represents a country, where bubble area is proportional to 
            population (in thousands) and color corresponds to the geographic region. In logarithmic view, the relationship 
            approximates a linear trend (R² ≈ 0.64). In linear view, notice how the curve rises precipitously between 
            $0 and $15,000, after which additional per-capita income yields progressively smaller longevity returns.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =========================================================================
    # PART 2: INTERACTIVE WORLD CHOROPLETH MAP
    # =========================================================================
    st.markdown("### Part 2: Interactive World Choropleth Map")
    st.caption(
        "Geospatial visualization of global indicators across 192 countries using verified ISO-3 country codes."
    )

    map_ctrl1, map_ctrl2 = st.columns([2.5, 1.5])
    
    with map_ctrl1:
        selected_indicator_key = st.selectbox(
            "Select Map Indicator:",
            list(MAP_INDICATOR_CONFIG.keys()),
            index=0,
            help="Switch the color-mapped metric across all 192 sovereign countries."
        )

    with map_ctrl2:
        selected_projection = st.selectbox(
            "Map Projection:",
            ["natural earth", "robinson", "equirectangular", "mercator"],
            index=0,
            help="Select cartographic projection."
        )

    # Indicator Info Badge
    ind_cfg = MAP_INDICATOR_CONFIG[selected_indicator_key]
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.88rem; color: #334155;">
            <strong style="color: {NAVY_PRIMARY};">Indicator Definition:</strong> {ind_cfg['desc']} 
            <span style="color: {TEAL_ACCENT}; font-weight: 700; margin-left: 0.5rem;">Unit: {ind_cfg['unit']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Build and display Map figure
    fig_map = build_world_choropleth_figure(
        df=df,
        indicator_key=selected_indicator_key,
        map_projection=selected_projection
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # PART 3: KEY EMPIRICAL INSIGHTS
    # =========================================================================
    render_stage2_insights(df)


def render_stage2_insights(df: pd.DataFrame):
    """
    Renders structured analytical insights regarding the Wealth vs Health Paradox.
    """
    st.markdown("### Part 3: Key Empirical Insights")
    st.caption("Analytical synthesis of the relationship between national income and population longevity.")

    col_ins1, col_ins2 = st.columns(2)

    with col_ins1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>📈 1. Wealth & Longevity: The Diminishing Returns Law</h3>
                <p>
                    Across the 192 countries, the data demonstrates that <strong>higher GDP per capita generally associates 
                    with higher life expectancy</strong>, but the relationship is strictly <strong>concave and nonlinear</strong>.
                </p>
                <ul style="color: #334155; font-size: 0.92rem; line-height: 1.7; margin-bottom: 0.8rem;">
                    <li>
                        <strong>The Steep Threshold ($0 – $15,000):</strong> In developing nations, modest gains in economic 
                        output correlate with dramatic gains in lifespan—rising from ~53 to ~75 years as basic sanitation, 
                        nutrition, and maternal healthcare become accessible.
                    </li>
                    <li>
                        <strong>The Saturation Plateau (&gt; $25,000):</strong> Past moderate income thresholds, the Preston 
                        Curve flattens noticeably. Adding $20,000 of GDP per capita in an advanced economy produces negligible 
                        additional longevity gains, as chronic lifestyle illnesses, environmental stressors, and healthcare 
                        distribution inefficiencies become dominant.
                    </li>
                    <li>
                        <strong>Linear vs. Log Correlation:</strong> The Pearson correlation between linear GDP per capita and 
                        average life expectancy is only <strong>r = +0.49</strong>, whereas the correlation with log-transformed 
                        GDP jumps to <strong>r = +0.80</strong> (R² = 0.64), empirically validating the logarithmic nature of the curve.
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-card">
                <h3>🌟 2. Positive Outliers: Longevity Overachievers</h3>
                <p>
                    Several nations achieve life expectancies rivaling the wealthiest nations in the world despite having 
                    only modest per-capita income:
                </p>
                <div style="background-color: #F8FAFC; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.9rem; font-size: 0.88rem; line-height: 1.6;">
                    • <strong>🇨🇷 Costa Rica (GDP/cap: $12,027):</strong> Reaches <strong>80.05 years</strong> life expectancy 
                    (surpassing the United States at $62,918), driven by integrated universal primary health clinics and high literacy.<br>
                    • <strong>🇨🇺 Cuba (GDP/cap: $8,822):</strong> Achieves <strong>78.70 years</strong> with infant mortality of just 
                    <strong>4.5 per 1,000</strong>, outperforming far wealthier regional neighbors.<br>
                    • <strong>🇱🇰 Sri Lanka (GDP/cap: $4,190):</strong> Attains <strong>76.65 years</strong> lifespan with 
                    <strong>102.6%</strong> female secondary enrollment.<br>
                    • <strong>🇻🇳 Viet Nam (GDP/cap: $2,563):</strong> Records <strong>75.30 years</strong> average longevity 
                    despite lower-middle-income status.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_ins2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>⚠️ 3. Negative Outliers: Longevity Underachievers</h3>
                <p>
                    Conversely, certain nations possess substantial per-capita GDP yet lag behind in population longevity, 
                    illustrating the paradox in reverse:
                </p>
                <div style="background-color: #F8FAFC; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.9rem; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.8rem;">
                    • <strong>🇬🇶 Equatorial Guinea (GDP/cap: $10,179):</strong> Despite oil revenues creating an income level 
                    comparable to Costa Rica and the Maldives, average life expectancy is only <strong>58.35 years</strong>, 
                    with severe infant mortality (<strong>66.1 per 1k</strong>) and low secondary education.<br>
                    • <strong>🇿🇦 South Africa (GDP/cap: $6,369):</strong> Records <strong>63.65 years</strong> life expectancy, 
                    constrained by deep structural inequalities and infectious disease burdens.<br>
                    • <strong>🇬🇦 Gabon (GDP/cap: $8,019):</strong> Reaches only <strong>66.10 years</strong> despite upper-middle 
                    resource income.<br>
                    • <strong>🇺🇸 United States (GDP/cap: $62,918):</strong> A high-income outlier with <strong>78.80 years</strong>, 
                    falling behind Costa Rica, Chile ($15,923), and Japan ($39,082 at 84.40 yrs).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-card">
                <h3>🗺️ 4. Regional Divide & Methodological Boundaries</h3>
                <p>
                    <strong>Macro-Regional Clustering:</strong>
                </p>
                <ul style="color: #334155; font-size: 0.9rem; line-height: 1.65; margin-bottom: 0.8rem;">
                    <li>
                        <strong>Top Longevity Clusters:</strong> Western Europe, Northern Europe, and Eastern Asia cluster 
                        between <strong>80.0 and 84.4 years</strong>, characterized by high wealth and infant mortality under 4 per 1k.
                    </li>
                    <li>
                        <strong>Latin America & Caribbean:</strong> Demonstrates high health efficiency, achieving 
                        <strong>75.0 – 80.0 years</strong> across modest income levels ($5,000 – $15,000).
                    </li>
                    <li>
                        <strong>Sub-Saharan Africa:</strong> Clusters below <strong>65.0 years</strong> (Central African Republic lowest at 52.7 yrs), 
                        where infant mortality frequently exceeds 50 per 1,000.
                    </li>
                </ul>
                <div style="background-color: #FEF2F2; border-left: 3.5px solid #EF4444; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.84rem; color: #991B1B;">
                    <strong>Methodological Integrity Notice:</strong> These relationships represent empirical associations and correlations; 
                    they do not establish direct causality. National wealth functions as an enabling condition whose impact is mediated by 
                    governance, healthcare infrastructure, education, and social equity.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
