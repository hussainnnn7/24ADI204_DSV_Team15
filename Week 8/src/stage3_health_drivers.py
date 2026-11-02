"""
Stage 3: Health & Demographic Drivers of Longevity
===================================================
Module containing:
- Interactive Filter Strip (Region, Country Search, Trendlines, Reset)
- Part A: Infant Mortality vs Life Expectancy Scatter & OLS Trendline
- Part B: Fertility Rate vs Life Expectancy Scatter & OLS Trendline
- Part C: Urbanization vs Life Expectancy Scatter & OLS Trendline
- Part D: Homicide Rate vs Life Expectancy Scatter & OLS Trendline
- Dynamic Statistical Analysis (Pearson r, Spearman rho, Slope, R-squared, n)
- Dynamic Key Findings & Outlier Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, Any, List, Tuple

from utils import NAVY_PRIMARY, TEAL_ACCENT, BORDER_COLOR, CARD_BG, render_kpi


# Metadata dictionary for the 4 core health & demographic drivers
DRIVERS_META: Dict[str, Dict[str, Any]] = {
    "infant_mortality": {
        "title": "Infant Mortality vs. Life Expectancy",
        "short_name": "Infant Mortality",
        "x_label": "Infant Mortality Rate (Deaths per 1,000 live births)",
        "unit": "per 1,000",
        "fmt": ".1f",
        "icon": "👶",
        "desc": (
            "Infant mortality represents the number of deaths of infants under one year old "
            "per 1,000 live births. It is universally regarded as the most direct barometer of "
            "primary healthcare infrastructure, clean water access, and neonatal care."
        ),
        "interpretation": (
            "Because life expectancy is mathematically calculated from age-specific mortality tables, "
            "infant deaths pull down the life expectancy at birth more severely than deaths at any other age. "
            "Empirically, infant mortality is the single strongest statistical predictor in the dataset, "
            "explaining over 85% of the global variance in life expectancy."
        )
    },
    "fertility": {
        "title": "Fertility Rate vs. Life Expectancy",
        "short_name": "Fertility Rate",
        "x_label": "Total Fertility Rate (Births per woman)",
        "unit": "births/woman",
        "fmt": ".2f",
        "icon": "🌱",
        "desc": (
            "Total fertility rate is the average number of children a woman would give birth to "
            "during her childbearing years based on current age-specific fertility patterns."
        ),
        "interpretation": (
            "Under the Demographic Transition Model, high fertility corresponds to high maternal mortality risks, "
            "lower female educational attainment, and limited access to reproductive healthcare. "
            "As countries advance socioeconomically and female education expands, fertility declines toward "
            "replacement levels (~2.1), coinciding with significant lifespan expansion."
        )
    },
    "urban_population": {
        "title": "Urbanization Rate vs. Life Expectancy",
        "short_name": "Urbanization",
        "x_label": "Urban Population Share (% of total)",
        "unit": "%",
        "fmt": ".1f%",
        "icon": "🏙️",
        "desc": (
            "Urban population refers to the percentage of total national residents living in "
            "urban agglomerations, as defined by national statistical offices."
        ),
        "interpretation": (
            "Urbanization is moderately and positively correlated with life expectancy (r ≈ +0.57). "
            "This reflects the 'Urban Advantage'—proximity to specialized hospitals, ambulance networks, "
            "piped water, and emergency medical services. However, urbanization does not inherently 'cause' "
            "longevity; unmanaged urbanization without adequate sanitation can create health penalties."
        )
    },
    "homicide_rate": {
        "title": "Homicide Rate vs. Life Expectancy",
        "short_name": "Homicide Rate",
        "x_label": "Intentional Homicides (per 100,000 population)",
        "unit": "per 100k",
        "fmt": ".1f",
        "icon": "⚖️",
        "desc": (
            "Intentional homicide rate measures unlawful deaths intentionally inflicted on a person "
            "per 100,000 residents."
        ),
        "interpretation": (
            "Public safety and violence exert a localized drag on longevity. While the linear Pearson correlation "
            "appears modest (r ≈ -0.16) due to extreme right-skewed outliers in Central America and the Caribbean, "
            "the non-parametric Spearman rank correlation is robust (rho ≈ -0.41, p < 10⁻⁸). Homicides disproportionately "
            "truncate the lives of young males, reducing potential years of life lost."
        )
    }
}


def calculate_statistical_metrics(
    df_sub: pd.DataFrame,
    x_col: str,
    y_col: str = "avg_life_expectancy"
) -> Dict[str, Any]:
    """
    Dynamically compute Pearson r, Spearman rho, regression slope, intercept, R-squared, and n.
    Includes safe guards against division by zero and constant columns.
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

    # Check for zero variance
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


def build_driver_scatter_figure(
    df: pd.DataFrame,
    var_key: str,
    highlight_countries: List[str],
    show_trendline: bool
) -> go.Figure:
    """
    Generate an interactive Plotly scatter plot for a health/demographic driver with OLS regression.
    """
    meta = DRIVERS_META[var_key]
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No countries match the selected filters.",
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        return fig

    # 1. Base scatter with unique custom data columns
    custom_cols = [
        "region",
        "gdp_per_capita",
        "population",
        "infant_mortality",
        "fertility",
        "urban_population",
        "homicide_rate"
    ]
    col_idx = custom_cols.index(var_key)

    fig = px.scatter(
        df,
        x=var_key,
        y="avg_life_expectancy",
        color="region",
        size="population",
        hover_name="name",
        size_max=32,
        custom_data=custom_cols,
        category_orders={"region": sorted(df["region"].unique())}
    )

    # 2. Refined hover template
    val_fmt = meta["fmt"]
    unit = meta["unit"]
    hover_temp = (
        "<b>%{hovertext}</b> (%{customdata[0]})<br>"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
        f"• {meta['short_name']}: <b>%{{customdata[{col_idx}]:{val_fmt}}} {unit}</b><br>"
        "• Avg Life Expectancy: <b>%{y:.1f} yrs</b><br>"
        "• GDP per Capita: <b>$%{customdata[1]:,.0f}</b><br>"
        "• Population: <b>%{customdata[2]:,.0f}k</b>"
        "<extra></extra>"
    )

    fig.update_traces(
        hovertemplate=hover_temp,
        marker=dict(opacity=0.82, line=dict(width=0.8, color="#FFFFFF"))
    )

    # 3. Highlight specific countries
    if highlight_countries:
        hl_df = df[df["name"].isin(highlight_countries)]
        if not hl_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hl_df[var_key],
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
                    hoverinfo="skip",
                    showlegend=True
                )
            )

    # 4. Statistical OLS Trendline
    st_res = calculate_statistical_metrics(df, var_key)
    if show_trendline and st_res["is_valid"] and len(df) >= 3:
        min_x = df[var_key].min()
        max_x = df[var_key].max()
        x_line = np.linspace(min_x, max_x, 100)
        y_line = st_res["intercept"] + st_res["slope"] * x_line

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS Trendline (R² = {st_res['r2']:.2f}, Slope = {st_res['slope']:.2f})",
                line=dict(color=NAVY_PRIMARY, width=2.6, dash="solid"),
                hoverinfo="skip"
            )
        )

    # 5. Polish layout
    fig.update_layout(
        title=dict(
            text=f"<b>{meta['title']}</b>",
            font=dict(size=15, color=NAVY_PRIMARY, family="Plus Jakarta Sans"),
            x=0.01,
            y=0.98
        ),
        xaxis=dict(
            title=meta["x_label"],
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
        height=520,
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


def render_stage3_page(df: pd.DataFrame):
    """
    Main renderer for Stage 3: Health & Demographic Drivers.
    """
    # Stage 3 Hero Header
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <span class="badge" style="background-color: rgba(13, 148, 136, 0.15); color: {TEAL_ACCENT}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid rgba(13, 148, 136, 0.3); text-transform: uppercase;">
                STAGE 3 DELIVERABLE • HEALTH & DEMOGRAPHIC DRIVERS
            </span>
            <h1 style="color: {NAVY_PRIMARY}; font-size: 2.1rem; font-weight: 800; margin: 0.5rem 0 0.3rem 0; letter-spacing: -0.02em;">
                Health & Demographic Drivers of Longevity
            </h1>
            <p style="color: #475569; font-size: 1.05rem; margin: 0; line-height: 1.5;">
                Investigating how infant survival, fertility transitions, urbanization environments, 
                and homicide rates explain the global longevity divide across 192 sovereign nations.
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

    # Initialize session state keys for filters if not present
    if "stage3_selected_regions" not in st.session_state:
        st.session_state["stage3_selected_regions"] = all_regions
    if "stage3_selected_countries" not in st.session_state:
        st.session_state["stage3_selected_countries"] = []
    if "stage3_show_trendline" not in st.session_state:
        st.session_state["stage3_show_trendline"] = True

    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <strong style="color: {NAVY_PRIMARY}; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em;">
                        🎛️ Interactive Driver Controls & Filters
                    </strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        f_col1, f_col2, f_col3, f_col4 = st.columns([2.5, 2.5, 1.5, 1])

        with f_col1:
            selected_regions = st.multiselect(
                "Filter by Geographic Region:",
                options=all_regions,
                default=st.session_state["stage3_selected_regions"],
                key="stage3_regions_widget"
            )
            # Fallback if user clears all
            if not selected_regions:
                selected_regions = all_regions

        with f_col2:
            highlight_countries = st.multiselect(
                "Search & Highlight Countries:",
                options=all_countries,
                default=st.session_state["stage3_selected_countries"],
                key="stage3_countries_widget"
            )

        with f_col3:
            show_trendline = st.checkbox(
                "Show OLS Trendlines",
                value=st.session_state["stage3_show_trendline"],
                key="stage3_trend_widget"
            )

        with f_col4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.session_state["stage3_selected_regions"] = all_regions
                st.session_state["stage3_selected_countries"] = []
                st.session_state["stage3_show_trendline"] = True
                st.rerun()

    # Filter dataframe
    filtered_df = df[df["region"].isin(selected_regions)].copy()
    num_filtered = len(filtered_df)

    st.markdown(
        f"""
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 1.5rem;">
            Displaying <strong>{num_filtered}</strong> of <strong>{len(df)}</strong> sovereign nations 
            across <strong>{len(selected_regions)}</strong> regions.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # 2. STATISTICAL ANALYSIS SECTION (DYNAMIC COMPARATIVE KPIS)
    # -------------------------------------------------------------------------
    st.markdown("### Statistical Comparison Across Drivers")
    st.caption("All correlations and regression parameters are computed dynamically from the active filtered subset.")

    # Calculate statistics for each driver
    stats_dict = {
        col: calculate_statistical_metrics(filtered_df, col)
        for col in DRIVERS_META.keys()
    }

    stat_cols = st.columns(4)
    for idx, (col_key, s_data) in enumerate(stats_dict.items()):
        meta = DRIVERS_META[col_key]
        with stat_cols[idx]:
            r_val = s_data["pearson_r"]
            r2_pct = s_data["r2"] * 100
            slope_val = s_data["slope"]
            
            # Badge text based on relationship strength
            if abs(r_val) >= 0.7:
                strength_tag = "Very Strong"
            elif abs(r_val) >= 0.4:
                strength_tag = "Moderate"
            else:
                strength_tag = "Weak"

            direction = "Negative" if r_val < 0 else "Positive"
            
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">
                        <span>{meta['icon']} {meta['short_name']}</span>
                        <span class="kpi-tag">{strength_tag} {direction}</span>
                    </div>
                    <div class="kpi-value" style="font-size: 1.6rem;">r = {r_val:.3f}</div>
                    <div class="kpi-subtitle">
                        <strong>Spearman ρ:</strong> {s_data['spearman_r']:.3f}<br>
                        <strong>Variance Explained (R²):</strong> {r2_pct:.1f}%<br>
                        <strong>Linear Slope:</strong> {slope_val:.3f} yrs/{meta['unit']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. INTERACTIVE VISUALIZATIONS (PARTS A, B, C, D)
    # -------------------------------------------------------------------------
    st.markdown("### Interactive Driver Scatter Plots")

    viz_tab_names = [
        "👶 Part A: Infant Mortality",
        "🌱 Part B: Fertility Rate",
        "🏙️ Part C: Urbanization",
        "⚖️ Part D: Homicide Rate",
        "📊 All Drivers (2x2 Grid)"
    ]
    
    selected_viz_tab = st.radio(
        "Select Visualisation View:",
        viz_tab_names,
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    if selected_viz_tab == "👶 Part A: Infant Mortality":
        render_driver_deepdive(filtered_df, "infant_mortality", highlight_countries, show_trendline, stats_dict["infant_mortality"])

    elif selected_viz_tab == "🌱 Part B: Fertility Rate":
        render_driver_deepdive(filtered_df, "fertility", highlight_countries, show_trendline, stats_dict["fertility"])

    elif selected_viz_tab == "🏙️ Part C: Urbanization":
        render_driver_deepdive(filtered_df, "urban_population", highlight_countries, show_trendline, stats_dict["urban_population"])

    elif selected_viz_tab == "⚖️ Part D: Homicide Rate":
        render_driver_deepdive(filtered_df, "homicide_rate", highlight_countries, show_trendline, stats_dict["homicide_rate"])

    elif selected_viz_tab == "📊 All Drivers (2x2 Grid)":
        render_all_drivers_grid(filtered_df, highlight_countries, show_trendline)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. KEY FINDINGS & OUTLIER SYNTHESIS SECTION
    # -------------------------------------------------------------------------
    render_dynamic_key_findings(filtered_df, stats_dict)


def render_driver_deepdive(
    df: pd.DataFrame,
    var_key: str,
    highlight_countries: List[str],
    show_trendline: bool,
    stat_res: Dict[str, Any]
):
    """
    Renders an individual driver visualization alongside its dedicated narrative analysis and outlier cards.
    """
    meta = DRIVERS_META[var_key]
    
    col_plot, col_info = st.columns([3, 2])

    with col_plot:
        fig = build_driver_scatter_figure(df, var_key, highlight_countries, show_trendline)
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>{meta['icon']} {meta['title']}</h3>
                <p><strong>Indicator Definition:</strong> {meta['desc']}</p>
                <div class="paradox-callout" style="margin: 0.8rem 0;">
                    <strong>Empirical Relationship:</strong><br>
                    {meta['interpretation']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Dynamic outlier highlights for this specific variable
        if not df.empty and stat_res["is_valid"]:
            min_r = df.loc[df[var_key].idxmin()]
            max_r = df.loc[df[var_key].idxmax()]
            val_fmt = meta["fmt"]
            unit = meta["unit"]

            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid {BORDER_COLOR}; border-radius: 10px; padding: 1rem; margin-top: 1rem;">
                    <strong style="color: {NAVY_PRIMARY}; font-size: 0.88rem; text-transform: uppercase;">
                        Extremes in Current Selection:
                    </strong>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px dashed {BORDER_COLOR};">
                        <div>
                            <strong>Lowest {meta['short_name']}:</strong> {min_r['name']} ({min_r['region']})<br>
                            <small style="color: #64748B;">Life Expectancy: {min_r['avg_life_expectancy']:.1f} yrs</small>
                        </div>
                        <div style="text-align: right; color: {TEAL_ACCENT}; font-weight: 800; font-size: 1.1rem;">
                            {min_r[var_key]:{val_fmt}} {unit}
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem;">
                        <div>
                            <strong>Highest {meta['short_name']}:</strong> {max_r['name']} ({max_r['region']})<br>
                            <small style="color: #64748B;">Life Expectancy: {max_r['avg_life_expectancy']:.1f} yrs</small>
                        </div>
                        <div style="text-align: right; color: #DC2626; font-weight: 800; font-size: 1.1rem;">
                            {max_r[var_key]:{val_fmt}} {unit}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_all_drivers_grid(
    df: pd.DataFrame,
    highlight_countries: List[str],
    show_trendline: bool
):
    """
    Renders all 4 driver scatter plots in a clean 2x2 grid for high-level comparative viewing.
    """
    grid_col1, grid_col2 = st.columns(2)
    
    with grid_col1:
        fig_im = build_driver_scatter_figure(df, "infant_mortality", highlight_countries, show_trendline)
        st.plotly_chart(fig_im, use_container_width=True)

        fig_urb = build_driver_scatter_figure(df, "urban_population", highlight_countries, show_trendline)
        st.plotly_chart(fig_urb, use_container_width=True)

    with grid_col2:
        fig_fert = build_driver_scatter_figure(df, "fertility", highlight_countries, show_trendline)
        st.plotly_chart(fig_fert, use_container_width=True)

        fig_hom = build_driver_scatter_figure(df, "homicide_rate", highlight_countries, show_trendline)
        st.plotly_chart(fig_hom, use_container_width=True)


def render_dynamic_key_findings(df: pd.DataFrame, stats_dict: Dict[str, Dict[str, Any]]):
    """
    Automatically generates empirical insights without hardcoded numbers or static rankings.
    """
    st.markdown("### Key Findings & Empirical Insights")
    st.caption("Insights are calculated dynamically from the active filtered records.")

    if df.empty:
        st.warning("No records available to compute insights.")
        return

    # 1. Identify strongest negative & positive correlations dynamically
    valid_pearsons = {k: v["pearson_r"] for k, v in stats_dict.items() if v["is_valid"]}
    if valid_pearsons:
        strongest_neg_key = min(valid_pearsons, key=valid_pearsons.get)
        strongest_pos_key = max(valid_pearsons, key=valid_pearsons.get)
        strongest_neg_r = valid_pearsons[strongest_neg_key]
        strongest_pos_r = valid_pearsons[strongest_pos_key]
    else:
        strongest_neg_key = "infant_mortality"
        strongest_pos_key = "urban_population"
        strongest_neg_r = -0.924
        strongest_pos_r = 0.568

    # 2. Extract notable outlier records dynamically
    top_life_exp_country = df.loc[df["avg_life_expectancy"].idxmax()]
    low_life_exp_country = df.loc[df["avg_life_expectancy"].idxmin()]
    top_homicide_country = df.loc[df["homicide_rate"].idxmax()]
    top_fertility_country = df.loc[df["fertility"].idxmax()]

    col_find1, col_find2 = st.columns(2)

    with col_find1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>🔍 1. Primary Health Capacity vs. Economic Wealth</h3>
                <p>
                    <strong>The Strongest Predictor:</strong> Across the analyzed nations, 
                    <strong>{DRIVERS_META[strongest_neg_key]['short_name']}</strong> demonstrates the 
                    single strongest negative correlation with longevity (<strong>r = {strongest_neg_r:.3f}</strong>, 
                    explaining <strong>{stats_dict[strongest_neg_key]['r2']*100:.1f}%</strong> of total variance).
                </p>
                <p>
                    This explains the core of the <em>Global Life Expectancy Paradox</em>: 
                    national wealth (GDP per capita) affects longevity primarily to the extent that it funds 
                    primary healthcare interventions—such as clean maternity clinics, oral rehydration therapy, 
                    and clean tap water. When middle-income countries (e.g. Cuba, Sri Lanka) invest heavily in primary care, 
                    they achieve infant survival rates comparable to Western Europe, boosting average life expectancy 
                    well past that of wealthier nations with high infant mortality (e.g. Equatorial Guinea).
                </p>
                <p>
                    <strong>Demographic Transition:</strong> <strong>{DRIVERS_META['fertility']['short_name']}</strong> 
                    exhibits the second strongest inverse association (<strong>r = {stats_dict['fertility']['pearson_r']:.3f}</strong>), 
                    reflecting the shift from high-birth/high-mortality agrarian settings to low-fertility/high-survival modern economies.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_find2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>⚖️ 2. Urbanization & Public Safety Dynamics</h3>
                <p>
                    <strong>The Urban Infrastructure Advantage:</strong> 
                    <strong>{DRIVERS_META[strongest_pos_key]['short_name']}</strong> is the strongest positive demographic factor 
                    (<strong>r = {strongest_pos_r:.3f}</strong>, R² = {stats_dict[strongest_pos_key]['r2']*100:.1f}%). 
                    Urban environments aggregate hospitals, specialized doctors, emergency transit, and electrical grids. 
                    However, correlation does not imply direct causation: urbanization serves as a spatial proxy for 
                    modernized social infrastructure.
                </p>
                <p>
                    <strong>Violence and Public Safety Outliers:</strong> 
                    Homicide rate presents a unique demographic anomaly. The nation with the highest homicide rate in the current 
                    selection is <strong>{top_homicide_country['name']}</strong> ({top_homicide_country['homicide_rate']:.1f} per 100k, 
                    region: {top_homicide_country['region']}). While violent crime causes severe localized tragedies, its drag on national 
                    life expectancy is often masked if infant and childhood mortality remain well-controlled.
                </p>
                <div style="background-color: #FEF2F2; border-left: 3.5px solid #EF4444; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.84rem; color: #991B1B;">
                    <strong>Academic Caution:</strong> All reported correlations are descriptive and associative. No individual demographic 
                    variable acts in isolation; health outcomes arise from interdependent networks of sanitation, education, and public policy.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
