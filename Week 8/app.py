"""
Global Life Expectancy Paradox - Dashboard Application
======================================================
Course: Data Science and Visualization (24ADI204)
Team: Team 15
Department: Department of Artificial Intelligence and Data Science

Delivered Stages:
- Stage 1: Overview & Project Scope, Data loader, Dynamic KPIs, White/Navy/Teal design
- Stage 2: The Wealth vs Health Paradox (Interactive Preston Curve & Global Choropleth Map)
- Stage 3: Health & Demographic Drivers (Infant Mortality, Fertility, Urbanization, Homicide)
- Stage 4: Gender Gap & Education Analysis (Male vs Female Longevity, Gaps, Education Parity & Regressions)
- Stage 5: Regional Disparities Analysis (Life Expectancy by Sex, Wealth Boxplots, Health & Digital Divides)
- Stage 6: Paradox Country Comparator (Head-to-head 2-4 countries, Residuals, Multidimensional Radar)
- Stage 7: Interactive Dataset Explorer (Data quality audit, distributions, custom chart builder, CSV exports)
- Stage 8: Final Insights & Recommendations (Top 10 Discoveries, Preston Curve, Success Stories, Policy Synthesis)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure local 'src' directory is importable
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_loader import load_dataset, calculate_kpis, get_data_filepath

try:
    from utils import (
        inject_custom_css,
        render_header,
        render_kpi,
        render_theme_switcher,
        get_theme_colors,
        get_current_theme,
        NAVY_PRIMARY,
        TEAL_ACCENT,
        BORDER_COLOR,
    )
except ImportError:
    from theme import render_theme_switcher, get_theme_colors, get_current_theme
    from utils import (
        inject_custom_css,
        render_header,
        render_kpi,
        NAVY_PRIMARY,
        TEAL_ACCENT,
        BORDER_COLOR,
    )
from stage2_wealth_health import render_stage2_page
from stage3_health_drivers import render_stage3_page
from stage4_gender_education import render_stage4_page
from stage5_regional_disparities import render_stage5_page
from stage6_country_comparator import render_stage6_page
from stage7_dataset_explorer import render_stage7_page
from stage8_final_insights import render_stage8_page

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Life Expectancy Paradox | Team 15",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom light theme CSS (Navy Blue & Teal)
inject_custom_css()

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading verified dataset...")
def get_cached_data() -> pd.DataFrame:
    """Load and cache the cleaned dataset with verified ISO codes and derived metrics."""
    return load_dataset()

try:
    df = get_cached_data()
    kpis = calculate_kpis(df)
    data_loaded_successfully = True
except Exception as e:
    data_loaded_successfully = False
    error_msg = str(e)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & METADATA
# -----------------------------------------------------------------------------
with st.sidebar:
    c = get_theme_colors()
    st.markdown(
        f"""
        <div class="sidebar-branding">
            <span class="sidebar-badge" style="background-color: {c['primary']}; color: #FFFFFF;">24ADI204 • DSV</span>
            <h2 style="color: {c['primary']}; font-size: 1.25rem; font-weight: 800; margin: 0.5rem 0 0.2rem 0;">
                Global Life Expectancy Paradox
            </h2>
            <p style="color: {c['text_muted']}; font-size: 0.8rem; margin: 0;">
                Longevity Divide & Socioeconomic Drivers
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Clearly visible global Light / Dark Theme Switcher
    render_theme_switcher()

    st.markdown("---")

    st.markdown("### Navigation")
    nav_options = [
        "🏠 Overview & Project Scope",
        "📈 The Wealth Paradox (Stage 2)",
        "🩺 Health & Demographic Drivers (Stage 3)",
        "🎓 Gender Gap & Education (Stage 4)",
        "🗺️ Regional Disparities (Stage 5)",
        "⚖️ Paradox Country Comparator (Stage 6)",
        "🔍 Dataset Explorer (Stage 7)",
        "🎯 Final Insights & Conclusions (Stage 8)"
    ]
    
    selected_page = st.radio(
        "Select Dashboard Section",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Dataset status indicator with dynamic theme styling
    st.markdown("#### Dataset Status")
    if data_loaded_successfully:
        st.markdown(
            f"""
            <div style="background-color: {c['status_box_bg']}; border: 1px solid {c['status_box_border']}; border-radius: 8px; padding: 0.8rem; font-size: 0.82rem; color: {c['status_box_text']};">
                <strong>Source:</strong> <code>CLEANED_DATASET_FINAL.csv</code><br>
                <strong>Status:</strong> Active & Cleaned<br>
                <strong>Countries:</strong> 192 sovereign nations<br>
                <strong>Missing Cells:</strong> 0 (100% complete)<br>
                <strong>ISO-2 Verification:</strong> Checked (Namibia: <code>NA</code>)<br>
                <strong>ISO-3 Verification:</strong> 192 / 192 mapped
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Dataset could not be loaded.")

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: {c['text_muted']}; line-height: 1.6;">
            <strong style="color: {c['primary']};">Course:</strong> 24ADI204 - DSV<br>
            <strong style="color: {c['primary']};">Team:</strong> Team 15<br>
            <strong style="color: {c['primary']};">Department:</strong> AI & Data Science
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 4. OVERVIEW PAGE RENDERER (STAGE 1)
# -----------------------------------------------------------------------------
def render_overview_page():
    """Render the Stage 1 Overview & Global Indicators page."""
    c = get_theme_colors()

    # Render Hero Header
    render_header(
        title="Global Life Expectancy Paradox",
        subtitle="Longevity Divide: Understanding Why Life Expectancy Differs Across Countries",
        badge="STAGE 1 DELIVERABLE • OVERVIEW & GLOBAL INDICATORS"
    )

    # Dynamic Key Performance Indicators
    st.markdown("### Global Indicators at a Glance")
    st.caption("All statistics are calculated dynamically from the 192 sovereign nations in `CLEANED_DATASET_FINAL.csv`.")

    # Row 1 of KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi(
            title="Total Countries",
            value=f"{kpis['total_countries']}",
            subtitle="Sovereign nations analyzed globally",
            tag="100% Complete"
        )
    with col2:
        render_kpi(
            title="Avg Life Expectancy",
            value=f"{kpis['avg_life_exp']:.2f} yrs",
            subtitle=f"Global median: {kpis['median_life_exp']:.2f} yrs",
            tag="Combined M/F"
        )
    with col3:
        render_kpi(
            title="Gender Longevity Gap",
            value=f"+{kpis['avg_gender_gap']:.2f} yrs",
            subtitle=f"Female: {kpis['female_avg_life_exp']:.1f}y | Male: {kpis['male_avg_life_exp']:.1f}y",
            tag="Female Advantage"
        )

    # Row 2 of KPIs
    col4, col5, col6 = st.columns(3)
    with col4:
        render_kpi(
            title="Highest Life Expectancy",
            value=f"{kpis['highest_val']:.1f} yrs",
            subtitle=f"{kpis['highest_country']} ({kpis['highest_region']})",
            tag="Global Leader"
        )
    with col5:
        render_kpi(
            title="Lowest Life Expectancy",
            value=f"{kpis['lowest_val']:.1f} yrs",
            subtitle=f"{kpis['lowest_country']} ({kpis['lowest_region']})",
            tag="Global Lowest"
        )
    with col6:
        render_kpi(
            title="Average GDP per Capita",
            value=f"${kpis['avg_gdp']:,.0f}",
            subtitle=f"Global median: ${kpis['median_gdp']:,.0f}",
            tag="Nominal US$"
        )

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # Project Introduction & Empirical Case
    intro_col1, intro_col2 = st.columns([3, 2])
    with intro_col1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>📖 The Project Background & Research Premise</h3>
                <p>
                    Life expectancy at birth is one of the most universally recognized barometers of national 
                    welfare and human development. Conventional wisdom posits that economic prosperity directly 
                    translates into longer, healthier lives: richer countries can afford superior hospitals, 
                    cleaner sanitation, modern medicines, and comprehensive social safety nets.
                </p>
                <p>
                    However, a rigorous examination of country-level empirical data reveals the 
                    <strong>Global Life Expectancy Paradox</strong>:
                </p>
                <div class="paradox-callout">
                    <strong>1. Diminishing Economic Returns (The Preston Curve):</strong><br>
                    Income increases longevity rapidly at low-income thresholds, but once national income crosses 
                    moderate levels, additional wealth yields negligible gains in life expectancy.
                    <br><br>
                    <strong>2. The 20-Year Longevity Chasm at Identical Incomes:</strong><br>
                    Countries with nearly identical GDP per capita frequently diverge by 15 to 25 years in life 
                    expectancy, proving that national income alone does not determine longevity.
                </div>
                <p>
                    This dashboard interrogates the underlying drivers that explain this chasm—analyzing 
                    health systems (infant mortality, fertility), human capital (secondary education parity), 
                    digital connectivity (internet adoption), and urbanization across 192 countries.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with intro_col2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>🔬 Empirical Case: The Paradox in Action</h3>
                <p>
                    Consider countries within the mid-income bracket (<strong>GDP per capita between $5,000 and $11,000</strong>):
                </p>
                <div style="background-color: {c['callout_bg']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; border-bottom: 1px dashed {c['border']}; padding-bottom: 0.4rem;">
                        <div>
                            <strong style="color: {c['primary']};">🇨🇺 Cuba</strong><br>
                            <small style="color: {c['text_muted']};">GDP/cap: $8,822 | Infant Mort: 4.5</small>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: {c['accent']}; font-size: 1.25rem; font-weight: 800;">78.7 yrs</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; border-bottom: 1px dashed {c['border']}; padding-bottom: 0.4rem;">
                        <div>
                            <strong style="color: {c['primary']};">🇲🇻 Maldives</strong><br>
                            <small style="color: {c['text_muted']};">GDP/cap: $10,331 | Infant Mort: 6.8</small>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: {c['accent']}; font-size: 1.25rem; font-weight: 800;">78.8 yrs</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #EF4444;">🇬🇶 Equatorial Guinea</strong><br>
                            <small style="color: {c['text_muted']};">GDP/cap: $7,566 | Infant Mort: 66.1</small>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #EF4444; font-size: 1.25rem; font-weight: 800;">58.4 yrs</span>
                        </div>
                    </div>
                </div>
                <p style="font-size: 0.88rem; color: {c['text_muted']}; margin: 0;">
                    💡 <em>Equatorial Guinea and Cuba have nearly identical per-capita income, yet citizens in Cuba 
                    enjoy an average lifespan over 20 years longer, primarily driven by primary health infrastructure 
                    reflected in infant mortality rates.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Data Governance & Methodology Documentation
    st.markdown("### Data Governance & Empirical Integrity")
    gov_col1, gov_col2 = st.columns(2)

    with gov_col1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>📐 Documented Derived Metric Formulas</h3>
                <p>
                    In strict adherence to project guidelines, derived metrics are calculated dynamically without 
                    altering the underlying CSV dataset on disk:
                </p>
                <ul style="color: {c['text_main']}; font-size: 0.92rem; line-height: 1.8;">
                    <li>
                        <strong>Average Life Expectancy:</strong><br>
                        <code>avg_life_expectancy = (life_expectancy_male + life_expectancy_female) / 2.0</code><br>
                        <em>Calculates the unweighted arithmetic mean of sex-disaggregated life expectancies.</em>
                    </li>
                    <li>
                        <strong>Female-to-Male Longevity Advantage:</strong><br>
                        <code>female_male_gap = life_expectancy_female - life_expectancy_male</code><br>
                        <em>Measures biological and behavioral longevity divergence across sexes.</em>
                    </li>
                    <li>
                        <strong>ISO-2 Verification & Namibia Correction:</strong><br>
                        <em>Corrected Namibia's ISO-2 country code from <code>AD</code> to <code>NA</code> in-memory, ensuring 100% collision-free geospatial mapping.</em>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with gov_col2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>🛡️ Dataset Boundaries & Scope Constraints</h3>
                <p>
                    To preserve academic rigor, the dashboard enforces strict data boundaries:
                </p>
                <ul style="color: {c['text_main']}; font-size: 0.92rem; line-height: 1.8;">
                    <li>
                        <strong>No Synthetic Immunisation Data:</strong><br>
                        <em>Vaccination indicators (e.g., Polio, DTP, HepB) are not present in the empirical dataset and are deliberately excluded rather than synthesized.</em>
                    </li>
                    <li>
                        <strong>Cross-Sectional Global Snapshot:</strong><br>
                        <em>The dataset captures cross-sectional country profiles rather than multi-year panel series; longitudinal trendlines are omitted in favor of robust cross-country comparisons.</em>
                    </li>
                    <li>
                        <strong>Preservation of Original Artifacts:</strong><br>
                        <em>All Week 1–7 milestone files and reports remain untouched and intact.</em>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Roadmap Preview
    st.markdown("### Dashboard Implementation Roadmap")
    roadmap_col1, roadmap_col2, roadmap_col3, roadmap_col4 = st.columns(4)

    with roadmap_col1:
        st.markdown(
            f"""
            <div style="background: {c['card_bg']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 1.2rem; border-top: 3px solid {c['accent']}; height: 100%;">
                <strong style="color: {c['primary']};">Stages 2 & 3: Wealth & Health Drivers</strong> <span style="background: {c['accent_bg']}; color: {c['accent']}; font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px; margin-left: 0.4rem;">COMPLETED</span>
                <p style="font-size: 0.85rem; color: {c['text_muted']}; margin: 0.5rem 0 0 0;">
                    Interactive Preston Curve, global choropleth map, and demographic drivers of longevity.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with roadmap_col2:
        st.markdown(
            f"""
            <div style="background: {c['card_bg']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 1.2rem; border-top: 3px solid {c['primary']}; height: 100%;">
                <strong style="color: {c['primary']};">Stages 4 & 5: Gender & Regional Divides</strong> <span style="background: {c['accent_bg']}; color: {c['accent']}; font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px; margin-left: 0.4rem;">COMPLETED</span>
                <p style="font-size: 0.85rem; color: {c['text_muted']}; margin: 0.5rem 0 0 0;">
                    Male vs female longevity parity, secondary education regressions, and regional rankings.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with roadmap_col3:
        st.markdown(
            f"""
            <div style="background: {c['card_bg']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 1.2rem; border-top: 3px solid {c['accent']}; height: 100%;">
                <strong style="color: {c['primary']};">Stages 6 & 7: Comparator & Explorer</strong> <span style="background: {c['accent_bg']}; color: {c['accent']}; font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px; margin-left: 0.4rem;">COMPLETED</span>
                <p style="font-size: 0.85rem; color: {c['text_muted']}; margin: 0.5rem 0 0 0;">
                    Head-to-head 2–4 country comparator, radar profiles, quality audit, and custom visualization builder.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with roadmap_col4:
        st.markdown(
            f"""
            <div style="background: {c['card_bg']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 1.2rem; border-top: 3px solid {c['primary']}; height: 100%;">
                <strong style="color: {c['primary']};">Stage 8: Final Insights & Synthesis</strong> <span style="background: {c['accent_bg']}; color: {c['accent']}; font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px; margin-left: 0.4rem;">COMPLETED</span>
                <p style="font-size: 0.85rem; color: {c['text_muted']}; margin: 0.5rem 0 0 0;">
                    Top 10 discoveries, Preston Curve explanation, country success stories, policy synthesis, and CSV export.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# 5. MAIN PAGE ROUTER
# -----------------------------------------------------------------------------
if not data_loaded_successfully:
    st.error(f"Failed to load the primary dataset. Details: {error_msg}")
    st.stop()

if selected_page == "🏠 Overview & Project Scope":
    render_overview_page()

elif selected_page == "📈 The Wealth Paradox (Stage 2)":
    render_stage2_page(df)

elif selected_page == "🩺 Health & Demographic Drivers (Stage 3)":
    render_stage3_page(df)

elif selected_page == "🎓 Gender Gap & Education (Stage 4)":
    render_stage4_page(df)

elif selected_page == "🗺️ Regional Disparities (Stage 5)":
    render_stage5_page(df)

elif selected_page == "⚖️ Paradox Country Comparator (Stage 6)":
    render_stage6_page(df)

elif selected_page == "🔍 Dataset Explorer (Stage 7)":
    render_stage7_page(df)

elif selected_page == "🎯 Final Insights & Conclusions (Stage 8)":
    render_stage8_page(df)

else:
    # Future stage placeholder
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <span class="badge" style="background-color: rgba(15, 43, 72, 0.1); color: {NAVY_PRIMARY}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; border: 1px solid {BORDER_COLOR}; text-transform: uppercase;">
                UPCOMING DELIVERABLE
            </span>
            <h1 style="color: {NAVY_PRIMARY}; font-size: 2rem; font-weight: 800; margin: 0.5rem 0 0.3rem 0;">
                {selected_page}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info(
        f"**Section Scheduled for Subsequent Implementation:**\n\n"
        f"You selected **{selected_page}**.\n\n"
        f"Currently active deliverables:\n"
        f"- **🏠 Overview & Project Scope** (Stage 1)\n"
        f"- **📈 The Wealth Paradox (Stage 2)** (Stage 2)\n"
        f"- **🩺 Health & Demographic Drivers (Stage 3)** (Stage 3)\n"
        f"- **🎓 Gender Gap & Education (Stage 4)** (Stage 4)\n"
        f"- **🗺️ Regional Disparities (Stage 5)** (Stage 5)\n"
        f"- **⚖️ Paradox Country Comparator (Stage 6)** (Stage 6)\n"
        f"- **🔍 Dataset Explorer (Stage 7)** (Stage 7)\n\n"
        f"Please use the sidebar navigation to explore the active sections."
    )
