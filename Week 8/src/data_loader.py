"""
Data Loading & Processing Module
=================================
Project: Global Life Expectancy Paradox
Dataset: Week 3/CLEANED_DATASET_FINAL.csv

Key Data Governance Rules:
1. Loads the primary cleaned dataset using relative, robust pathing.
2. Preserves the original file on disk; all corrections and feature additions are applied in-memory.
3. Does NOT synthesize or invent immunisation or year-wise data.
4. Corrects authoritative ISO2 country codes (e.g., Namibia: 'NA', not 'AD').
5. Dynamically derives Average Life Expectancy and Gender Longevity Gap.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Define relative path to the primary cleaned dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Week 3" / "CLEANED_DATASET_FINAL.csv"

# ISO-2 to ISO-3 authoritative mapping for standard choropleth mapping
ISO2_TO_ISO3: Dict[str, str] = {
    "AF": "AFG", "AL": "ALB", "DZ": "DZA", "AD": "AND", "AO": "AGO", "AG": "ATG",
    "AR": "ARG", "AM": "ARM", "AU": "AUS", "AT": "AUT", "AZ": "AZE", "BS": "BHS",
    "BH": "BHR", "BD": "BGD", "BB": "BRB", "BY": "BLR", "BE": "BEL", "BZ": "BLZ",
    "BJ": "BEN", "BT": "BTN", "BO": "BOL", "BA": "BIH", "BW": "BWA", "BR": "BRA",
    "BN": "BRN", "BG": "BGR", "BF": "BFA", "BI": "BDI", "CV": "CPV", "KH": "KHM",
    "CM": "CMR", "CA": "CAN", "CF": "CAF", "TD": "TCD", "CL": "CHL", "CN": "CHN",
    "CO": "COL", "KM": "COM", "CG": "COG", "CR": "CRI", "CI": "CIV", "HR": "HRV",
    "CU": "CUB", "CY": "CYP", "CZ": "CZE", "CD": "COD", "DK": "DNK", "DJ": "DJI",
    "DM": "DMA", "DO": "DOM", "EC": "ECU", "EG": "EGY", "SV": "SLV", "GQ": "GNQ",
    "ER": "ERI", "EE": "EST", "SZ": "SWZ", "ET": "ETH", "FJ": "FJI", "FI": "FIN",
    "FR": "FRA", "GA": "GAB", "GM": "GMB", "GE": "GEO", "DE": "DEU", "GH": "GHA",
    "GR": "GRC", "GD": "GRD", "GT": "GTM", "GN": "GIN", "GW": "GNB", "GY": "GUY",
    "HT": "HTI", "HN": "HND", "HU": "HUN", "IS": "ISL", "IN": "IND", "ID": "IDN",
    "IR": "IRN", "IQ": "IRQ", "IE": "IRL", "IL": "ISR", "IT": "ITA", "JM": "JAM",
    "JP": "JPN", "JO": "JOR", "KZ": "KAZ", "KE": "KEN", "KI": "KIR", "KW": "KWT",
    "KG": "KGZ", "LA": "LAO", "LV": "LVA", "LB": "LBN", "LS": "LSO", "LR": "LBR",
    "LY": "LBY", "LT": "LTU", "LU": "LUX", "MG": "MDG", "MW": "MWI", "MY": "MYS",
    "MV": "MDV", "ML": "MLI", "MT": "MLT", "MH": "MHL", "MR": "MRT", "MU": "MUS",
    "MX": "MEX", "FM": "FSM", "MD": "MDA", "MC": "MCO", "MN": "MNG", "ME": "MNE",
    "MA": "MAR", "MZ": "MOZ", "MM": "MMR", "NA": "NAM", "NR": "NRU", "NP": "NPL",
    "NL": "NLD", "NZ": "NZL", "NI": "NIC", "NE": "NER", "NG": "NGA", "MK": "MKD",
    "NO": "NOR", "OM": "OMN", "PK": "PAK", "PW": "PLW", "PA": "PAN", "PG": "PNG",
    "PY": "PRY", "PE": "PER", "PH": "PHL", "PL": "POL", "PT": "PRT", "QA": "QAT",
    "KR": "KOR", "KP": "PRK", "RO": "ROU", "RU": "RUS", "RW": "RWA", "KN": "KNA",
    "LC": "LCA", "LI": "LIE", "VC": "VCT", "WS": "WSM", "SM": "SMR", "ST": "STP",
    "SA": "SAU", "SN": "SEN", "RS": "SRB", "SC": "SYC", "SL": "SLE", "SG": "SGP",
    "SK": "SVK", "SI": "SVN", "SB": "SLB", "SO": "SOM", "ZA": "ZAF", "SS": "SSD",
    "ES": "ESP", "LK": "LKA", "SD": "SDN", "SR": "SUR", "SE": "SWE", "CH": "CHE",
    "SY": "SYR", "TJ": "TJK", "TZ": "TZA", "TH": "THA", "TL": "TLS", "TG": "TGO",
    "TO": "TON", "TT": "TTO", "TN": "TUN", "TR": "TUR", "TM": "TKM", "TV": "TUV",
    "UG": "UGA", "UA": "UKR", "AE": "ARE", "GB": "GBR", "US": "USA", "UY": "URY",
    "UZ": "UZB", "VA": "VAT", "VU": "VUT", "VE": "VEN", "VN": "VNM", "YE": "YEM",
    "ZM": "ZMB", "ZW": "ZWE"
}


def get_data_filepath() -> Path:
    """Return the resolved Path to the primary dataset."""
    return DATA_PATH


def load_dataset() -> pd.DataFrame:
    """
    Load and preprocess the cleaned dataset for the dashboard.
    
    Processing Steps:
    1. Read CSV from Week 3/CLEANED_DATASET_FINAL.csv.
    2. Correct Namibia ISO-2 code from 'AD' (mode imputation artifact) to 'NA'.
    3. Map ISO-2 codes to ISO-3 codes for universal geospatial choropleth support.
    4. Compute Average Life Expectancy:
       avg_life_expectancy = (life_expectancy_male + life_expectancy_female) / 2.0
    5. Compute Female-to-Male Longevity Gap:
       female_male_gap = life_expectancy_female - life_expectancy_male
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Primary cleaned dataset not found at expected location: {DATA_PATH}"
        )
    
    # Read the dataset (keep_default_na=False ensures string 'NA' is not converted to NaN)
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    
    # Rule 6: Authoritative ISO2 correction for Namibia (replaces imputed 'AD' with 'NA')
    namibia_mask = df["name"].str.strip() == "Namibia"
    if namibia_mask.any():
        df.loc[namibia_mask, "iso2"] = "NA"
    
    # Map ISO-3 country codes for standard Plotly choropleth maps
    df["iso3"] = df["iso2"].map(ISO2_TO_ISO3).fillna("")
    
    # Rule 5: Dynamic calculation of Average Life Expectancy
    # Documented calculation: arithmetic mean of male and female longevity
    df["avg_life_expectancy"] = (df["life_expectancy_male"] + df["life_expectancy_female"]) / 2.0
    
    # Longevity Gap: Female longevity minus male longevity
    df["female_male_gap"] = df["life_expectancy_female"] - df["life_expectancy_male"]
    
    return df


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all high-level Key Performance Indicators dynamically from the dataset.
    """
    total_countries = int(len(df))
    avg_life_exp = float(df["avg_life_expectancy"].mean())
    median_life_exp = float(df["avg_life_expectancy"].median())
    
    # Highest life expectancy country
    max_idx = df["avg_life_expectancy"].idxmax()
    highest_country = str(df.loc[max_idx, "name"])
    highest_val = float(df.loc[max_idx, "avg_life_expectancy"])
    highest_region = str(df.loc[max_idx, "region"])
    
    # Lowest life expectancy country
    min_idx = df["avg_life_expectancy"].idxmin()
    lowest_country = str(df.loc[min_idx, "name"])
    lowest_val = float(df.loc[min_idx, "avg_life_expectancy"])
    lowest_region = str(df.loc[min_idx, "region"])
    
    # Economic indicators
    avg_gdp = float(df["gdp_per_capita"].mean())
    median_gdp = float(df["gdp_per_capita"].median())
    
    # Gender longevity gap
    avg_gender_gap = float(df["female_male_gap"].mean())
    male_avg_life_exp = float(df["life_expectancy_male"].mean())
    female_avg_life_exp = float(df["life_expectancy_female"].mean())
    
    return {
        "total_countries": total_countries,
        "avg_life_exp": avg_life_exp,
        "median_life_exp": median_life_exp,
        "highest_country": highest_country,
        "highest_val": highest_val,
        "highest_region": highest_region,
        "lowest_country": lowest_country,
        "lowest_val": lowest_val,
        "lowest_region": lowest_region,
        "avg_gdp": avg_gdp,
        "median_gdp": median_gdp,
        "avg_gender_gap": avg_gender_gap,
        "male_avg_life_exp": male_avg_life_exp,
        "female_avg_life_exp": female_avg_life_exp,
    }

