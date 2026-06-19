"""
src/normalize.py — Team Name Normalization

The single most critical preprocessing module. Mismatched team names between
historical data and fixture lists silently destroy feature vectors — a team with
no historical data gets zeroed features and produces garbage predictions.
This module enforces canonical names everywhere.

CONTRACT: ALL team name strings entering the system must pass through
normalize_team_name() before being used in any DataFrame operation,
model input, or UI display.
"""

import pandas as pd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEAM NAME MAP: variant → canonical
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEAM_NAME_MAP = {
    # North America
    "United States": "USA",
    "United States of America": "USA",
    "U.S.A.": "USA",
    "U.S.": "USA",

    # Asia
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Kyrgyz Republic": "Kyrgyzstan",
    "UAE": "United Arab Emirates",

    # Africa
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",

    # Europe
    "Czech Republic": "Czechia",
    "FYR Macedonia": "North Macedonia",
    "Republic of Ireland": "Ireland",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herz.": "Bosnia and Herzegovina",
    "Northern Ireland": "Northern Ireland",  # keep as-is, distinct from Ireland
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",

    # Caribbean / Americas
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "St. Kitts & Nevis": "Saint Kitts and Nevis",
    "St. Vincent & the Grenadines": "Saint Vincent and the Grenadines",
    "St. Lucia": "Saint Lucia",
    "Curacao": "Curaçao",

    # Additional variants found in historical data
    "China": "China",
    "eSwatini": "Eswatini",
    "Swaziland": "Eswatini",
    "Burma": "Myanmar",
    "Zaire": "DR Congo",
    "Dahomey": "Benin",
    "Upper Volta": "Burkina Faso",
    "Rhodesia": "Zimbabwe",
    "Western Samoa": "Samoa",
    "Dutch East Indies": "Indonesia",
    "Ceylon": "Sri Lanka",
    "Tanganyika": "Tanzania",
    "South Yemen": "Yemen",
    "North Yemen": "Yemen",
    "German DR": "Germany",
    "West Germany": "Germany",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czechia",
}


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name to its canonical form.
    
    Args:
        name: Raw team name string (may be None/NaN).
    
    Returns:
        Canonical team name string.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "Unknown"
    stripped = str(name).strip()
    return TEAM_NAME_MAP.get(stripped, stripped)


def normalize_dataframe(
    df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """
    Normalize team names in both home and away columns of a DataFrame.
    
    Args:
        df: DataFrame with team name columns.
        home_col: Name of the home team column.
        away_col: Name of the away team column.
    
    Returns:
        DataFrame with normalized team names (modified in place and returned).
    """
    df[home_col] = df[home_col].apply(normalize_team_name)
    df[away_col] = df[away_col].apply(normalize_team_name)
    return df


def get_all_wc2026_teams(fixtures_df: pd.DataFrame) -> list:
    """
    Extract unique canonical team names from WC 2026 fixtures.
    Excludes placeholder strings (containing "Winner", "Runner-up", "TBD", "3rd").
    
    Args:
        fixtures_df: DataFrame of WC 2026 fixtures (already normalized).
    
    Returns:
        Sorted list of real team name strings.
    """
    home_teams = set(fixtures_df["home_team"])
    away_teams = set(fixtures_df["away_team"])
    all_teams = home_teams | away_teams
    
    placeholder_keywords = ["Winner", "Runner-up", "TBD", "3rd", "Loser", "Group"]
    real_teams = {
        t for t in all_teams
        if not any(kw in str(t) for kw in placeholder_keywords)
    }
    return sorted(real_teams)
