"""
src/etl.py — Data Pipeline

Transform raw CSVs into a clean, feature-engineered training dataset.
Callable both as a standalone script AND as an importable function (run_etl()).
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAW_RESULTS_PATH,
    RAW_FIXTURES_PATH,
    PROCESSED_PATH,
    NEW_RESULTS_PATH,
    FEATURE_COLUMNS,
    LOOKBACK_MATCHES,
    MIN_MATCH_YEAR,
    OUTCOME_ENCODING,
    FIXTURES_COLUMN_MAP,
    FIXTURES_STAGE_MAP,
    TEAM_TO_GROUP,
)
from src.normalize import normalize_dataframe, normalize_team_name, get_all_wc2026_teams


def load_fixtures():
    """Load and normalize WC 2026 fixtures, adapting column names as needed."""
    fix_df = pd.read_csv(RAW_FIXTURES_PATH)

    # Rename columns to match our schema
    fix_df = fix_df.rename(columns=FIXTURES_COLUMN_MAP)

    # Map stage values to canonical names
    if "stage" in fix_df.columns:
        fix_df["stage"] = fix_df["stage"].map(
            lambda x: FIXTURES_STAGE_MAP.get(x, x)
        )

    # Derive group from team lookup if 'group' column doesn't exist
    if "group" not in fix_df.columns:
        fix_df = normalize_dataframe(fix_df)
        fix_df["group"] = fix_df["home_team"].map(
            lambda t: TEAM_TO_GROUP.get(t, None)
        )
    else:
        fix_df = normalize_dataframe(fix_df)

    return fix_df


def calculate_defensive_stats(df, window=5):
    """Calculate rolling defensive stats by melting the wide dataframe."""
    
    # 1. Temporarily melt the dataframe to track chronological history
    home_df = df[['date', 'home_team', 'away_score']].rename(
        columns={'home_team': 'team', 'away_score': 'goals_conceded'}
    )
    away_df = df[['date', 'away_team', 'home_score']].rename(
        columns={'away_team': 'team', 'home_score': 'goals_conceded'}
    )
    
    # Combine and sort chronologically
    melted = pd.concat([home_df, away_df]).sort_values('date').reset_index(drop=True)
    
    # Identify clean sheets
    melted['clean_sheet'] = (melted['goals_conceded'] == 0).astype(int)
    
    # 2. Calculate rolling metrics using shift(1) to strictly prevent data leakage
    melted['goals_conceded_per_match'] = melted.groupby('team')['goals_conceded'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    melted['clean_sheet_ratio'] = melted.groupby('team')['clean_sheet'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    # 3. Merge back to the original wide df for Home teams
    df = df.merge(
        melted[['date', 'team', 'goals_conceded_per_match', 'clean_sheet_ratio']],
        left_on=['date', 'home_team'],
        right_on=['date', 'team'],
        how='left'
    ).rename(columns={
        'goals_conceded_per_match': 'home_goals_conceded_per_match',
        'clean_sheet_ratio': 'home_clean_sheet_ratio'
    }).drop(columns=['team'])
    
    # Merge back to the original wide df for Away teams
    df = df.merge(
        melted[['date', 'team', 'goals_conceded_per_match', 'clean_sheet_ratio']],
        left_on=['date', 'away_team'],
        right_on=['date', 'team'],
        how='left'
    ).rename(columns={
        'goals_conceded_per_match': 'away_goals_conceded_per_match',
        'clean_sheet_ratio': 'away_clean_sheet_ratio'
    }).drop(columns=['team'])
    
    # 4. Fill NaNs with the respective column means
    def_cols = [
        'home_goals_conceded_per_match', 'home_clean_sheet_ratio', 
        'away_goals_conceded_per_match', 'away_clean_sheet_ratio'
    ]
    for col in def_cols:
        df[col] = df[col].fillna(df[col].mean())
        
    return df


def add_draw_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features that capture match competitiveness.
    Tighter Elo gap = higher draw probability.
    """
    # Absolute Elo gap — lower = more evenly matched
    df['elo_gap'] = abs(df['home_elo_score'] - df['away_elo_score'])

    # Normalized competitiveness score (1.0 = perfectly equal, 0.0 = huge mismatch)
    max_gap = df['elo_gap'].max()
    if max_gap == 0: max_gap = 1
    df['competitiveness'] = 1 - (df['elo_gap'] / max_gap)

    # Defensive parity — both teams similarly hard to score against
    df['defensive_parity'] = 1 - abs(
        df['home_clean_sheet_ratio'] - df['away_clean_sheet_ratio']
    )

    # Combined draw signal
    df['draw_likelihood'] = (df['competitiveness'] + df['defensive_parity']) / 2

    return df


def run_etl():
    """
    Main ETL pipeline: Load → clean → normalize → feature engineer → save.
    """
    # ──────────────────────────────────────
    # 1. Data Loading
    # ──────────────────────────────────────
    print("Loading raw data...")
    historical_df = pd.read_csv(RAW_RESULTS_PATH)
    historical_df["date"] = pd.to_datetime(historical_df["date"])
    print(f"  Historical results: {historical_df.shape}")

    # Load fixtures for reference
    fixtures_df = load_fixtures()
    print(f"  WC 2026 fixtures: {fixtures_df.shape}")
    print(f"  Fixture columns: {list(fixtures_df.columns)}")

    # Merge new results if they exist
    if os.path.exists(NEW_RESULTS_PATH):
        new_df = pd.read_csv(NEW_RESULTS_PATH)
        new_df["date"] = pd.to_datetime(new_df["date"])
        historical_df = pd.concat([historical_df, new_df], ignore_index=True)
        historical_df = historical_df.drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="last"
        )
        historical_df = historical_df.sort_values("date").reset_index(drop=True)
        print(f"  After merging new results: {historical_df.shape}")

    # ──────────────────────────────────────
    # 2. Cleaning
    # ──────────────────────────────────────
    print("Cleaning and normalizing...")
    historical_df = normalize_dataframe(historical_df)

    # Filter by year
    historical_df = historical_df[historical_df["date"].dt.year >= MIN_MATCH_YEAR]

    # Drop rows with missing scores
    historical_df = historical_df.dropna(subset=["home_score", "away_score"])

    # Cast scores to int
    historical_df["home_score"] = historical_df["home_score"].astype(int)
    historical_df["away_score"] = historical_df["away_score"].astype(int)

    # Sort chronologically (CRITICAL for rolling window correctness)
    historical_df = historical_df.sort_values("date").reset_index(drop=True)
    print(f"  After cleaning: {historical_df.shape}")

    # ──────────────────────────────────────
    # 3. Outcome Labeling
    # ──────────────────────────────────────
    conditions = [
        historical_df["home_score"] > historical_df["away_score"],
        historical_df["home_score"] < historical_df["away_score"],
    ]
    choices = ["Home Win", "Away Win"]
    historical_df["outcome"] = np.select(conditions, choices, default="Draw")
    historical_df["outcome_encoded"] = historical_df["outcome"].map(OUTCOME_ENCODING)

    # ──────────────────────────────────────
    # 4. Rolling Feature Engineering
    # ──────────────────────────────────────
    print("Engineering features (this may take a few minutes)...")

    # Apply wide format defensive stats
    historical_df = calculate_defensive_stats(historical_df, window=5)

    # Compute global averages for fallback
    global_avg_goals = historical_df[["home_score", "away_score"]].values.mean()

    # Pre-allocate feature arrays
    n = len(historical_df)
    features = {col: np.full(n, np.nan) for col in FEATURE_COLUMNS}
    
    # Carry over the vectorized defensive stats into the feature arrays
    for col in ['home_goals_conceded_per_match', 'home_clean_sheet_ratio', 'away_goals_conceded_per_match', 'away_clean_sheet_ratio']:
        if col in FEATURE_COLUMNS:
            features[col] = historical_df[col].values

    # Team match history: team → list of match dicts (chronological)
    team_history = defaultdict(list)

    # Head-to-head history: frozenset({team_a, team_b}) → list of match dicts
    h2h_history = defaultdict(list)

    for idx in range(n):
        row = historical_df.iloc[idx]
        home = row["home_team"]
        away = row["away_team"]
        h_score = row["home_score"]
        a_score = row["away_score"]

        home_hist = team_history[home]
        away_hist = team_history[away]
        h2h_key = frozenset({home, away})
        h2h_matches = h2h_history[h2h_key]

        # ── Home team features ──
        if len(home_hist) >= 3:
            recent_home = home_hist[-LOOKBACK_MATCHES:]
            features["home_avg_goals_scored"][idx] = np.mean(
                [m["goals_scored"] for m in recent_home]
            )
            features["home_avg_goals_conceded"][idx] = np.mean(
                [m["goals_conceded"] for m in recent_home]
            )
            features["home_win_rate"][idx] = np.mean(
                [1.0 if m["result"] == "win" else 0.0 for m in recent_home]
            )
            features["home_draw_rate"][idx] = np.mean(
                [1.0 if m["result"] == "draw" else 0.0 for m in recent_home]
            )
            features["home_goal_diff_avg"][idx] = np.mean(
                [m["goals_scored"] - m["goals_conceded"] for m in recent_home]
            )
        else:
            features["home_avg_goals_scored"][idx] = global_avg_goals
            features["home_avg_goals_conceded"][idx] = global_avg_goals
            features["home_win_rate"][idx] = 0.33
            features["home_draw_rate"][idx] = 0.33
            features["home_goal_diff_avg"][idx] = 0.0

        if len(home_hist) > 0:
            features["home_last_match_pts"][idx] = home_hist[-1]["pts"]
            recent_3 = home_hist[-3:]
            features["home_form_last_3"][idx] = np.mean([m["pts"] for m in recent_3]) * 3
            
            total_wins = sum(1 for m in home_hist if m["result"] == "win")
            total_draws = sum(1 for m in home_hist if m["result"] == "draw")
            total_points = sum(m["pts"] for m in home_hist)
            features["home_elo_score"][idx] = (total_wins * 3 + total_draws) / len(home_hist)
            features["home_points_per_match"][idx] = total_points / len(home_hist)
            features["home_goal_efficiency"][idx] = sum(m["goals_scored"] for m in recent_3) / len(recent_3)
        else:
            features["home_last_match_pts"][idx] = 1.0
            features["home_form_last_3"][idx] = 3.0
            features["home_elo_score"][idx] = 1.0
            features["home_points_per_match"][idx] = 1.0
            features["home_goal_efficiency"][idx] = global_avg_goals

        # ── Away team features ──
        if len(away_hist) >= 3:
            recent_away = away_hist[-LOOKBACK_MATCHES:]
            features["away_avg_goals_scored"][idx] = np.mean(
                [m["goals_scored"] for m in recent_away]
            )
            features["away_avg_goals_conceded"][idx] = np.mean(
                [m["goals_conceded"] for m in recent_away]
            )
            features["away_win_rate"][idx] = np.mean(
                [1.0 if m["result"] == "win" else 0.0 for m in recent_away]
            )
            features["away_draw_rate"][idx] = np.mean(
                [1.0 if m["result"] == "draw" else 0.0 for m in recent_away]
            )
            features["away_goal_diff_avg"][idx] = np.mean(
                [m["goals_scored"] - m["goals_conceded"] for m in recent_away]
            )
        else:
            features["away_avg_goals_scored"][idx] = global_avg_goals
            features["away_avg_goals_conceded"][idx] = global_avg_goals
            features["away_win_rate"][idx] = 0.33
            features["away_draw_rate"][idx] = 0.33
            features["away_goal_diff_avg"][idx] = 0.0

        if len(away_hist) > 0:
            features["away_last_match_pts"][idx] = away_hist[-1]["pts"]
            recent_3 = away_hist[-3:]
            features["away_form_last_3"][idx] = np.mean([m["pts"] for m in recent_3]) * 3
            
            total_wins = sum(1 for m in away_hist if m["result"] == "win")
            total_draws = sum(1 for m in away_hist if m["result"] == "draw")
            total_points = sum(m["pts"] for m in away_hist)
            features["away_elo_score"][idx] = (total_wins * 3 + total_draws) / len(away_hist)
            features["away_points_per_match"][idx] = total_points / len(away_hist)
            features["away_goal_efficiency"][idx] = sum(m["goals_scored"] for m in recent_3) / len(recent_3)
        else:
            features["away_last_match_pts"][idx] = 1.0
            features["away_form_last_3"][idx] = 3.0
            features["away_elo_score"][idx] = 1.0
            features["away_points_per_match"][idx] = 1.0
            features["away_goal_efficiency"][idx] = global_avg_goals
            
        # ── Cross features ──
        features["strength_diff"][idx] = features["home_elo_score"][idx] - features["away_elo_score"][idx]
        features["home_adj_goal_efficiency"][idx] = features["home_goal_efficiency"][idx] * features["away_elo_score"][idx]
        features["away_adj_goal_efficiency"][idx] = features["away_goal_efficiency"][idx] * features["home_elo_score"][idx]

        # ── Head-to-head features ──
        if len(h2h_matches) > 0:
            home_wins_h2h = sum(
                1 for m in h2h_matches
                if (m["home"] == home and m["h_score"] > m["a_score"])
                or (m["away"] == home and m["a_score"] > m["h_score"])
            )
            features["h2h_home_win_rate"][idx] = home_wins_h2h / len(h2h_matches)
            features["h2h_total_matches"][idx] = len(h2h_matches)
        else:
            features["h2h_home_win_rate"][idx] = 0.33  # uniform prior
            features["h2h_total_matches"][idx] = 0

        # ── Context features ──
        neutral_val = row.get("neutral", False)
        if isinstance(neutral_val, str):
            features["is_neutral_venue"][idx] = 1 if neutral_val.upper() == "TRUE" else 0
        else:
            features["is_neutral_venue"][idx] = 1 if neutral_val else 0

        tournament = str(row.get("tournament", ""))
        features["is_world_cup"][idx] = 1 if "FIFA World Cup" in tournament else 0

        # ── Update histories AFTER computing features (no leakage) ──
        # Determine result for home team perspective
        if h_score > a_score:
            home_result = "win"
            away_result = "loss"
            home_pts = 3
            away_pts = 0
        elif h_score < a_score:
            home_result = "loss"
            away_result = "win"
            home_pts = 0
            away_pts = 3
        else:
            home_result = "draw"
            away_result = "draw"
            home_pts = 1
            away_pts = 1

        team_history[home].append({
            "goals_scored": h_score,
            "goals_conceded": a_score,
            "result": home_result,
            "pts": home_pts,
        })
        team_history[away].append({
            "goals_scored": a_score,
            "goals_conceded": h_score,
            "result": away_result,
            "pts": away_pts,
        })
        h2h_history[h2h_key].append({
            "home": home,
            "away": away,
            "h_score": h_score,
            "a_score": a_score,
        })

        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{n} matches...")

    # ──────────────────────────────────────
    # 5. Assemble and Save
    # ──────────────────────────────────────
    for col in FEATURE_COLUMNS:
        historical_df[col] = features[col]
        
    # Inject Draw Signals
    historical_df = add_draw_signals(historical_df)

    # Create Symmetric Duplicate Rows to remove positional bias
    sym_df = historical_df.copy()
    sym_df["home_team"], sym_df["away_team"] = historical_df["away_team"], historical_df["home_team"]
    sym_df["home_score"], sym_df["away_score"] = historical_df["away_score"], historical_df["home_score"]
    sym_df["outcome_encoded"] = sym_df["outcome_encoded"] * -1
    
    for col in FEATURE_COLUMNS:
        if col.startswith("home_"):
            away_col = "away_" + col[5:]
            sym_df[col] = historical_df[away_col]
        elif col.startswith("away_"):
            home_col = "home_" + col[5:]
            sym_df[col] = historical_df[home_col]
            
    sym_df["h2h_home_win_rate"] = 1.0 - historical_df["h2h_home_win_rate"]
    sym_df["strength_diff"] = sym_df["strength_diff"] * -1
    
    historical_df = pd.concat([historical_df, sym_df], ignore_index=True)

    # Drop rows with NaN in any feature column
    before = len(historical_df)
    historical_df = historical_df.dropna(subset=FEATURE_COLUMNS)
    after = len(historical_df)
    if before != after:
        print(f"  Dropped {before - after} rows with NaN features.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    # Save features + labels + identifiers
    output_cols = (
        ["date", "home_team", "away_team", "home_score", "away_score",
         "tournament", "neutral", "outcome", "outcome_encoded"]
        + FEATURE_COLUMNS
    )
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in historical_df.columns]
    historical_df[output_cols].to_csv(PROCESSED_PATH, index=False)

    print(f"ETL complete. Shape: {historical_df[output_cols].shape}. Saved to {PROCESSED_PATH}")
    return historical_df[output_cols]


if __name__ == "__main__":
    run_etl()
