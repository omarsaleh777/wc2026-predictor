"""
update_data.py — Dynamic Update Pipeline

Accept a completed WC 2026 result -> update standings -> retrain models.
Callable as an importable function from app.py (NOT as a subprocess).
"""

import datetime
import os
import sys

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bracket_logic import rebuild_all_standings, update_standings
from src.normalize import normalize_team_name
from src.etl import run_etl
from src.train import run_training
from config import NEW_RESULTS_PATH, STANDINGS_PATH


def write_all_results(list_of_result_dicts) -> None:
    """
    Overwrite new_results.csv entirely with the provided list of matches.
    This allows the UI data_editor to declaratively control all custom/catchup matches.
    """
    for idx, row in enumerate(list_of_result_dicts):
        home = normalize_team_name(row["home_team"])
        away = normalize_team_name(row["away_team"])

        if home == away:
            raise ValueError(f"Row {idx}: home and away teams are identical: '{home}'")
        if int(row["home_score"]) < 0 or int(row["away_score"]) < 0:
            raise ValueError(f"Row {idx}: scores cannot be negative ({home} {row['home_score']}–{row['away_score']} {away})")

    new_rows = []
    for row in list_of_result_dicts:
        new_rows.append({
            "date"       : row.get("date", datetime.date.today().isoformat()),
            "home_team"  : normalize_team_name(row["home_team"]),
            "away_team"  : normalize_team_name(row["away_team"]),
            "home_score" : int(row["home_score"]),
            "away_score" : int(row["away_score"]),
            "tournament" : "FIFA World Cup",
            "city"       : "",
            "country"    : "",
            "neutral"    : True,
            "group"      : row.get("group"),
            "stage"      : row.get("stage", "Group Stage")
        })

    new_df = pd.DataFrame(new_rows)
    os.makedirs(os.path.dirname(NEW_RESULTS_PATH), exist_ok=True)
    if new_df.empty:
        pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral", "group", "stage"]).to_csv(NEW_RESULTS_PATH, index=False)
    else:
        new_df.to_csv(NEW_RESULTS_PATH, index=False)
    print(f"Overwrote {NEW_RESULTS_PATH} with {len(new_rows)} result(s).")


def run_full_update(list_of_result_dicts, fixtures_df) -> bool:
    """
    Orchestrate the complete batch update pipeline.
    ETL and model training are called EXACTLY ONCE regardless of batch size.
    This is the primary performance optimization for batch submission.
    """
    write_all_results(list_of_result_dicts)

    completed_df = get_completed_results()
    rebuild_all_standings(fixtures_df, completed_df)
    print(f"Standings rebuilt after batch of {len(list_of_result_dicts)} result(s).")

    try:
        run_etl()
        print("ETL complete.")
        run_training()
        print("Models retrained.")
    except Exception as e:
        print(e)
        return False

    print(f"Full update complete. {len(list_of_result_dicts)} result(s) processed.")
    return True


def delete_result(home_team, away_team, fixtures_df) -> bool:
    """
    Remove a previously submitted result from new_results.csv, then
    trigger a full standings rebuild and model retrain.
    """
    home = normalize_team_name(home_team)
    away = normalize_team_name(away_team)

    if not os.path.exists(NEW_RESULTS_PATH) or os.path.getsize(NEW_RESULTS_PATH) == 0:
        raise FileNotFoundError("No results file found. Nothing to delete.")

    df = pd.read_csv(NEW_RESULTS_PATH)

    mask = (df["home_team"] == home) & (df["away_team"] == away)

    if mask.sum() == 0:
        raise ValueError(
            f"No submitted result found for: {home} vs {away}. "
            f"Check team names and try again."
        )

    if mask.sum() > 1:
        print(f"WARNING: {mask.sum()} rows matched for {home} vs {away}. Deleting all matches.")

    df_clean = df[~mask].reset_index(drop=True)
    df_clean.to_csv(NEW_RESULTS_PATH, index=False)
    print(f"Deleted result: {home} vs {away}. Remaining results: {len(df_clean)}.")

    completed_df = get_completed_results()
    rebuild_all_standings(fixtures_df, completed_df)
    print("Standings rebuilt after deletion.")

    try:
        run_etl()
        run_training()
        print("Models retrained after deletion.")
        return True
    except Exception as e:
        print(f"Retrain failed after deletion: {e}")
        return False


def edit_result(home_team, away_team, new_home_score, new_away_score, fixtures_df) -> bool:
    """
    Correct an existing submitted result (wrong scoreline entered).
    Finds the matching row by team names, updates ONLY the score columns,
    then triggers a standings rebuild and retrain.
    """
    if int(new_home_score) < 0 or int(new_away_score) < 0:
        raise ValueError("Corrected scores cannot be negative.")

    home = normalize_team_name(home_team)
    away = normalize_team_name(away_team)

    if not os.path.exists(NEW_RESULTS_PATH):
        raise FileNotFoundError("No results file found. Nothing to edit.")

    df = pd.read_csv(NEW_RESULTS_PATH)

    mask = (df["home_team"] == home) & (df["away_team"] == away)

    if mask.sum() == 0:
        raise ValueError(
            f"No submitted result found for: {home} vs {away}. Cannot edit a non-existent entry."
        )

    df.loc[mask, "home_score"] = int(new_home_score)
    df.loc[mask, "away_score"] = int(new_away_score)
    df.to_csv(NEW_RESULTS_PATH, index=False)
    print(f"Updated: {home} {new_home_score}–{new_away_score} {away} (was previously different score).")

    completed_df = get_completed_results()
    rebuild_all_standings(fixtures_df, completed_df)
    print("Standings rebuilt after edit.")

    try:
        run_etl()
        run_training()
        print("Models retrained after edit.")
        return True
    except Exception as e:
        print(f"Retrain failed after edit: {e}")
        return False


def get_completed_results():
    """
    Load all submitted WC 2026 results.
    
    Returns:
        DataFrame of submitted results, or empty DataFrame if none exist.
    """
    if os.path.exists(NEW_RESULTS_PATH):
        return pd.read_csv(NEW_RESULTS_PATH)
    else:
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_score",
                     "away_score", "group", "stage"]
        )


def append_batch_results(list_of_match_dicts) -> None:
    """
    Append one or more results to new_results.csv.
    Assumes caller has already filtered out duplicates.
    """
    new_rows = []
    for idx, row in enumerate(list_of_match_dicts):
        home = normalize_team_name(row["home_team"])
        away = normalize_team_name(row["away_team"])
        h_score = int(row["home_score"])
        a_score = int(row["away_score"])
        
        if h_score < 0 or a_score < 0:
            raise ValueError(f"Row {idx}: scores cannot be negative ({home} {h_score}-{a_score} {away})")
            
        new_rows.append({
            "date": row.get("date", datetime.date.today().isoformat()),
            "home_team": home,
            "away_team": away,
            "home_score": h_score,
            "away_score": a_score,
            "tournament": row.get("tournament", "FIFA World Cup"),
            "city": row.get("city", ""),
            "country": row.get("country", ""),
            "neutral": row.get("neutral", True),
            "group": row.get("group"),
            "stage": row.get("stage", "Group Stage")
        })
        
    new_df = pd.DataFrame(new_rows)
    
    if os.path.exists(NEW_RESULTS_PATH):
        existing_df = pd.read_csv(NEW_RESULTS_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(NEW_RESULTS_PATH), exist_ok=True)
        combined_df = new_df
        
    combined_df.to_csv(NEW_RESULTS_PATH, index=False)
    print(f"Appended {len(new_rows)} batch result(s).")

def run_batch_update(list_of_match_dicts) -> bool:
    """
    Orchestrate batch update for Catch-Up Tab.
    Calls update_standings directly for each group stage match instead of rebuild_all_standings.
    """
    if not list_of_match_dicts:
        return True
        
    append_batch_results(list_of_match_dicts)
    
    for row in list_of_match_dicts:
        if row.get("stage") == "Group Stage" and row.get("group"):
            update_standings(
                normalize_team_name(row["home_team"]),
                normalize_team_name(row["away_team"]),
                int(row["home_score"]),
                int(row["away_score"]),
                row["group"]
            )
            
    try:
        print("Running ETL for batch...")
        run_etl()
        print("Running Training for batch...")
        run_training()
    except Exception as e:
        print(f"Batch update failed during ETL/Train: {e}")
        return False
        
    return True

