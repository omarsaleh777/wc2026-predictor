"""
app.py -- Streamlit UI Entry Point

World Cup 2026 Match Outcome & Score Predictor.
Three major UI zones: header, prediction panel (tabs), and sidebar for result submission.
"""

import os
import sys
import json

import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_RESULTS_PATH,
    RAW_FIXTURES_PATH,
    NEW_RESULTS_PATH,
    STANDINGS_PATH,
    CLASSIFIER_PATH,
    HOME_GOALS_PATH,
    AWAY_GOALS_PATH,
    FEATURE_COLUMNS_PATH,
    MIN_MATCH_YEAR,
    WC2026_GROUPS,
    WC2026_GROUP_TEAMS,
    TEAM_TO_GROUP,
    STAGE_ORDER,
    KNOCKOUT_STAGES,
    FIXTURES_COLUMN_MAP,
    FIXTURES_STAGE_MAP,
)
from src.normalize import normalize_team_name, normalize_dataframe, get_all_wc2026_teams
from src.bracket_logic import (
    initialize_standings,
    load_standings,
    sort_group,
    build_placeholder_map,
    resolve_fixtures,
    is_match_predictable,
)
from src.predict import predict_match, load_models
import update_data
import bracket_component
from src import bracket_builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="WC 2026 Match Predictor",
    page_icon="\U0001F3C6",
    layout="wide",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.scoreline {
    text-align: center;
    font-size: 72px;
    font-weight: 900;
    letter-spacing: 6px;
    color: #FFFFFF !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    padding: 10px 0;
}
.team-label {
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: #F8F9FA !important;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
    padding: 8px 0;
}
.section-header {
    border-bottom: 2px solid #333;
    padding-bottom: 4px;
    font-weight: 700;
}
.prob-bar {
    display: flex;
    width: 100%;
    height: 40px;
    border-radius: 12px;
    overflow: hidden;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.prob-bar .home { background: linear-gradient(135deg, #667eea, #764ba2); }
.prob-bar .draw { background: linear-gradient(135deg, #f093fb, #f5576c); }
.prob-bar .away { background: linear-gradient(135deg, #4facfe, #00f2fe); }
.prob-bar span {
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 13px;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.metric-card h3 {
    font-size: 14px;
    color: #666;
    margin-bottom: 4px;
}
.metric-card p {
    font-size: 28px;
    font-weight: 900;
    margin: 0;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.group-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 10px 10px 0 0;
    font-weight: 700;
    font-size: 16px;
    margin-top: 12px;
}
.warning-badge {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 10px 16px;
    color: #856404;
    font-size: 14px;
    margin: 8px 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Caching Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=None)
def load_raw_data():
    """Load historical results + any new WC results, normalized."""
    hist_df = pd.read_csv(RAW_RESULTS_PATH)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = normalize_dataframe(hist_df)

    if os.path.exists(NEW_RESULTS_PATH):
        new_df = pd.read_csv(NEW_RESULTS_PATH)
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df = normalize_dataframe(new_df)
        hist_df = pd.concat([hist_df, new_df], ignore_index=True)
        hist_df = hist_df.drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="last"
        )

    hist_df = hist_df[hist_df["date"].dt.year >= MIN_MATCH_YEAR]
    hist_df = hist_df.sort_values("date").reset_index(drop=True)
    return hist_df


@st.cache_data(ttl=None)
def load_fixtures_data():
    """Load and normalize WC 2026 fixtures."""
    fix_df = pd.read_csv(RAW_FIXTURES_PATH)
    fix_df = fix_df.rename(columns=FIXTURES_COLUMN_MAP)
    if "stage" in fix_df.columns:
        fix_df["stage"] = fix_df["stage"].map(lambda x: FIXTURES_STAGE_MAP.get(x, x))
    fix_df = normalize_dataframe(fix_df)
    if "group" not in fix_df.columns:
        fix_df["group"] = fix_df["home_team"].map(lambda t: TEAM_TO_GROUP.get(t, None))
    return fix_df


@st.cache_resource
def load_all_models():
    """Load all ML models. Returns None if any are missing."""
    try:
        return load_models()
    except FileNotFoundError:
        return None


def clear_all_caches():
    """Clear all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Startup Checks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fixtures_df = load_fixtures_data()

if not os.path.exists(STANDINGS_PATH):
    initialize_standings(fixtures_df)

models = load_all_models()
if models is None:
    st.error("\u26A0\uFE0F Models not trained. Run `python src/train.py` in your terminal.")
    st.info("After running training, refresh this page.")
    st.stop()

historical_df = load_raw_data()
completed_results = update_data.get_completed_results()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# \U0001F3C6")
with col_title:
    st.title("World Cup 2026 \u2014 Match Predictor")
    st.caption("Predictions powered by historical data \u00B7 Updated with live WC 2026 results")

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tabs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_predict, tab_groups, tab_bracket, tab_data, tab_catchup = st.tabs([
    "⚽ Predict",
    "📊 Group Standings",
    "🏆 Bracket",
    "📥 Data Management",
    "⚡ Live Catch-Up"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_predict:
    st.markdown("### Select a Match to Predict")

    # Get all WC 2026 teams
    all_teams = sorted(TEAM_TO_GROUP.keys())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox(
            "Home Team",
            options=all_teams,
            index=all_teams.index("Brazil") if "Brazil" in all_teams else 0,
            key="home_team_select",
        )
    with col2:
        away_options = [t for t in all_teams if t != home_team]
        default_away = "Argentina" if "Argentina" in away_options else away_options[0]
        away_team = st.selectbox(
            "Away Team",
            options=away_options,
            index=away_options.index(default_away) if default_away in away_options else 0,
            key="away_team_select",
        )

    if st.button("\U0001F52E  Predict Match Outcome", type="primary", use_container_width=True):
        with st.spinner("Crunching the numbers..."):
            result = predict_match(home_team, away_team, historical_df, models)

        # Warning banner
        if result["warning"]:
            st.markdown(
                f'<div class="warning-badge">\u26A0\uFE0F {result["warning"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Scoreline display
        col_h, col_score, col_a = st.columns([3, 4, 3])
        with col_h:
            st.markdown(
                f'<div class="team-label">{result["home_team"]}</div>',
                unsafe_allow_html=True,
            )
        with col_score:
            st.markdown(
                f'<div class="scoreline">{result["home_goals"]} - {result["away_goals"]}</div>',
                unsafe_allow_html=True,
            )
        with col_a:
            st.markdown(
                f'<div class="team-label">{result["away_team"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Probability bar
        hw = result["home_win_pct"]
        dw = result["draw_pct"]
        aw = result["away_win_pct"]

        st.markdown(
            f"""
            <div class="prob-bar">
                <span class="home" style="width:{hw}%">{hw:.1f}%</span>
                <span class="draw" style="width:{dw}%">{dw:.1f}%</span>
                <span class="away" style="width:{aw}%">{aw:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability metric cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(
                f"""<div class="metric-card">
                    <h3>{result["home_team"]} Win</h3>
                    <p>{hw:.1f}%</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""<div class="metric-card">
                    <h3>Draw</h3>
                    <p>{dw:.1f}%</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""<div class="metric-card">
                    <h3>{result["away_team"]} Win</h3>
                    <p>{aw:.1f}%</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.info(f"🧠 **AI Reasoning:** {result.get('reasoning', 'No reasoning available.')}")

        # H2H history snippet
        st.markdown("#### Recent Head-to-Head")
        h2h_mask = (
            ((historical_df["home_team"] == result["home_team"]) &
             (historical_df["away_team"] == result["away_team"])) |
            ((historical_df["home_team"] == result["away_team"]) &
             (historical_df["away_team"] == result["home_team"]))
        )
        h2h_df = historical_df[h2h_mask].tail(5)
        if len(h2h_df) > 0:
            display_df = h2h_df[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df.columns = ["Date", "Home", "Away", "H", "A", "Tournament"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No previous head-to-head matches found in the historical data.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: GROUP STANDINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_groups:
    st.markdown("### Group Stage Standings")
    st.caption("Tables update automatically when you submit match results via the sidebar.")

    standings = load_standings()

    # Display in 3-column grid
    cols_per_row = 3
    groups = WC2026_GROUPS
    for i in range(0, len(groups), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(groups):
                break
            group = groups[idx]
            with col:
                st.markdown(
                    f'<div class="group-header">Group {group}</div>',
                    unsafe_allow_html=True,
                )

                if group in standings and standings[group]:
                    sorted_teams = sort_group(standings[group])
                    table_data = []
                    for rank, (team, stats) in enumerate(sorted_teams, 1):
                        table_data.append({
                            "#": rank,
                            "Team": team,
                            "P": stats["played"],
                            "W": stats["won"],
                            "D": stats["drawn"],
                            "L": stats["lost"],
                            "GF": stats["gf"],
                            "GA": stats["ga"],
                            "GD": stats["gd"],
                            "Pts": stats["pts"],
                        })
                    st.dataframe(
                        pd.DataFrame(table_data),
                        use_container_width=True,
                        hide_index=True,
                        height=195,
                    )
                else:
                    # Show teams from config even if no standings yet
                    teams = WC2026_GROUP_TEAMS.get(group, [])
                    table_data = [
                        {"#": i+1, "Team": t, "P": 0, "W": 0, "D": 0, "L": 0,
                         "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
                        for i, t in enumerate(teams)
                    ]
                    st.dataframe(
                        pd.DataFrame(table_data),
                        use_container_width=True,
                        hide_index=True,
                        height=195,
                    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: BRACKET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_bracket:
    st.markdown("### 🏆 Knockout Stage Bracket")
    
    # 1. Build DataFrame from Standings
    if "bracket_data" not in st.session_state:
        if os.path.exists("bracket_state.json"):
            with open("bracket_state.json", "r") as f:
                st.session_state.bracket_data = json.load(f)
        else:
            standings = load_standings()
        rows = []
        for group_name, teams in standings.items():
            if teams:
                sorted_teams = sort_group(teams)
                for team_name, stats in sorted_teams:
                    rows.append({
                        "Group": group_name,
                        "Team": team_name,
                        "Pts": stats["pts"],
                        "GD": stats["gd"],
                        "GF": stats["gf"]
                    })
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            if "resolved_matchups" not in st.session_state:
                st.session_state.resolved_matchups = bracket_builder.resolve_qualified_teams(df)
            
            with st.expander("⚙️ Manual Matchup Override (Pre-Bracket Generation)", expanded=True):
                st.info("Review and manually swap teams before running the ML predictions.")
                
                # Extract all unique teams for the dropdowns
                all_teams_list = sorted(df["Team"].unique().tolist())
                
                # Configure the data editor columns to use Selectbox
                df_matchups = pd.DataFrame(st.session_state.resolved_matchups)
                edited_df = st.data_editor(
                    df_matchups, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "home": st.column_config.SelectboxColumn("Home Team", options=all_teams_list, required=True),
                        "away": st.column_config.SelectboxColumn("Away Team", options=all_teams_list, required=True),
                        "match_id": st.column_config.TextColumn("Match ID", disabled=True)
                    }
                )
                
                if st.button("Generate Bracket & Run Predictions", type="primary"):
                    st.session_state.resolved_matchups = edited_df.to_dict('records')
                    st.session_state.bracket_data = bracket_builder.generate_bracket_data(
                        st.session_state.resolved_matchups, 
                        historical_df, 
                        models
                    )
                    with open("bracket_state.json", "w") as f:
                        json.dump(st.session_state.bracket_data, f)
                    st.rerun()
        else:
            st.warning("Group stage not complete. Using fallback seeding.")
            # Fallback
            st.session_state.bracket_data = bracket_component.build_initial_bracket()

    if "bracket_data" in st.session_state:
        if st.button("Reset Bracket (Clear Saves)"):
            if os.path.exists("bracket_state.json"):
                os.remove("bracket_state.json")
            del st.session_state.bracket_data
            st.rerun()
            
        bracket_component.render_data_entry_ui(st.session_state.bracket_data, historical_df, models)
        bracket_component.render_bracket(st.session_state.bracket_data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: DATA MANAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_data:
    st.header("📥 Data Management")
    st.caption(
        "Batch-submit match results, correct mistakes, and manage the training dataset. "
        "All changes trigger an automatic model retrain."
    )

    fixtures_df  = load_fixtures_data()
    completed    = update_data.get_completed_results()
    placeholder_map = build_placeholder_map(fixtures_df, completed)

    if not completed.empty:
        completed_pairs = set(zip(completed["home_team"], completed["away_team"]))
    else:
        completed_pairs = set()

    pending_matches = [
        row for _, row in fixtures_df.iterrows()
        if (row["home_team"], row["away_team"]) not in completed_pairs
        and not any(kw in str(row.get("home_team","")) for kw in ["Winner","Runner-up","TBD","3rd"])
        and not any(kw in str(row.get("away_team","")) for kw in ["Winner","Runner-up","TBD","3rd"])
    ]

    st.subheader("🗂️ Batch Submit Results")
    st.caption(
        "Select any number of pending fixtures, enter their scores, then click "
        "**Submit All & Retrain** once. The model retrains only a single time regardless "
        "of how many results you enter."
    )

    if len(pending_matches) == 0:
        st.success("✅ All fixtures with known teams have been submitted.")
    else:
        pending_labels = [
            f"🏳️ {row['home_team']}  vs  🏳️ {row['away_team']}  ({row.get('group', '')} · {row.get('stage','')})"
            for row in pending_matches
        ]

        selected_labels = st.multiselect(
            "Select matches to submit (pick as many as needed):",
            options=pending_labels,
            default=[],
            key="batch_multiselect"
        )

        if len(selected_labels) > 0:
            selected_rows = [
                pending_matches[pending_labels.index(label)]
                for label in selected_labels
            ]

            st.markdown(f"**Enter scores for {len(selected_rows)} selected match(es):**")

            score_inputs = {}

            for idx, row in enumerate(selected_rows):
                home = row["home_team"]
                away = row["away_team"]

                col_matchname, col_hscore, col_separator, col_ascore = st.columns([4, 1.5, 0.5, 1.5])

                with col_matchname:
                    st.markdown(f"**🏳️ {home}** vs **🏳️ {away}**")
                with col_hscore:
                    h_score = st.number_input(
                        f"Home",
                        min_value=0, max_value=30, step=1, value=0,
                        key=f"batch_home_{home}_{away}_{idx}",
                        label_visibility="collapsed"
                    )
                with col_separator:
                    st.markdown("<div style='text-align:center;padding-top:8px;font-weight:bold;'>–</div>",
                                unsafe_allow_html=True)
                with col_ascore:
                    a_score = st.number_input(
                        f"Away",
                        min_value=0, max_value=30, step=1, value=0,
                        key=f"batch_away_{home}_{away}_{idx}",
                        label_visibility="collapsed"
                    )

                score_inputs[(home, away)] = (h_score, a_score, row.get("group"), row.get("stage","Group Stage"))

            st.markdown("---")

            batch_submit_btn = st.button(
                f"✅ Submit All {len(selected_rows)} Result(s) & Retrain Models",
                use_container_width=True,
                type="primary",
                key="batch_submit_btn"
            )

            if batch_submit_btn:
                results_to_submit = [
                    {
                        "home_team"  : home,
                        "away_team"  : away,
                        "home_score" : hs,
                        "away_score" : as_,
                        "group"      : grp,
                        "stage"      : stg
                    }
                    for (home, away), (hs, as_, grp, stg) in score_inputs.items()
                ]

                with st.spinner(
                    f"Submitting {len(results_to_submit)} result(s) and retraining models... "
                    f"This runs ETL and training ONCE. Please wait (~30–90 seconds)."
                ):
                    try:
                        success = update_data.run_full_update(results_to_submit, fixtures_df)
                        if success:
                            clear_all_caches()
                            st.success(
                                f"✅ {len(results_to_submit)} result(s) submitted. "
                                f"Group standings updated. Models retrained!"
                            )
                            st.rerun()
                        else:
                            st.error(
                                "⚠️ Results saved and standings updated, but model retraining failed. "
                                "Check the terminal for the error. Predictions use the previous model."
                            )
                    except ValueError as e:
                        st.error(f"❌ Submission rejected: {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error during batch submit: {e}")
                        st.exception(e)

    st.divider()

    st.subheader("✏️ Manage Submitted Results")

    completed_fresh = update_data.get_completed_results()

    if completed_fresh.empty:
        st.info("No results submitted yet. Use Batch Submit above to add matches.")
    else:
        st.caption(
            f"{len(completed_fresh)} result(s) on record. "
            f"Use the controls below to correct mistakes. Each action retrains the model."
        )

        if "editing_match" not in st.session_state:
            st.session_state["editing_match"] = None

        for idx, row in completed_fresh.iterrows():
            home      = row["home_team"]
            away      = row["away_team"]
            hs        = int(row["home_score"])
            as_       = int(row["away_score"])
            match_key = f"{home}__{away}"

            col_match, col_score, col_edit, col_delete = st.columns([4.5, 2, 1.25, 1.25])

            with col_match:
                grp_label = row.get("group","") or ""
                stg_label = row.get("stage","") or ""
                context   = f"Group {grp_label}" if grp_label else stg_label
                st.markdown(
                    f"**🏳️ {home}** vs **🏳️ {away}**  "
                    f"<span style='color:gray;font-size:0.85em;'>({context})</span>",
                    unsafe_allow_html=True
                )
            with col_score:
                st.markdown(
                    f"<div style='font-size:1.1em;font-weight:700;'>{hs} – {as_}</div>",
                    unsafe_allow_html=True
                )
            with col_edit:
                edit_btn = st.button("✏️ Edit", key=f"edit_btn_{match_key}_{idx}")
            with col_delete:
                delete_btn = st.button("🗑️ Delete", key=f"delete_btn_{match_key}_{idx}")

            if delete_btn:
                with st.spinner(f"Deleting {home} vs {away} and retraining..."):
                    try:
                        success = update_data.delete_result(home, away, fixtures_df)
                        if success:
                            clear_all_caches()
                            st.success(f"🗑️ Deleted: {home} vs {away}. Standings corrected. Models retrained.")
                            st.rerun()
                        else:
                            st.error("Deletion succeeded but retraining failed. Check the terminal.")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

            if edit_btn:
                if st.session_state["editing_match"] == (home, away):
                    st.session_state["editing_match"] = None
                else:
                    st.session_state["editing_match"] = (home, away)

            if st.session_state["editing_match"] == (home, away):
                with st.container():
                    st.markdown(
                        f"<div style='background:#1a1a2e;padding:12px;border-radius:8px;margin:4px 0 12px 0;'>"
                        f"Correcting score for <b>{home}</b> vs <b>{away}</b>:</div>",
                        unsafe_allow_html=True
                    )
                    edit_col1, edit_col2, edit_col3 = st.columns([2, 2, 3])

                    with edit_col1:
                        new_hs = st.number_input(
                            f"{home} Goals (corrected)",
                            min_value=0, max_value=30, step=1, value=hs,
                            key=f"edit_home_{match_key}_{idx}"
                        )
                    with edit_col2:
                        new_as = st.number_input(
                            f"{away} Goals (corrected)",
                            min_value=0, max_value=30, step=1, value=as_,
                            key=f"edit_away_{match_key}_{idx}"
                        )
                    with edit_col3:
                        edit_confirm = st.button(
                            f"💾 Save Correction",
                            key=f"edit_confirm_{match_key}_{idx}",
                            use_container_width=True
                        )
                        edit_cancel = st.button(
                            "✖ Cancel",
                            key=f"edit_cancel_{match_key}_{idx}",
                            use_container_width=True
                        )

                    if edit_cancel:
                        st.session_state["editing_match"] = None
                        st.rerun()

                    if edit_confirm:
                        if new_hs == hs and new_as == as_:
                            st.warning("⚠️ Scores are unchanged. Nothing to save.")
                        else:
                            with st.spinner(f"Saving correction and retraining..."):
                                try:
                                    success = update_data.edit_result(home, away, new_hs, new_as, fixtures_df)
                                    if success:
                                        st.session_state["editing_match"] = None
                                        clear_all_caches()
                                        st.success(
                                            f"✅ Corrected: {home} {new_hs}–{new_as} {away}. "
                                            f"Standings rebuilt. Models retrained."
                                        )
                                        st.rerun()
                                    else:
                                        st.error("Correction saved but retraining failed. Check the terminal.")
                                except Exception as e:
                                    st.error(f"Edit failed: {e}")
                                    st.exception(e)

            st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: LIVE CATCH-UP & DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_catchup:
    st.header("⚡ Match Database & Live Catch-Up")
    st.caption("Manage all submitted matches. Check 'Played?' to include a match in the dataset. Uncheck to remove it. You can edit scores freely or add new rows at the bottom.")

    fixtures_df  = load_fixtures_data()
    completed    = update_data.get_completed_results()
    
    # Map existing completed scores
    completed_map = {}
    if not completed.empty:
        for _, row in completed.iterrows():
            completed_map[(row["home_team"], row["away_team"])] = (row["home_score"], row["away_score"], row.get("date", ""))

    # Find all valid World Cup fixtures
    valid_fixtures = [
        row for _, row in fixtures_df.iterrows()
        if not any(kw in str(row.get("home_team","")) for kw in ["Winner","Runner-up","TBD","3rd"])
        and not any(kw in str(row.get("away_team","")) for kw in ["Winner","Runner-up","TBD","3rd"])
    ]

    catchup_data = []
    
    # 1. Add all valid fixtures (either played or not)
    fixture_pairs = set()
    for row in valid_fixtures:
        h = row["home_team"]
        a = row["away_team"]
        fixture_pairs.add((h, a))
        
        has_result = (h, a) in completed_map
        if has_result:
            h_score, a_score, c_date = completed_map[(h, a)]
        else:
            h_score, a_score, c_date = 0, 0, row.get("date", "")
            
        catchup_data.append({
            "Played?": has_result,
            "Date": c_date,
            "Home Team": h,
            "Home Score": int(h_score),
            "Away Score": int(a_score),
            "Away Team": a,
            "Group": row.get("group"),
            "Stage": row.get("stage", "Group Stage")
        })
        
    # 2. Add any completed matches that weren't in the fixtures list (custom added matches)
    if not completed.empty:
        for _, row in completed.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            if (h, a) not in fixture_pairs:
                catchup_data.append({
                    "Played?": True,
                    "Date": row.get("date", ""),
                    "Home Team": h,
                    "Home Score": int(row["home_score"]),
                    "Away Score": int(row["away_score"]),
                    "Away Team": a,
                    "Group": row.get("group"),
                    "Stage": row.get("stage", "Group Stage")
                })
                
    catchup_df = pd.DataFrame(catchup_data)
    all_wc_teams = sorted(TEAM_TO_GROUP.keys())

    edited_catchup = st.data_editor(
        catchup_df,
        num_rows="dynamic",
        column_config={
            "Played?": st.column_config.CheckboxColumn("Played?", required=True),
            "Date": st.column_config.TextColumn("Date", required=True),
            "Home Team": st.column_config.SelectboxColumn("Home Team", options=all_wc_teams, required=True),
            "Home Score": st.column_config.NumberColumn("Home Score", min_value=0, max_value=30),
            "Away Score": st.column_config.NumberColumn("Away Score", min_value=0, max_value=30),
            "Away Team": st.column_config.SelectboxColumn("Away Team", options=all_wc_teams, required=True),
            "Group": st.column_config.TextColumn("Group"),
            "Stage": st.column_config.TextColumn("Stage")
        },
        hide_index=True,
        key="full_catchup_editor",
        use_container_width=True,
        height=600,
    )

    st.markdown("---")
    if st.button("🚀 Submit Batch & Rebuild Database", type="primary", use_container_width=True):
        # We only submit matches where Played? is True
        to_submit = edited_catchup[edited_catchup["Played?"] == True]
        
        gathered_matches = []
        for _, row in to_submit.iterrows():
            if pd.isna(row["Home Team"]) or pd.isna(row["Away Team"]):
                continue
            gathered_matches.append({
                "home_team"  : row["Home Team"],
                "away_team"  : row["Away Team"],
                "home_score" : int(row["Home Score"]),
                "away_score" : int(row["Away Score"]),
                "group"      : row["Group"] if pd.notna(row["Group"]) else TEAM_TO_GROUP.get(row["Home Team"]),
                "stage"      : row["Stage"] if pd.notna(row["Stage"]) else "Group Stage",
                "date"       : row["Date"] if pd.notna(row["Date"]) else ""
            })
            
        with st.spinner(f"Saving {len(gathered_matches)} matches and retraining..."):
            try:
                # Run full update overwriting the file
                if update_data.run_full_update(gathered_matches, fixtures_df):
                    clear_all_caches()
                    st.success("✅ Models and standings rebuilt perfectly!")
                    st.rerun()
                else:
                    st.error("Retraining failed. Check terminal.")
            except Exception as e:
                st.error(f"Error: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR: Status Panel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.header("🏆 WC 2026 Predictor")
    st.caption("v2.1")
    st.divider()

    completed_sb = update_data.get_completed_results()
    total_fixtures = len(load_fixtures_data())
    submitted_count = len(completed_sb)

    st.metric("Results Submitted", f"{submitted_count} / {total_fixtures}")

    if total_fixtures > 0:
        st.progress(min(1.0, submitted_count / total_fixtures))
    else:
        st.progress(0.0)

    st.divider()

    st.markdown("**📥 To submit or edit results:**")
    st.markdown("Go to the **Data Management** tab above.")

    st.divider()

    import os
    from config import CLASSIFIER_PATH
    model_exists = os.path.exists(CLASSIFIER_PATH)

    if model_exists:
        st.success("✅ Model: Trained & Ready")
    else:
        st.error("⚠️ Model: Not trained")
        st.code("python src/train.py", language="bash")

    st.divider()
    st.caption("Predictions are statistical estimates. Not for betting.")
