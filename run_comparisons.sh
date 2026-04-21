#!/bin/bash

# ============================================================
# MCTS Agent Comparison Runner
# ============================================================
# Runs all meaningful matchups between MCTS variants,
# in both directions, 20 games each.
# Results are logged per-matchup AND appended to a summary file.
# ============================================================

GAMES=7
SUMMARY_FILE="comparison_results/SUMMARY.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

mkdir -p comparison_results

echo "============================================================" >> "$SUMMARY_FILE"
echo "MCTS Comparison Run — $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Games per matchup: $GAMES" >> "$SUMMARY_FILE"
echo "============================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# -----------------------------------------------------------
# Helper: run one matchup in both directions
#   $1 = agent A (e.g. "mcts/mcts_2.py")
#   $2 = agent B (e.g. "mcts/mcts_3.py")
#   $3 = human-readable label for the log folder (e.g. "mcts2_vs_mcts3")
# -----------------------------------------------------------
run_matchup() {
    local A="$1"
    local B="$2"
    local LABEL="$3"

    local LOG_DIR="comparison_results/${LABEL}"
    mkdir -p "$LOG_DIR"

    # ---- Forward direction: A as P0, B as P1 ----
    local LOG_FWD="${LOG_DIR}/${LABEL}__p0_${LABEL%%_vs_*}__p1_${LABEL##*_vs_}"
    echo ">>> Running: $A (P0) vs $B (P1)  [$GAMES games]"
    python3 manager.py -p0 "$A" -p1 "$B" -n "$GAMES" -l "$LOG_FWD" \
        2>&1 | tee "${LOG_FWD}.txt"

    # Append to summary
    {
        echo "--- $A (P0)  vs  $B (P1) ---"
        grep -E "wins:|Draws:" "${LOG_FWD}.txt" | tail -4
        echo ""
    } >> "$SUMMARY_FILE"

    # ---- Reverse direction: B as P0, A as P1 ----
    local LOG_REV="${LOG_DIR}/${LABEL}__p0_${LABEL##*_vs_}__p1_${LABEL%%_vs_*}"
    echo ">>> Running: $B (P0) vs $A (P1)  [$GAMES games]"
    python3 manager.py -p0 "$B" -p1 "$A" -n "$GAMES" -l "$LOG_REV" \
        2>&1 | tee "${LOG_REV}.txt"

    {
        echo "--- $B (P0)  vs  $A (P1) ---"
        grep -E "wins:|Draws:" "${LOG_REV}.txt" | tail -4
        echo ""
    } >> "$SUMMARY_FILE"

    echo "" >> "$SUMMARY_FILE"
}

# ============================================================
# GROUP 1 — Fixed-time ladder (consecutive pairs)
# ============================================================
echo "" >> "$SUMMARY_FILE"
echo "════ GROUP 1: Fixed-time ladder ════" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

#run_matchup "mcts/mcts_2.py" "mcts/mcts_3.py" "mcts2_vs_mcts3"
#run_matchup "mcts/mcts_3.py" "mcts/mcts_4.py" "mcts3_vs_mcts4"
#run_matchup "mcts/mcts_4.py" "mcts/mcts_5.py" "mcts4_vs_mcts5"
#run_matchup "mcts/mcts_5.py" "mcts/mcts_6.py" "mcts5_vs_mcts6"
#run_matchup "mcts/mcts_6.py" "mcts/mcts_7.py" "mcts6_vs_mcts7"
#run_matchup "mcts/mcts_7.py" "mcts/mcts_8.py" "mcts7_vs_mcts8"

# ============================================================
# GROUP 2 — Transposition Table vs representative fixed times
# ============================================================
echo "" >> "$SUMMARY_FILE"
echo "════ GROUP 2: TT vs fixed-time agents ════" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

run_matchup "mcts/mcts_transposition_table.py" "mcts/mcts_2.py" "tt_vs_mcts2"
run_matchup "mcts/mcts_transposition_table.py" "mcts/mcts_4.py" "tt_vs_mcts4"
#run_matchup "mcts/mcts_transposition_table.py" "mcts/mcts_6.py" "tt_vs_mcts6"
#run_matchup "mcts/mcts_transposition_table.py" "mcts/mcts_8.py" "tt_vs_mcts8"

# ============================================================
# GROUP 3 — TT+Dynamic vs representative fixed times
# ============================================================
echo "" >> "$SUMMARY_FILE"
echo "════ GROUP 3: TT+Dynamic vs fixed-time agents ════" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

run_matchup "mcts/mcts_transposition_table_dynamic_time.py" "mcts/mcts_2.py" "tt_dyn_vs_mcts2"
run_matchup "mcts/mcts_transposition_table_dynamic_time.py" "mcts/mcts_4.py" "tt_dyn_vs_mcts4"
#run_matchup "mcts/mcts_transposition_table_dynamic_time.py" "mcts/mcts_6.py" "tt_dyn_vs_mcts6"
#run_matchup "mcts/mcts_transposition_table_dynamic_time.py" "mcts/mcts_8.py" "tt_dyn_vs_mcts8"

# ============================================================
# GROUP 4 — TT vs TT+Dynamic (head-to-head)
# ============================================================
echo "" >> "$SUMMARY_FILE"
echo "════ GROUP 4: TT vs TT+Dynamic ════" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

#run_matchup "mcts/mcts_transposition_table.py" "mcts/mcts_transposition_table_dynamic_time.py" "tt_vs_tt_dyn"

# ============================================================
echo ""
echo "All matchups complete."
echo "Summary written to: $SUMMARY_FILE"
echo "Individual logs in: comparison_results/"