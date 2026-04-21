#!/usr/bin/env python3
"""
parse_and_plot.py
=================
Reads the .txt log files produced by run_comparisons.sh and generates
clear comparison charts.

Usage:
    python3 parse_and_plot.py                  # reads ./comparison_results/
    python3 parse_and_plot.py --dir my_logs    # custom directory
"""

import os
import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Short display names ──────────────────────────────────────────────────────

def short_name(agent_path: str) -> str:
    name = Path(agent_path).stem          # e.g. "mcts_3"
    mapping = {
        "mcts_transposition_table_dynamic_time": "TT+Dyn",
        "mcts_transposition_table": "TT",
    }
    for k, v in mapping.items():
        if k in name:
            return v
    m = re.search(r"mcts_(\d+)", name)
    if m:
        return f"MCTS-{m.group(1)}s"
    return name

# ── Log parser ───────────────────────────────────────────────────────────────

def parse_log(filepath: str) -> dict | None:
    """
    Extract win counts from a manager.py output file.
    Returns dict with keys: p0, p1, p0_wins, p1_wins, draws, total
    or None if the file doesn't contain a results block.
    """
    text = Path(filepath).read_text(errors="replace")

    # Find the results block
    block = re.search(
        r"=== Results over (\d+) game\(s\) ===(.*?)(?=\n===|\Z)",
        text, re.DOTALL
    )
    if not block:
        return None

    total = int(block.group(1))
    body = block.group(2)

    rows = re.findall(
        r"^\s+(.+?)\s+wins:\s+(\d+)\s+\([\d.]+%\)", body, re.MULTILINE
    )
    draws_m = re.search(r"Draws:\s+(\d+)", body)
    draws = int(draws_m.group(1)) if draws_m else 0

    if len(rows) < 2:
        return None

    return {
        "p0":       rows[0][0].strip(),
        "p1":       rows[1][0].strip(),
        "p0_wins":  int(rows[0][1]),
        "p1_wins":  int(rows[1][1]),
        "draws":    draws,
        "total":    total,
    }

# ── Collect results ──────────────────────────────────────────────────────────

def collect_results(results_dir: str) -> list[dict]:
    records = []
    for txt in sorted(Path(results_dir).rglob("*.txt")):
        if txt.name == "SUMMARY.txt":
            continue
        r = parse_log(str(txt))
        if r:
            r["file"] = str(txt)
            records.append(r)
    return records

# ── Helpers ──────────────────────────────────────────────────────────────────

COLORS = {
    "win":  "#4CAF50",
    "loss": "#F44336",
    "draw": "#9E9E9E",
}

def win_pct(record, perspective_agent):
    """Win % for perspective_agent in this record."""
    if perspective_agent == record["p0"]:
        return 100 * record["p0_wins"] / record["total"]
    return 100 * record["p1_wins"] / record["total"]

# ── Plot 1: Fixed-time ladder (win % vs time budget) ────────────────────────

def plot_ladder(records, out_dir):
    """
    For each consecutive pair (Ns vs (N+1)s), show the win % of the faster
    agent (lower time) — gives an ELO-flavoured time-scaling curve.
    """
    ladder_pairs = [
        (f"mcts/mcts_{i}.py", f"mcts/mcts_{i+1}.py") for i in range(2, 8)
    ]

    xs, ys_lower, ys_higher = [], [], []

    for a, b in ladder_pairs:
        sa, sb = short_name(a), short_name(b)
        # find the record where these two played (either direction)
        match = [r for r in records
                 if {r["p0"], r["p1"]} == {a, b}]
        if not match:
            continue
        # aggregate both directions
        total_a = sum(
            r["p0_wins"] if r["p0"] == a else r["p1_wins"] for r in match
        )
        total_games = sum(r["total"] for r in match)
        pct_a = 100 * total_a / total_games if total_games else 0

        m = re.search(r"mcts_(\d+)", a)
        xs.append(int(m.group(1)))
        ys_lower.append(pct_a)          # win % of lower time agent
        ys_higher.append(100 - pct_a)  # win % of higher time agent

    if not xs:
        print("[plot_ladder] No ladder data found, skipping.")
        return

    x = np.array(xs)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, ys_lower,  0.4, label="Temps moindre (Ns)", color=COLORS["loss"], alpha=0.85)
    ax.bar(x + 0.2, ys_higher, 0.4, label="Temps plus élevé (N+1s)", color=COLORS["win"], alpha=0.85)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}s vs {i+1}s" for i in xs])
    ax.set_ylabel("% de victoire (peu importe qui joue en premier)")
    ax.set_ylim(0, 100)
    ax.set_title("MCTS à temps fixe: impact du temps par coup")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "01_fixed_time_ladder.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ── Plot 2: TT and TT+Dyn vs fixed-time agents ──────────────────────────────

def plot_advanced_vs_fixed(records, out_dir):
    """
    Grouped bar chart: for each fixed-time opponent, show win % of
    TT and TT+Dyn (averaged over both directions).
    """
    fixed_agents = [f"mcts/mcts_{t}.py" for t in [2, 4, 6, 8]]
    advanced = {
        "TT":    "mcts/mcts_transposition_table.py",
        "TT+Dyn": "mcts/mcts_transposition_table_dynamic_time.py",
    }

    results = {label: [] for label in advanced}
    x_labels = []

    for fixed in fixed_agents:
        x_labels.append(short_name(fixed))
        for label, adv in advanced.items():
            match = [r for r in records
                     if {r["p0"], r["p1"]} == {adv, fixed}]
            if not match:
                results[label].append(None)
                continue
            total_adv = sum(
                r["p0_wins"] if r["p0"] == adv else r["p1_wins"]
                for r in match
            )
            total_games = sum(r["total"] for r in match)
            results[label].append(100 * total_adv / total_games if total_games else 0)

    x = np.arange(len(x_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    adv_colors = ["#2196F3", "#FF9800"]
    for i, (label, vals) in enumerate(results.items()):
        cleaned = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - 0.5) * width, cleaned, width,
                      label=label, color=adv_colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("% de victoire (peu importe qui joue en premier)")
    ax.set_ylim(0, 110)
    ax.set_title("MCTS : TT & TT+temps dynamique vs Fagents à temps fixe")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "02_advanced_vs_fixed.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ── Plot 3: Head-to-head matrix (all matchups) ──────────────────────────────

def plot_heatmap(records, out_dir):
    """
    N×N win-rate matrix. Cell [i,j] = win % of agent i against agent j
    (averaged over both directions if available).
    """
    # Collect all agents seen in logs
    agents_seen = set()
    for r in records:
        agents_seen.add(r["p0"])
        agents_seen.add(r["p1"])

    # Preferred order
    order_keys = [f"mcts/mcts_{i}.py" for i in range(2, 9)] + [
        "mcts/mcts_transposition_table.py",
        "mcts/mcts_transposition_table_dynamic_time.py",
    ]
    agents = [a for a in order_keys if a in agents_seen]
    # append any unexpected agents
    for a in sorted(agents_seen):
        if a not in agents:
            agents.append(a)

    n = len(agents)
    if n < 2:
        return

    matrix = np.full((n, n), np.nan)

    for i, ai in enumerate(agents):
        for j, aj in enumerate(agents):
            if i == j:
                continue
            match = [r for r in records
                     if {r["p0"], r["p1"]} == {ai, aj}]
            if not match:
                continue
            wins_i = sum(
                r["p0_wins"] if r["p0"] == ai else r["p1_wins"]
                for r in match
            )
            total = sum(r["total"] for r in match)
            matrix[i, j] = 100 * wins_i / total if total else 0

    labels = [short_name(a) for a in agents]

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Opponent (column)")
    ax.set_ylabel("Agent (row)")
    ax.set_title("Win % matrix — row agent vs column agent")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.0f}",
                        ha="center", va="center",
                        fontsize=9,
                        color="black" if 25 < matrix[i, j] < 75 else "white")

    plt.colorbar(im, ax=ax, label="Win %")
    fig.tight_layout()
    path = os.path.join(out_dir, "03_win_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ── Plot 4: TT vs TT+Dynamic head-to-head ───────────────────────────────────

def plot_tt_vs_tt_dyn(records, out_dir):
    a = "mcts/mcts_transposition_table.py"
    b = "mcts/mcts_transposition_table_dynamic_time.py"

    match = [r for r in records if {r["p0"], r["p1"]} == {a, b}]
    if not match:
        print("[plot_tt_vs_tt_dyn] No data found, skipping.")
        return

    wins_a = sum(r["p0_wins"] if r["p0"] == a else r["p1_wins"] for r in match)
    wins_b = sum(r["p0_wins"] if r["p0"] == b else r["p1_wins"] for r in match)
    #draws  = sum(r["draws"] for r in match)
    total  = sum(r["total"] for r in match)

    #labels = [short_name(a), short_name(b), "Draw"]
    labels = [short_name(a), short_name(b)]
    #values = [100 * wins_a / total, 100 * wins_b / total, 100 * draws / total]
    values = [100 * wins_a / total, 100 * wins_b / total]
    #colors = [COLORS["loss"], COLORS["win"], COLORS["draw"]]
    colors = [COLORS["loss"], COLORS["win"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom")
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("% de victoire")
    ax.set_ylim(0, 110)
    #ax.set_title(f"Head-to-head: {short_name(a)} vs {short_name(b)}\n({total} games total, both directions)")
    ax.set_title(f"{short_name(a)} vs {short_name(b)}\n(sur {total} jeux, peu importe qui commence)")
    fig.tight_layout()
    path = os.path.join(out_dir, "04_tt_vs_tt_dyn.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ── Plot 5: First-move advantage ────────────────────────────────────────────

def plot_first_move_advantage(records, out_dir):
    """
    For every matchup played in both directions, compare P0 win rate vs P1 win rate.
    Shows whether going first matters.
    """
    # Group records by unordered pair
    from collections import defaultdict
    pairs = defaultdict(list)
    for r in records:
        key = tuple(sorted([r["p0"], r["p1"]]))
        pairs[key].append(r)

    p0_win_rates, p1_win_rates, pair_labels = [], [], []
    for (a, b), group in pairs.items():
        fwd = [r for r in group if r["p0"] == a]
        rev = [r for r in group if r["p0"] == b]
        if not fwd or not rev:
            continue
        p0_wr_fwd = np.mean([r["p0_wins"] / r["total"] * 100 for r in fwd])
        p0_wr_rev = np.mean([r["p0_wins"] / r["total"] * 100 for r in rev])
        p0_win_rates.append(p0_wr_fwd)
        p1_win_rates.append(p0_wr_rev)  # p0 of rev game = originally p1
        pair_labels.append(f"{short_name(a)}\nvs\n{short_name(b)}")

    if not pair_labels:
        print("[plot_first_move_advantage] Not enough bidirectional data, skipping.")
        return

    x = np.arange(len(pair_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(pair_labels) * 1.2), 5))
    ax.bar(x - width/2, p0_win_rates, width, label="Win % as P0 (goes first)", color="#1976D2", alpha=0.85)
    ax.bar(x + width/2, p1_win_rates, width, label="Win % as P1 (goes second)", color="#E64A19", alpha=0.85)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=7)
    ax.set_ylabel("Win % when playing as P0 / P1")
    ax.set_ylim(0, 110)
    ax.set_title("First-Move Advantage Analysis")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "05_first_move_advantage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse MCTS comparison logs and plot results.")
    parser.add_argument("--dir", default="comparison_results",
                        help="Directory containing .txt log files (default: comparison_results)")
    parser.add_argument("--out", default=None,
                        help="Output directory for plots (default: same as --dir)")
    args = parser.parse_args()

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading logs from: {args.dir}")
    records = collect_results(args.dir)
    print(f"Found {len(records)} result records.\n")

    if not records:
        print("No results found. Run run_comparisons.sh first.")
        return

    plot_ladder(records, out_dir)
    plot_advanced_vs_fixed(records, out_dir)
    plot_heatmap(records, out_dir)
    plot_tt_vs_tt_dyn(records, out_dir)
    plot_first_move_advantage(records, out_dir)

    print(f"\nAll plots saved to: {out_dir}/")

if __name__ == "__main__":
    main()
