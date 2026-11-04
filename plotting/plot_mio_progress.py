"""
plot_gurobi_progress.py

Parses Gurobi MIP log files, extracts incumbent (best solution) updates,
normalizes them by the final optimized value, and plots the solving progress
across multiple runs with a median/mean line and a shaded ribbon.

Usage:
    python plot_gurobi_progress.py

Edit LOG_FILES below to point to your log files.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

LOG_FILES = [
    f"batch_from_cluster/neurips_eval/gerry_spsf-ACSEmployment/.submitit/10898228_{i}/10898228_{i}_0_log.out" for i in range(5,25)
]

# Choose x-axis: "time" (seconds) or "nodes" (B&B nodes explored)
X_AXIS = "time"

# Whether to normalize y-axis by the final incumbent of each run (True),
# or plot raw objective values (False)
NORMALIZE = True

# Aggregate line: "median" or "mean"
AGGREGATE = "median"

# ── Parsing ───────────────────────────────────────────────────────────────────

# Matches lines that update the incumbent (H or * prefix), e.g.:
#   H    0     0                       0.0431869    0.24178   460%     -    2s
#   *   14    12              15       0.0521000    0.24178   364%  23.4    5s
# Groups: (incumbent_value, time_seconds)
INCUMBENT_RE = re.compile(
    r"^[H*]\s+"           # H or * at start
    r"\d+\s+\d+\s+"       # Expl  Unexpl
    r"(?:\S+\s+\S+\s+)?"  # optional Obj Depth (not always present on H lines)
    r"\s*(\d[\d.e+\-]+)"  # incumbent value
    r"\s+\S+"             # BestBd
    r"\s+\S+"             # Gap
    r"\s+\S+"             # It/Node
    r"\s+([\d.]+)s"       # time in seconds
)

# Matches regular (non-incumbent) tree rows to capture elapsed time for nodes
NODE_RE = re.compile(
    r"^\s*(\d+)\s+\d+"    # Expl nodes
    r".*?([\d.]+)s\s*$"   # time at end
)


def parse_log(filepath: str) -> dict:
    """
    Returns a dict with:
        incumbents: list of (x_value, incumbent_value) tuples
                    x_value is time (s) or node count depending on X_AXIS
        final: the last incumbent value seen (used for normalization)
    """
    incumbents = []
    path = Path(filepath)
    if not path.exists():
        print(f"  [warning] File not found: {filepath}")
        return {"incumbents": [], "final": None}

    with open(path, "r", errors="replace") as f:
        for raw_line in f:
            # Strip log prefixes like "[2026-05-01 22:59:15,429][gurobipy][INFO] - "
            line = re.sub(r"^\[.*?\]\[.*?\]\[.*?\]\s*-\s*", "", raw_line).strip()

            m = INCUMBENT_RE.match(line)
            if m:
                value = float(m.group(1))
                t = float(m.group(2))
                x = t  # default: use time
                if X_AXIS == "nodes":
                    # Approximate: use Expl count from line start
                    nm = re.match(r"^[H*]\s+(\d+)", line)
                    x = int(nm.group(1)) if nm else t
                incumbents.append((x, value))

    if not incumbents:
        print(f"  [warning] No incumbent updates found in: {filepath}")
        return {"incumbents": [], "final": None}

    final_value = incumbents[-1][1]
    return {"incumbents": incumbents, "final": final_value}


# ── Interpolation helpers ─────────────────────────────────────────────────────

def interpolate_to_grid(incumbents, x_grid):
    """
    Given a step-function of (x, y) incumbent updates,
    interpolate (hold previous value) onto x_grid.
    Returns array of y values aligned with x_grid.
    """
    if not incumbents:
        return np.full(len(x_grid), np.nan)

    xs, ys = zip(*incumbents)
    xs, ys = np.array(xs), np.array(ys)

    result = np.full(len(x_grid), np.nan)
    for i, xg in enumerate(x_grid):
        # Find last incumbent update at or before xg
        idx = np.searchsorted(xs, xg, side="right") - 1
        if idx >= 0:
            result[i] = ys[idx]
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Parsing log files...")
    runs = []
    for fpath in LOG_FILES:
        print(f"  {fpath}")
        data = parse_log(fpath)
        if data["incumbents"] and data["final"] is not None:
            runs.append(data)
        else:
            print("Error")

    if not runs:
        print("No valid runs found. Check LOG_FILES paths.")
        return

    # Fixed x range: 1 second → 1 day (86400 s)
    x_min = 1
    x_max = 86400
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    # Interpolate each run onto the grid, optionally normalizing.
    # Runs that have not yet produced any incumbent at a given x get -inf,
    # so they appear to come from below the plot rather than causing a jump.
    curves = []
    for r in runs:
        y = interpolate_to_grid(r["incumbents"], x_grid)
        if NORMALIZE and r["final"] != 0:
            y = y / r["final"]
        # Replace leading NaNs (before first incumbent) with -inf
        first_valid = np.argmax(~np.isnan(y))
        if np.isnan(y[0]):
            y[:first_valid] = -np.inf
        curves.append(y)

    curves = np.array(curves, dtype=float)  # shape (n_runs, n_grid)

    # For aggregate/ribbon, treat -inf as NaN so they don't drag the stats down
    curves_finite = np.where(np.isfinite(curves), curves, np.nan)

    # Compute aggregate and ribbon (ignoring NaNs)
    if AGGREGATE == "median":
        agg_line = np.nanmedian(curves_finite, axis=0)
    else:
        agg_line = np.nanmean(curves_finite, axis=0)

    lower = np.nanmin(curves_finite, axis=0)
    upper = np.nanmax(curves_finite, axis=0)

    # ── Plotting ──────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # fig.patch.set_facecolor("#f8f7f4")
    # ax.set_facecolor("#f8f7f4")

    ribbon_color = "#6083ab"
    line_color   = "#1a3d6e"
    run_color    = "#3072be"

    # Individual run step-lines (thin, semi-transparent).
    # Runs start from their first incumbent; before that they are simply absent.
    for r in runs:
        xs, ys = zip(*r["incumbents"])
        if NORMALIZE and r["final"] != 0:
            ys = tuple(v / r["final"] for v in ys)
        ax.step(xs, ys, where="post", color=run_color, alpha=0.18, linewidth=1.0)

    # Ribbon
    ax.fill_between(x_grid, lower, upper,
                    where=~np.isnan(lower) & ~np.isnan(upper),
                    color=ribbon_color, alpha=0.18, label="Min–max range")

    # Aggregate line
    ax.step(x_grid, agg_line, where="post",
            color=line_color, linewidth=2.2,
            label=f"{AGGREGATE.capitalize()} across {len(runs)} runs")

    # Reference line at 1.0 when normalized
    # if NORMALIZE:
    #     ax.axhline(1.0, color="#cc4a2e", linewidth=1.2, linestyle="--",
    #                alpha=0.7, label="Final optimum (1.0)")

    # Formatting
    xlabel = "Solve time" if X_AXIS == "time" else "B&B nodes explored"
    ylabel = "Incumbent / final value" if NORMALIZE else "Incumbent objective"
    ax.set_xlabel(xlabel, fontsize=12)#, color="#333")
    ax.set_ylabel(ylabel, fontsize=12)#, color="#333")
    # ax.set_title("Gurobi MIP solving progress", fontsize=14, fontweight="bold",
                #  color="#1a1a1a", pad=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(colors="#555", labelsize=10)
    ax.grid(axis="both", color="#ddd", linewidth=0.7, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)

    # Hand-picked time ticks (in seconds) with clean labels
    if X_AXIS == "time":
        import matplotlib.ticker as ticker
        time_ticks = [
            (1,        "1s"),
            (10,        "10s"),
            (60,        "1m"),
            (5*60,      "5m"),
            (10*60,     "10m"),
            (30*60,     "30m"),
            (3600,      "1h"),
            (5*3600,    "5h"),
            (12*3600,   "12h"),
            (86400,     "1d"),
        ]
        tick_positions = [t for t, _ in time_ticks]
        tick_labels    = [l for _, l in time_ticks]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    # ax.legend(fontsize=10, framealpha=0.6, loc="upper right")

    plt.tight_layout()
    out_path = "gurobi_progress.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
