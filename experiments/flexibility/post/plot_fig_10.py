import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from plot_config import get_full_column_fig, get_three_quarter_fig
from collections import defaultdict

LOG_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "parametric_sweep", "all_logs")
MEAN_DURATION = 30.0  # seconds

POLICY_LABELS = {
    "fifo_policy": "FIFO",
    "shortest_first": "Shortest First",
    "longest_first": "Longest First",
    "largest_first": "Largest First",
}

POLICY_MARKERS = {
    "fifo_policy": "o",
    "shortest_first": "s",
    "longest_first": "^",
    "largest_first": "D",
}

# Parse all log directories
data = defaultdict(lambda: defaultdict(list))

for dirname in sorted(os.listdir(LOG_BASE)):
    if not dirname.startswith("logs_"):
        continue

    logfile = os.path.join(LOG_BASE, dirname, "script.log")
    if not os.path.isfile(logfile) or os.path.getsize(logfile) == 0:
        continue

    suffix = dirname[len("logs_"):]
    parts = suffix.rsplit("_", 2)
    if len(parts) != 3:
        continue
    policy_list_str, variance_str, iteration_str = parts

    try:
        variance = float(variance_str)
    except ValueError:
        continue

    cv = math.sqrt(variance) / MEAN_DURATION

    with open(logfile) as f:
        content = f.read()

    policy_pattern = re.compile(
        r"\*+Running (\S+) policy\*+.*?"
        r"Total execution time: ([\d.]+)",
        re.DOTALL,
    )
    for match in policy_pattern.finditer(content):
        policy = match.group(1)
        duration = float(match.group(2))
        data[policy][cv].append(duration)

# Plot — IEEE half-column: 3.5 x 2.5 inches
fig, ax = get_three_quarter_fig()

for policy in ["fifo_policy", "shortest_first", "longest_first", "largest_first"]:
    if policy not in data:
        continue
    cvs = sorted(data[policy].keys())
    mean_durations = [np.mean(data[policy][cv]) for cv in cvs]
    std_durations = [np.std(data[policy][cv]) for cv in cvs]

    ax.errorbar(
        cvs, mean_durations, yerr=std_durations,
        marker=POLICY_MARKERS[policy],
        capsize=3,
        label=POLICY_LABELS[policy],
    )

ax.set_xlabel("Coefficient of Variation")
ax.set_ylabel("Execution Time (s)")
handles, labels_ = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels_,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.12),
    frameon=False,
)
fig.tight_layout(rect=[0, 0, 1, 0.85], pad=0.3)

outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ["pdf"]:
    outpath = os.path.join(outdir, f"fig_10.{ext}")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved {outpath}")
