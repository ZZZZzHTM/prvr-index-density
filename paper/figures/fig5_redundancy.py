"""Plot per-video coverage and frame-output distributions (Figure 5)."""
from figure_io import figure_paths, load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DATASETS = {"tvr": "TVR", "activitynet": "ActivityNet", "charades": "Charades"}
PANEL_LABELS = {
    "all_top8_coverage": "Top-8 coverage",
    "frame_adjacent_cosine": "Adjacent cosine",
    "frame_pc90": "Components for\n90% energy",
}
FONT_PT = 7.4
plt.rcParams.update({
    "font.family": "STIXGeneral", "font.size": FONT_PT, "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.55,
    "axes.edgecolor": "#64696E", "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.labelpad": 2,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
})

results, output = figure_paths(__doc__)
summary = load(results, "rq2_usage_redundancy/RESULTS.json")
statistics = {}
for dataset in DATASETS:
    case = next(c for c in summary["cases"] if c["case"]["key"] == f"B0_{dataset}")
    distributions = case["summaries"]["all"]["distributions"]
    statistics[dataset] = {}
    for metric in PANEL_LABELS:
        d = distributions[metric]
        quantiles = d["quantiles"]
        statistics[dataset][metric] = {
            "n": d["n"], "dropped": d["missing"], "mean": d["mean"],
            "whislo": quantiles["0.05"], "q1": quantiles["0.25"],
            "med": quantiles["0.5"], "q3": quantiles["0.75"], "whishi": quantiles["0.95"],
        }
fig, axes = plt.subplots(1, 3, figsize=(12.2 / 2.54, 3.36 / 2.54))
fig.subplots_adjust(left=0.095, right=0.992, bottom=0.305, top=0.82, wspace=0.64)
for panel, (col, label), ax in zip("abc", PANEL_LABELS.items(), axes):
    stats = [statistics[ds][col] for ds in DATASETS]
    ax.bxp(stats, showfliers=False, widths=0.50, patch_artist=True,
           boxprops={"edgecolor": "#1F8A7A", "facecolor": "#E1F0EC", "linewidth": 0.8},
           capprops={"color": "#51676A", "linewidth": 0.7},
           whiskerprops={"color": "#51676A", "linewidth": 0.7},
           medianprops={"color": "#165D53", "linewidth": 1.1})
    ax.set_xticks([1, 2, 3], labels=list(DATASETS.values()), rotation=34,
                  horizontalalignment="right", rotation_mode="anchor")
    ax.set_ylabel(label)
    ax.text(0, 1.06, f"({panel})", transform=ax.transAxes, va="bottom", ha="left")
    ax.tick_params(axis="x", length=0)
axes[0].set_ylim(0.19, 0.685)
axes[0].set_yticks([0.2, 0.4, 0.6])
axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
axes[1].set_ylim(0.75, 1)
axes[1].set_yticks([0.8, 0.9, 1.0])
axes[2].set_ylim(0, 17)
axes[2].set_yticks([0, 5, 10, 15])

fig.savefig(output / "fig5_redundancy.pdf", metadata={"Author": ""})
plt.close(fig)
