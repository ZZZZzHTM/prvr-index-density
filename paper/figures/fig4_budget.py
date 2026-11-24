"""Plot retrieval quality against the allocation of clip and frame slots (Figure 4)."""
from figure_io import figure_paths, load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

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
data = load(results, "budget_split/FRONTIER.json")
rows = data["points"]
frontier = set(data["frontier"])
assert frontier == {p["configuration"] for p in rows if p["pareto_frontier"]}
fig, ax = plt.subplots(figsize=(7.564 / 2.54, 4.35 / 2.54))
fig.subplots_adjust(left=0.16, right=0.984, bottom=0.21, top=0.95)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticks([12, 16, 24, 40, 80, 160], labels=["12", "16", "24", "40", "80", "160"])
ax.set_xlim(10.9, 177)
ax.set_ylim(-1.49, 0.16)
ax.set_yticks([-1.2, -0.8, -0.4, 0.0], labels=["−1.2", "−0.8", "−0.4", "0"])
ax.axhline(0, color="#93989D", linewidth=0.65, linestyle=(0, (2.3, 2.3)))
ax.set(xlabel="Nominal slots per video", ylabel="ΔSumR")
styles = {8: ("#719AC2", "o"), 16: ("#4F86BD", "s"),
          24: ("#2F6DB5", "^"), 32: ("#174B79", "D")}
handles = []
for clip, (color, marker) in styles.items():
    group = sorted((p for p in rows if p["k_c"] == clip), key=lambda p: p["k_f"])
    ax.plot([p["nominal_slots"] for p in group], [p["mean_delta"] for p in group],
            linewidth=0.9, color=color, zorder=2)
    for p in group:
        assert p["nominal_slots"] == p["k_c"] + p["k_f"]
        ax.plot(p["nominal_slots"], p["mean_delta"], marker=marker, markersize=4.1,
                markeredgewidth=0.88, color=color, linestyle="none", zorder=3,
                markerfacecolor=color if p["configuration"] in frontier else "white")
    handles.append(Line2D([], [], linewidth=0.9, color=color, marker=marker, markersize=3.7,
                          markerfacecolor="white", label=f"{clip} clips"))
ax.legend(handles=handles, loc="lower right", ncol=2, frameon=False, handlelength=1.05,
          columnspacing=0.8, handletextpad=0.35, borderpad=0.1, labelspacing=0.3)

fig.savefig(output / "fig4_budget.pdf", metadata={"Author": ""})
plt.close(fig)
