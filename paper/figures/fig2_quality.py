"""Plot frame compression at fixed retrieval checkpoints (Figure 2)."""
from figure_io import figure_paths, load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATASETS = [("tvr", "TVR"), ("activitynet", "ActivityNet"), ("charades", "Charades")]
FUSION = "0.7/0.3"
WIDTH_CM, HEIGHT_CM = 12.2, 3.37
FONT = 7.2
TICKS = [36, 40, 48, 64, 96]
SLOT_POSITIONS = {slot: index for index, slot in enumerate((36, 40, 48, 64, 96))}
X_LIMITS = [-.22, 4.16]
Y_LIMITS = [-1.35, .35]
# name, colour, marker, line style, line width, marker size
STYLES = {
    "ward_cos": ("Ward", "#1F8A7A", "o", "-", 1.6, 3.4),
    "uniform": ("Uniform", "#D9822B", "^", "--", 1.3, 3.25),
    "adjacent": ("Adjacent", "#A9AEB3", "s", ":", .7, 2.9),
    "kmeans": ("k-means", "#A9AEB3", "D", "-.", .7, 2.9),
    "farthest": ("Farthest", "#A9AEB3", "v", (0, (4, 2)), .7, 2.9),
    "hybrid": ("Hybrid", "#A9AEB3", "P", (0, (1, 1)), .7, 2.9),
}
results, output = figure_paths(__doc__)
data = load(results, "index_density_stats/RESULTS.json")
baseline_rows = data["branch_symmetry_20261030"]["rows"]
curves = {}
for dataset, _ in DATASETS:
    baseline_index = next(i for i, row in enumerate(baseline_rows)
                          if row["dataset"] == dataset and row["variant"] == "C32_F128"
                          and row["fusion"] == FUSION)
    clip_slots = baseline_rows[baseline_index]["clip_target"]
    curves[dataset] = {}
    for family in STYLES:
        selected = [(i, row) for i, row in enumerate(data["rows"])
                    if row["dataset"] == dataset and row["fusion"] == FUSION and row["family"] == family]
        assert selected and len({row["k"] for _, row in selected}) == len(selected)
        points = []
        for index, row in sorted(selected, key=lambda pair: pair[1]["k"]):
            frame_budget = row["k"]
            delta = row["delta_vs_B0_07_03"]
            points.append(dict(nominal_slots=clip_slots + frame_budget, delta=delta,
                               frame_budget=frame_budget, source_row=index, variant=row["variant"]))
        curves[dataset][family] = points

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
    "legend.fontsize": FONT, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .5,
    "savefig.dpi": 300, "axes.unicode_minus": True,
})
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54))
fig.subplots_adjust(left=.085, right=.982, bottom=.255, top=.715, wspace=.21)
handles = [Line2D([], [], label=name, color=colour, marker=marker, ls=style, lw=width,
                  markersize=size, markeredgewidth=.45)
           for name, colour, marker, style, width, size in STYLES.values()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .972), ncol=6,
           frameon=False, borderpad=0, borderaxespad=0, columnspacing=1.15,
           handlelength=1.45, handletextpad=.35)
for ax, (dataset, label) in zip(axes, DATASETS):
    ax.axhline(0, color="#B0B4B8", lw=.52, ls=(0, (3, 3)), zorder=0)
    ax.axvline(SLOT_POSITIONS[40], color="#E6E8EA", lw=.58, zorder=0)
    for family, (_, colour, marker, style, width, size) in STYLES.items():
        points = curves[dataset][family]
        ax.plot([SLOT_POSITIONS[p["nominal_slots"]] for p in points], [p["delta"] for p in points],
                color=colour, marker=marker, ls=style, lw=width, markersize=size, gid=family,
                markeredgewidth=.45, clip_on=True,
                zorder=5 if family == "ward_cos" else 4 if family == "uniform" else 2)
        for point in points:
            value = point["delta"]
            assert value <= Y_LIMITS[1], (dataset, family, point)
            if value >= Y_LIMITS[0]:
                continue
            x = SLOT_POSITIONS[point["nominal_slots"]]
            # An inward-positioned arrow points down toward the omitted value at this budget.
            pointer = ax.annotate("", xy=(x, Y_LIMITS[0]), xytext=(0, 13.4), textcoords="offset points",
                                  arrowprops=dict(arrowstyle="-|>", color=colour, lw=.85, mutation_scale=7.2,
                                                  shrinkA=0, shrinkB=1.4), zorder=6)
            formatted = format(value, ".2f").replace("-", "−")
            label_offset = (-1.6, 15.8)
            value_label = ax.annotate(formatted, xy=(x, Y_LIMITS[0]), xytext=label_offset,
                                      textcoords="offset points", ha="left", va="bottom", color=colour, zorder=6)
    ax.set_xlim(*X_LIMITS)
    ax.set_xticks([SLOT_POSITIONS[v] for v in TICKS], TICKS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([-1, -.5, 0], ["−1", "−0.5", "0"])
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#5D6267")
    ax.tick_params(length=2, pad=1.6, width=.5)
    ax.text(.5, 1.035, label, ha="center", va="bottom", transform=ax.transAxes)
axes[0].set_ylabel("ΔSumR", labelpad=2)
fig.text(.535, .023, "Nominal slots per video (clip + frame)", ha="center", va="bottom")

fig.savefig(output / "fig2_quality.pdf", metadata={"Author": ""})
plt.close(fig)
