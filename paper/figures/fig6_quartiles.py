"""Plot score changes and budget allocation by effective-rank quartile (Figure 6)."""
from figure_io import figure_paths, load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DATASETS = (("charades", "Charades"), ("activitynet", "ActivityNet"),
            ("tvr", "TVR"), ("youcook2", "YouCook2"))
QUARTILES = range(1, 5)
FONT, WIDTH = 7.2, 12.2
TEAL, ORANGE, GRAY, BLACK = "#1F8A7A", "#D9822B", "#62676D", "#25282B"


results, output = figure_paths(__doc__)
terms = load(results, "rq2_decomposition/RESULTS.json")["decomposition"]["our"]
rows = load(results, "rq2_group_ci/RESULTS.json")["rows"]
split = {(ds, q, term): terms[ds][f"Q{q}"][term]["point"]
         for ds, _ in DATASETS for q in QUARTILES for term in ("O", "C", "F")}
allocation = {}
for row in rows:
    if (row["branch"] == "fusion" and row["group_kind"] == "effective_rank_quartile"
            and row["method"] in ("Ward8", "R1 B8") and "our run" in row["model"]):
        key = row["dataset"], int(row["group"].split("Q")[-1]), row["method"]
        if key in allocation:
            raise ValueError(f"Repeated observation for {key}")
        allocation[key] = row["point"], row["ci_low"], row["ci_high"]
height_cm = 5.85

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": FONT,
                     "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
                     "legend.fontsize": FONT, "axes.linewidth": .5, "axes.unicode_minus": True})
fig = plt.figure(figsize=(WIDTH / 2.54, height_cm / 2.54))
grid = fig.add_gridspec(2, 4, left=.091, right=.992, top=.817, bottom=.152,
                        hspace=.43, wspace=.30)
lowers = []
checked_fixed_markers = 0
for col, (ds, label) in enumerate(DATASETS):
    upper = fig.add_subplot(grid[0, col])
    lower = fig.add_subplot(grid[1, col], sharex=upper, sharey=lowers[0] if lowers else None)
    lowers.append(lower)
    for ax in (upper, lower):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=.5, pad=2)
        ax.set_xlim(.55, 4.45)
        ax.axhline(0, color="#AAB0B5", linestyle=(0, (3, 3)), linewidth=.55, zorder=0)
    upper.tick_params(axis="x", bottom=False, labelbottom=False)
    upper.set_title(label, fontsize=FONT, pad=2)
    upper.bar([q-.16 for q in QUARTILES], [split[ds, q, "O"] for q in QUARTILES],
              width=.32, color=ORANGE, alpha=.75)
    upper.bar([q+.16 for q in QUARTILES], [split[ds, q, "C"] for q in QUARTILES],
              width=.32, color=TEAL, alpha=.75)
    actual, = upper.plot(QUARTILES, [split[ds, q, "F"] for q in QUARTILES], color=BLACK,
                         linewidth=1.4, marker="o", markersize=3.4, zorder=4)
    limit, step = {"charades": (14, 10), "activitynet": (15, 10),
                   "tvr": (6.7, 5), "youcook2": (32, 20)}[ds]
    upper.set_ylim(-limit, limit)
    upper.set_yticks([-step, 0, step])
    for method, offset, color, marker, face in (("Ward8", -.11, BLACK, "o", BLACK),
                                                 ("R1 B8", .11, GRAY, "s", "white")):
        for q in QUARTILES:
            point, low, high = allocation[ds, q, method]
            interval = lower.errorbar(q+offset, point, yerr=[[point-low], [high-point]], fmt=marker,
                                      color=color, markersize=3.1, markerfacecolor=face,
                                      markeredgewidth=.65, elinewidth=.65, capsize=1.4, capthick=.65)
            if method == "Ward8":
                dot, caps, bars = interval.lines
                assert dot.get_ydata()[0] == actual.get_ydata()[q-1]
                assert dot.get_marker() == actual.get_marker() == "o"
                assert len(caps) == 2 and len(bars) == 1
                colors = [actual.get_color(), dot.get_markerfacecolor(), dot.get_markeredgecolor(),
                          *(cap.get_markeredgecolor() for cap in caps), *bars[0].get_colors()]
                assert all(to_hex(c).upper() == BLACK for c in colors)
                checked_fixed_markers += 1
    lower.set_ylim(-7, 7)
    lower.set_yticks([-4, 0, 4])
    lower.set_xticks(list(QUARTILES), [f"Q{q}" for q in QUARTILES])
    if col:
        lower.tick_params(axis="y", left=True, labelleft=False)
    else:
        upper.set_ylabel("ΔSumR", labelpad=2)
        lower.set_ylabel("ΔSumR", labelpad=2)
handles = [Line2D([], [], marker="o", color=BLACK, linewidth=1.4, markersize=3.4,
                      label="both (actual)"),
           Patch(color=ORANGE, alpha=.75, label="positive only"),
           Patch(color=TEAL, alpha=.75, label="competitors only"),
           Line2D([], [], marker="o", color=BLACK, linestyle="none",
                  markersize=3.1, label="fixed k = 8"),
           Line2D([], [], marker="s", color=GRAY, markerfacecolor="white",
                  linestyle="none", markersize=3.1, label="rank-aware (mean 8)")]
legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .981),
                    ncol=len(handles), frameon=False, handlelength=1.05, handletextpad=.36,
                    columnspacing=1.0, borderaxespad=0, borderpad=0)
fixed_legend, = [h for h in legend.legend_handles if h.get_label() == "fixed k = 8"]
assert fixed_legend.get_marker() == "o"
assert to_hex(fixed_legend.get_markerfacecolor()).upper() == to_hex(fixed_legend.get_markeredgecolor()).upper() == BLACK
assert checked_fixed_markers == 16
for label, position in [("(a)", grid[0, 0]), ("(b)", grid[1, 0])]:
    fig.text(.012, position.get_position(fig).y1 + .01, label, ha="left",
             va="bottom", fontsize=FONT)
fig.text(.545, .023, "lower → higher effective rank", ha="center", va="bottom", fontsize=FONT)

fig.savefig(output / "fig6_quartiles.pdf", metadata={"Author": ""})
plt.close(fig)
