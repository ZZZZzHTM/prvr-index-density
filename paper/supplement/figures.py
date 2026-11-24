"""Render Figures S1-S4 of the supplementary material from the exported results.

Usage from the repository root:
    python paper/supplement/figures.py --output /path/to/plots
Each figure is written as PDF and PNG. Figure S4 is Figure 6 of the paper drawn by
paper/figures/fig6_quartiles.py at a larger size; its data and content are unchanged.
"""
import json
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures"
sys.path.insert(0, str(FIGURES))
from figure_io import figure_paths  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

R, OUT = figure_paths(__doc__.splitlines()[0])
load = lambda p: json.loads((R / p).read_text(encoding="utf8"))
FONT = 7.2
BLACK, GRAY, TEAL, ORANGE, BLUE = "#25282B", "#62676D", "#1F8A7A", "#D9822B", "#2F6DB5"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FONT,
    "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
    "legend.fontsize": FONT, "axes.linewidth": .5, "axes.unicode_minus": True,
    "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"charades": "Charades", "activitynet": "ActivityNet", "tvr": "TVR", "youcook2": "YouCook2"}
SEEDS = {9527: (BLACK, "o", BLACK), 20261024: (BLUE, "s", "white"), 20261025: (ORANGE, "^", "white")}
log = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={"Author": ""})
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


# ---------- S1: effective-rank quartiles for every seed ----------
seeds = load("rq2_seeds/RESULTS.json")
cells = {(c["dataset"], c["seed"]): c["own_quartiles"]["groups"] for c in seeds["cells"]}
opt = seeds["optional"]; assert opt["dataset"] == "youcook2" and opt["status"] == "COMPLETE"
cells[("youcook2", opt["seed"])] = opt["own_quartiles"]["groups"]
rows = load("rq2_group_ci/RESULTS.json")["rows"]
def ci_rows(ds, model_pred):
    out = {}
    for r in rows:
        if (r["dataset"] == ds and r["branch"] == "fusion" and r["group_kind"] == "effective_rank_quartile"
                and r["method"] == "Ward8" and model_pred(r["model"])):
            q = int(r["group"].split("Q")[-1]); assert q not in out; out[q] = (r["point"], r["ci_low"], r["ci_high"])
    return out
# YouCook2 seed 9527 (centered features) comes from the main-paper Figure 6 analysis
yc = ci_rows("youcook2", lambda m: "our run s9527 (centered)" in m); assert len(yc) == 4
# seed 9527 uses the Figure 6(b) intervals (resampling within each quartile); points must agree
g9527 = {"youcook2": yc}
for ds in ("charades", "activitynet", "tvr"):
    g = ci_rows(ds, lambda m: m.endswith("our run s9527")); assert len(g) == 4, ds
    g9527[ds] = g
    for q in range(1, 5):
        assert abs(g[q][0] - cells[(ds, 9527)][f"Q{q}"]["delta_SumR"]["point"]) < 1e-9, (ds, q)
fig, axes = plt.subplots(1, 4, figsize=(12.2 / 2.54, 4.6 / 2.54), sharex=True)
fig.subplots_adjust(left=.075, right=.995, top=.80, bottom=.2, wspace=.38)
S1 = {}
for ax, ds in zip(axes, ("charades", "activitynet", "tvr", "youcook2")):
    ax.axhline(0, color="#AAB0B5", linestyle=(0, (3, 3)), linewidth=.55, zorder=0)
    series = [s for s in (9527, 20261024, 20261025) if (ds, s) in cells or (ds == "youcook2" and s == 9527)]
    offsets = {s: o for s, o in zip(series, [-.18, 0, .18] if len(series) == 3 else [-.1, .1])}
    for s in series:
        color, marker, face = SEEDS[s]
        for q in range(1, 5):
            if s == 9527:
                p, lo, hi = g9527[ds][q]
            else:
                d = cells[(ds, s)][f"Q{q}"]["delta_SumR"]; p, lo, hi = d["point"], d["ci_low"], d["ci_high"]
            S1[(ds, s, q)] = (p, lo, hi)
            ax.errorbar(q + offsets[s], p, yerr=[[p - lo], [hi - p]], fmt=marker, color=color,
                        markerfacecolor=face, markersize=2.8, markeredgewidth=.6, elinewidth=.6,
                        capsize=1.2, capthick=.6)
    ax.set_title(NAMES[ds], fontsize=FONT, pad=2)
    ax.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"]); ax.set_xlim(.5, 4.5)
    ax.tick_params(length=2, width=.5, pad=2)
axes[0].set_ylabel("ΔSumR", labelpad=2)
handles = [Line2D([], [], marker=m, color=c, markerfacecolor=f, linestyle="none", markersize=3,
                  label=f"seed {s}") for s, (c, m, f) in SEEDS.items()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=3, frameon=False,
           handletextpad=.3, columnspacing=1.2, borderaxespad=0)
fig.text(.535, .02, "lower → higher effective rank", ha="center", va="bottom", fontsize=FONT)
save(fig, "figS1_quartile_seeds")
log["S1"] = {f"{k[0]}|{k[1]}|Q{k[2]}": v for k, v in S1.items()}

# ---------- S2: budget split per dataset ----------
fr = load("budget_split/FRONTIER.json")
pts = fr["points"]
fig, axes = plt.subplots(1, 3, figsize=(12.2 / 2.54, 4.6 / 2.54))
fig.subplots_adjust(left=.075, right=.995, top=.80, bottom=.22, wspace=.33)
styles = {8: ("#719AC2", "o"), 16: ("#4F86BD", "s"), 24: ("#2F6DB5", "^"), 32: ("#174B79", "D")}
S2 = {}
for ax, ds in zip(axes, ("tvr", "activitynet", "charades")):
    vals = [(p["configuration"], p["k_c"], p["k_f"], p["nominal_slots"], p["dataset_deltas"][ds]) for p in pts]
    pareto = set()
    for cfg, kc, kf, n, d in vals:
        dominated = any((n2 <= n and d2 >= d) and (n2 < n or d2 > d) for _, _, _, n2, d2 in vals)
        if not dominated: pareto.add(cfg)
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([12, 24, 40, 80, 160], labels=["12", "24", "40", "80", "160"]); ax.set_xlim(10.9, 177)
    ax.axhline(0, color="#93989D", linewidth=.6, linestyle=(0, (2.3, 2.3)))
    for kc, (color, marker) in styles.items():
        group = sorted((v for v in vals if v[1] == kc), key=lambda v: v[2])
        ax.plot([v[3] for v in group], [v[4] for v in group], linewidth=.8, color=color, zorder=2)
        for cfg, _, kf, n, d in group:
            assert n == kc + kf
            ax.plot(n, d, marker=marker, markersize=3.2, markeredgewidth=.75, color=color, linestyle="none",
                    zorder=3, markerfacecolor=color if cfg in pareto else "white")
            S2[f"{ds}|{cfg}"] = {"slots": n, "delta": d, "pareto": cfg in pareto}
    ax.set_title(NAMES[ds], fontsize=FONT, pad=2); ax.tick_params(length=2, width=.5, pad=2)
    ax.set_xlabel("Nominal slots per video", labelpad=1.5)
axes[0].set_ylabel("ΔSumR", labelpad=2)
handles = [Line2D([], [], linewidth=.8, color=c, marker=m, markersize=3, markerfacecolor="white",
                  label=f"{k} clips") for k, (c, m) in styles.items()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=4, frameon=False,
           handletextpad=.3, columnspacing=1.2, borderaxespad=0)
save(fig, "figS2_budget_per_dataset")
log["S2"] = S2

# ---------- S3: three links of the redistribution chain ----------
ch = load("rq2_tvr_chain/RESULTS.json")
defs = ch["correlation_definitions"]
assert (defs["A"]["x"], defs["A"]["y"]) == ("er", "D")
assert (defs["B"]["x"], defs["B"]["y"]) == ("D", "score_drop")
assert (defs["C"]["x"], defs["C"]["y"]) == ("score_drop", "O_rank")
fig, axes = plt.subplots(1, 3, figsize=(12.2 / 2.54, 4.4 / 2.54), sharey=True)
fig.subplots_adjust(left=.075, right=.995, top=.80, bottom=.16, wspace=.12)
titles = {"A": "A: ER vs. distortion", "B": "B: distortion vs. score drop",
          "C": "C: score drop vs. own rank"}
S3 = {}
order = ("charades", "activitynet", "tvr")
for ax, link in zip(axes, ("A", "B", "C")):
    ax.axhline(0, color="#AAB0B5", linestyle=(0, (3, 3)), linewidth=.55, zorder=0)
    for c in ch["cells"]:
        ds, s = c["dataset"], c["seed"]
        x = order.index(ds) + 1 + {9527: -.2, 20261024: 0, 20261025: .2}[s]
        r = c["correlations"][link]; p = r["point"]; lo, hi = r["CI95"]
        color, marker, face = SEEDS[s]
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt=marker, color=color, markerfacecolor=face,
                    markersize=2.8, markeredgewidth=.6, elinewidth=.6, capsize=1.2, capthick=.6)
        S3[f"{link}|{ds}|{s}"] = (p, lo, hi)
    ax.set_xticks([1, 2, 3], [NAMES[d] for d in order]); ax.set_xlim(.5, 3.5)
    ax.set_title(titles[link], fontsize=FONT, pad=2); ax.tick_params(length=2, width=.5, pad=2)
axes[0].set_ylabel("Spearman ρ", labelpad=2); axes[0].set_ylim(-.1, 1.0)
handles = [Line2D([], [], marker=m, color=c, markerfacecolor=f, linestyle="none", markersize=3,
                  label=f"seed {s}") for s, (c, m, f) in SEEDS.items()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=3, frameon=False,
           handletextpad=.3, columnspacing=1.2, borderaxespad=0)
save(fig, "figS3_chain_links")
log["S3"] = S3
(OUT / "plotted_values.json").write_text(json.dumps(log, indent=1), encoding="utf8")

# ---------- S4: Figure 6 of the paper at a larger size ----------
src = (FIGURES / "fig6_quartiles.py").read_text(encoding="utf8")
for old, new in (("FONT, WIDTH = 7.2, 12.2", "FONT, WIDTH = 8.6, 19.0"),
                 ("height_cm = 5.85", "height_cm = 9.4"),
                 ('fig.savefig(output / "fig6_quartiles.pdf", metadata={"Author": ""})',
                  'fig.savefig(output / "figS4_fig6_large.pdf", metadata={"Author": ""}); '
                  'fig.savefig(output / "figS4_fig6_large.png", dpi=220)')):
    assert src.count(old) == 1, old
    src = src.replace(old, new)
plt.rcdefaults()
exec(compile(src, str(FIGURES / "fig6_quartiles.py"), "exec"), {"__name__": "__main__"})
print("Figures S1-S4 written to", OUT)
