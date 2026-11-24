"""Plot compression across six retrieval methods (Figure 3)."""
from figure_io import figure_paths, load
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

WIDTH_CM, HEIGHT_CM, FONT = 12.2, 4.5, 7.2
models = [
    ('B0 (ours)', 'GMMFormerV2 (our run)'),
    ('GMMFormerV2 (author)', 'GMMFormerV2 (released)'),
    ('DreamPRVR (ours)', 'DreamPRVR (our run)'),
    ('DreamPRVR (author)', 'DreamPRVR (released)'),
    ('MS-SL', 'MS-SL'), ('HLFormer', 'HLFormer'), ('Holmes', 'Holmes'), ('SDM', 'SDM')]
datasets = [('tvr','TVR'), ('activitynet','ActivityNet'), ('charades','Charades'), ('youcook2','YouCook2')]
results, output = figure_paths(__doc__)
assets = load(results, "generality_six_methods/RESULTS.json")
centered = load(results, "protocol/youcook2_centered/RESULTS.json")
plot_values = {}
for row in assets["generality"]:
    for budget in ("Ward8", "Ward16"):
        plot_values[row["model"], row["dataset"], budget] = row[budget]["value"]
for budget, record_index in (("Ward8", 10), ("Ward16", 12)):
    original = centered["analysis"]["9527"]["records"][record_index]["paired_vs_uncompressed"]["delta_SumR"]
    assert plot_values["B0 (ours)", "youcook2", budget] == original
shown_datasets = datasets[:3]

plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral','DejaVu Serif'],
                     'mathtext.fontset':'stix', 'font.size':FONT, 'pdf.fonttype':42,
                     'ps.fonttype':42, 'axes.linewidth':.5, 'savefig.dpi':300})
cmap = LinearSegmentedColormap.from_list('signed_change', ['#D9822B', '#FFFFFF', '#1F8A7A'], N=257)
norm = Normalize(vmin=-2, vmax=2, clip=True)
assert cmap(norm(0)) == (1., 1., 1., 1.)
fig = plt.figure(figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54))
ax = fig.add_axes([.348, .035, .514, .875])
column_count = len(shown_datasets)
ax.set_xlim(0, column_count)
ax.set_ylim(len(models), 0)

# Place the budget labels in the gutter, aligned with each upper and lower half.
for row_id, (model, label) in enumerate(models):
    for half, budget in enumerate(('8', '16')):
        ax.text(-.116, row_id+.25+half*.5, budget, ha='center', va='center',
                color='#62696E', fontsize=FONT, clip_on=False)
    for col_id, (dataset, _) in enumerate(shown_datasets):
        pair = [plot_values[model,dataset,b] for b in ('Ward8','Ward16')]
        if pair == [None, None]:
            ax.add_patch(Rectangle((col_id, row_id), 1, 1, facecolor='#F0F1F2',
                                   edgecolor='#BCC1C6', hatch='////', linewidth=.3))
            ax.text(col_id+.5, row_id+.5, 'NA', ha='center', va='center', fontsize=7.5,
                    color='#555B62', bbox=dict(facecolor='#F0F1F2', edgecolor='none', pad=.35))
            continue
        for half, value in enumerate(pair):
            assert value is not None
            colour = cmap(norm(value))
            ax.add_patch(Rectangle((col_id,row_id+half*.5), 1, .5, facecolor=colour,
                                   edgecolor='white', linewidth=.25))
            label_value = ('0.00' if abs(value)<.005 else f'{value:+.2f}').replace('-', '\u2212')
            ax.text(col_id+.5, row_id+.25+half*.5, label_value, ha='center', va='center',
                    fontsize=7.5, color='#272B2E')
    ax.plot([0,column_count], [row_id,row_id], color='#D8DCE0', lw=.3)
ax.set_xticks([i+.5 for i in range(column_count)], [name for _,name in shown_datasets], fontsize=7.8)
ax.xaxis.tick_top()
ax.set_yticks([i+.5 for i in range(len(models))], [label for _,label in models], fontsize=7.5)
ax.tick_params(axis='x', length=0, pad=3)
ax.tick_params(axis='y', length=0, pad=16)
for spine in ax.spines.values():
    spine.set_edgecolor('#BFC5CB')
cax = fig.add_axes([.895, .15, .020, .65])
bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), cax=cax,
                   ticks=[-2,-1,0,1,2], extend='both', extendfrac=.045)
bar.ax.tick_params(labelsize=FONT, length=2, pad=1.7, width=.5)
bar.outline.set_linewidth(.4)
bar.ax.set_ylabel('\u0394SumR', fontsize=7.5, labelpad=3)

fig.savefig(output / "fig3_generality.pdf", metadata={"Author": ""})
plt.close(fig)
