"""
viz_dashboard.py — single-page visualization of pseudonym scoring state.
Panels:
  1. top20 candidates: ethresearched + reddit proximity bars by era-color
  2. score distribution: histogram of proximity for both corpora
  3. year hit-count: stacked bar by era for top-50 from each corpus
  4. constellation inset: re-use existing PNG with axis labels simplified
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'
OUT = ROOT / 'data' / 'pseudonym_dashboard.png'
OUT.parent.mkdir(exist_ok=True)

era_color = {
    y: ('#4e79a7', 'pre-merge') if y <= 2020 else ('#59a14f', 'post-merge') if y <= 2022 else ('#e15759', 'd/acc-era')
    for y in range(2014, 2030)
}

def load_top(path, n=50):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r['proximity'], reverse=True)
    return rows[:n]

eth = load_top(PSEUDO / 'ethresearched_clean_384_scored.jsonl')
red = load_top(PSEUDO / 'candidates_clean_384_scored.jsonl')

fig = plt.figure(figsize=(14, 10), dpi=160)
fig.suptitle('Vitalik Mind Checkpoints — Pseudonym Scoring State', fontsize=14, fontweight='bold')

# --- Panel 1: top20 proximity by era ---
ax1 = fig.add_subplot(2, 2, 1)
y_pos = 0
yticks, yticklabels = [], []
markers = []
for corpus_name, rows, marker in [('ethresearched', eth, 's'), ('reddit', red, 'o')]:
    for i, r in enumerate(rows[:20], 1):
        clr, lbl = era_color[r['nearest_year']]
        ax1.barh(y_pos, r['proximity'], color=clr, height=0.6, alpha=0.9 if marker=='s' else 0.4)
        if i <= 5:
            markers.append((clr, lbl, r['nearest_year'], r['title'][:45], r['proximity']))
        y_pos += 1
    yticks.append(np.median(range(y_pos-20, y_pos)))
    yticklabels.append(corpus_name)
    ax1.axhline(y_pos-0.5, color='k', alpha=0.08)
    y_pos += 1

ax1.set_yticks(yticks)
ax1.set_yticklabels(yticklabels)
ax1.set_xlabel('proximity')
ax1.set_title('Top-20 candidates by corpus')
for clr, lbl, year, title, val in markers:
    ax1.text(val+0.002, yticks[0]-19, f'{year}·{title}', color=clr, va='center', fontsize=6)

# --- Panel 2: distribution of proximity ---
ax2 = fig.add_subplot(2, 2, 2)
colors = ['#59a14f', '#4e79a7']
for rows, c, label in [(eth, colors[1], 'ethresearched'), (red, colors[0], 'reddit')]:
    vals = [r['proximity'] for r in rows]
    ax2.hist(vals, bins=20, alpha=0.55, color=c, label=label, density=False)
ax2.set_xlabel('proximity')
ax2.set_ylabel('count in top-200')
ax2.set_title('Score distribution (top-200 docs)')
ax2.legend()

# --- Panel 3: era hit-count in top-50 ---
ax3 = fig.add_subplot(2, 2, 3)
eras = ['pre-merge\n(≤2020)', 'post-merge\n(2021-2022)', 'd/acc-era\n(2023+)']
eth_counts = [0,0,0]
red_counts = [0,0,0]
for r in eth[:50]:
    eidx = 0 if r['nearest_year']<=2020 else 1 if r['nearest_year']<=2022 else 2
    eth_counts[eidx] += 1
for r in red[:50]:
    eidx = 0 if r['nearest_year']<=2020 else 1 if r['nearest_year']<=2022 else 2
    red_counts[eidx] += 1
x = np.arange(len(eras))
ax3.bar(x-0.18, red_counts, 0.35, color='#59a14f', alpha=0.75, label='reddit')
ax3.bar(x+0.18, eth_counts, 0.35, color='#4e79a7', alpha=0.75, label='ethresearched')
ax3.set_xticks(x)
ax3.set_xticklabels(eras)
ax3.set_ylabel('hits in top-50')
ax3.set_title('Era concentration')
ax3.legend()

# --- Panel 4: constellation snapshot ---
ax4 = fig.add_subplot(2, 2, 4)
img_path = ROOT / 'data' / 'constellation.png'
if img_path.exists():
    img = plt.imread(str(img_path))
    ax4.imshow(img)
    ax4.axis('off')
    ax4.set_title('Constellation (style vs embed)')
else:
    ax4.text(0.5, 0.5, 'constellation.png missing', ha='center', va='center')
    ax4.axis('off')

# --- Legend by era (single legend for whole figure) ---
legend_patches = []
for y in [2016, 2017, 2020, 2021, 2022, 2023, 2026]:
    clr, lbl = era_color[y]
    legend_patches.append(mpatches.Patch(color=clr, label=f'{y} ({lbl})'))
fig.legend(handles=legend_patches, loc='lower center', ncol=7, fontsize=8, frameon=False)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(str(OUT), bbox_inches='tight')
plt.close()
print('wrote', OUT)
