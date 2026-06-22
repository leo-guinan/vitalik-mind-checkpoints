#!/usr/bin/env python3
"""
Constellation plot: UMAP style vs embed projections, same-year pairs linked
by a polarity line. Distance = cross-layer drift for that year.
"""
import csv, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TSV = DATA / "umap_clusters.tsv"
OUT = DATA / "constellation.png"

# Parse TSV
rows = list(csv.DictReader(open(TSV), delimiter="\t"))
style_rows = [r for r in rows if r["layer"] == "style"]
embed_rows = [r for r in rows if r["layer"] == "embed"]
s_map = {int(r["year"]): r for r in style_rows}
e_map = {int(r["year"]): r for r in embed_rows}
common = sorted(set(s_map) & set(e_map))

# Figure layout: main constellation + polarity inset
fig = plt.figure(figsize=(10, 8), dpi=150)
fig.patch.set_facecolor("#0b0c10")
ax = fig.add_axes([0.08, 0.08, 0.58, 0.82])
ax.set_facecolor("#0b0c10")
ax2 = fig.add_axes([0.72, 0.55, 0.25, 0.35])
ax2.set_facecolor("#0b0c10")

year_min = min(common)
year_max = max(common)
cmap = plt.cm.plasma
norm = matplotlib.colors.Normalize(vmin=year_min, vmax=year_max)

# ── Main constellation ──────────────────────────────────────────────────────
for y in common:
    s = s_map[y]
    e = e_map[y]
    sx, sy = float(s["umap_x"]), float(s["umap_y"])
    ex, ey = float(e["umap_x"]), float(e["umap_y"])
    c = cmap(norm(y))

    # Points
    ax.plot(sx, sy, "o", color=c, ms=9, zorder=3)
    ax.plot(ex, ey, "s", color=c, ms=9, zorder=3)

    # Polarity line (same-year bridge)
    dist = math.hypot(ex - sx, ey - sy)
    alpha = min(1.0, max(0.15, dist / 15.0))
    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-", color=c, alpha=alpha, lw=1.2,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )

    # Year label near midpoint
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    ax.text(
        mx, my, str(y),
        color="white", fontsize=6.5, ha="center", va="center",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        zorder=5,
    )

# Axis cosmetics
ax.set_xlabel("UMAP-1", color="#c5c6c7", fontsize=9)
ax.set_ylabel("UMAP-2", color="#c5c6c7", fontsize=9)
ax.tick_params(colors="#c5c6c7", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.set_title(
    "Vitalik mental model constellation\nmeme layer (○) vs model layer (□)",
    color="white", fontsize=11, pad=10,
)

# Legend
legend_handles = [
    Line2D([0], [0], marker="o", color="#66fcf1", label="meme (style)", linestyle=""),
    Line2D([0], [0], marker="s", color="#66fcf1", label="model (embed)", linestyle=""),
    Line2D([0], [0], color="#c5c6c7", lw=1.5, label="polarity bridge"),
]
ax.legend(handles=legend_handles, loc="lower right", framealpha=0.2,
          facecolor="#0b0c10", edgecolor="#333", labelcolor="#c5c6c7", fontsize=7)

# ── Polarity over time ──────────────────────────────────────────────────────
distances = []
for y in common:
    s = s_map[y]
    e = e_map[y]
    d = math.hypot(float(e["umap_x"]) - float(s["umap_x"]),
                   float(e["umap_y"]) - float(s["umap_y"]))
    distances.append((y, d))

years_d, dists = zip(*distances)
ax2.plot(years_d, dists, color="#45a29e", lw=1.5, zorder=3)
ax2.fill_between(years_d, dists, alpha=0.15, color="#45a29e")
ax2.scatter(years_d, dists, color="#66fcf1", s=28, zorder=4)

# Color scatter by year
for y, d in zip(years_d, dists):
    ax2.scatter([y], [d], color=cmap(norm(y)), s=40, zorder=5)

ax2.set_xlabel("year", color="#c5c6c7", fontsize=7)
ax2.set_ylabel("polarity", color="#c5c6c7", fontsize=7)
ax2.tick_params(colors="#c5c6c7", labelsize=6)
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.set_title("cross-layer drift", color="white", fontsize=8, pad=4)

# Annotate peaks
peak_y, peak_d = max(distances, key=lambda x: x[1])
ax2.annotate(
    f"peak {peak_y}\n{peak_d:.1f}",
    xy=(peak_y, peak_d),
    xytext=(peak_y - 2, peak_d + 1),
    color="#f56565", fontsize=6.5,
    arrowprops=dict(arrowstyle="->", color="#f56565", lw=0.8),
)

fig.savefig(OUT, facecolor=fig.get_facecolor(), dpi=150)
print(f"Saved {OUT}")
