#!/usr/bin/env python3
"""
UMAP clustering on both representation layers:

  MEME LAYER   — style_profiles/style_YEAR.json (numeric features)
  MODEL LAYER  — embeddings/year_YEAR.npy (384-dim pooled vectors)
                 OR concept frequency histograms from checkpoints/

Outputs a single TSV:
  year, layer, umap_x, umap_y, <feature columns>

Then plot in 2D:
  - UMAP model coordinates show concept-space drift over time
  - UMAP meme coordinates show style-space drift
  - same-year pairs between layers reveal decoupling or coupling
"""
import json, csv, math, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import umap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STYLE_DIR = DATA / "style_profiles"
EMB_DIR = DATA / "embeddings"
CKPT_DIR = DATA / "checkpoints"
OUT_PATH = DATA / "umap_clusters.tsv"

# Features to extract from style profiles (in fixed order)
STYLE_KEYS = [
    "compression_total",
    "meme_vocab_hits",
    "meme_vocab_density",
    "analogy_density",
    "parenthetical_density",
    "spread_score",
    # readability
    "avg_sentence_len",
    "avg_char_per_word",
    "pct_questions",
    "pct_first_person",
    "pct_math_sentences",
    # compression category breakdown (7)
    "compression_framing",
    "numbered_structure",
    "X_vs_y_frame",
    "possible_futures",
    "self_ref",
    "demystification",
]
# All keys extracted from compression dict
COMPRESSION_CATS = [
    "compression_framing",
    "numbered_structure",
    "X_vs_y_frame",
    "possible_futures",
    "self_ref",
    "demystification",
]


def load_style_vectors() -> tuple[np.ndarray, list[int], list[str]]:
    """Return (matrix, years, features) from style profiles."""
    rows, years = [], []
    for p in sorted(STYLE_DIR.glob("style_*.json")):
        y = int(p.stem.split("_")[1])
        data = json.loads(p.read_text())
        row = []
        for k in STYLE_KEYS:
            v = data.get(k, 0.0)
            if isinstance(v, dict):
                v = sum(v.values())
            row.append(float(v))
        # Compression subcats
        comp = data.get("compression", {})
        for cat in COMPRESSION_CATS:
            row.append(float(comp.get(cat, 0)))
        rows.append(row)
        years.append(y)

    feat_names = STYLE_KEYS + COMPRESSION_CATS
    return np.array(rows, dtype=float), years, feat_names


def load_embedding_vectors() -> tuple[np.ndarray, list[int]]:
    """Return (matrix, years) from cached year embeddings."""
    rows, years = [], []
    for p in sorted(EMB_DIR.glob("year_*.npy")):
        y = int(p.stem.split("_")[1])
        rows.append(np.load(p))
        years.append(y)
    if rows:
        return np.vstack(rows), years
    return np.empty((0, 384)), []


def build_combined(
    n_neighbors: int = 3,
    min_dist: float = 0.3,
    metric: str = "cosine",
):
    # Load layers
    style_mat, style_years, style_feats = load_style_vectors()
    emb_mat, emb_years = load_embedding_vectors()

    if len(style_years) == 0 and len(emb_years) == 0:
        print("No data found.")
        return

    reducer_style, reducer_emb = None, None

    with open(OUT_PATH, "w", newline="") as csvf:
        writer = csv.writer(csvf, delimiter="\t")
        headers = ["year", "layer", "umap_x", "umap_y"]
        if style_feats:
            headers += [f"style__{f}" for f in style_feats]
        if emb_mat.shape[1] > 0:
            dim = emb_mat.shape[1]
            headers += [f"emb__d{i}" for i in range(dim)]
        writer.writerow(headers)

        objs = []

        # MEME LAYER
        if len(style_years) > 0:
            reducer_style = umap.UMAP(
                n_neighbors=min(n_neighbors, max(2, len(style_years) - 1)),
                min_dist=min_dist,
                metric="cosine",
                n_components=2,
                random_state=42,
            )
            # Normalize style features: min-max per feature
            smin, smax = style_mat.min(axis=0), style_mat.max(axis=0)
            denom = smax - smin
            denom[denom == 0] = 1.0
            s_norm = (style_mat - smin) / denom
            s_coords = reducer_style.fit_transform(s_norm)
            print(f"[style UMAP] n={len(style_years)}, neighbors={reducer_style.n_neighbors}, min_dist={reducer_style.min_dist}")
            objs.append(("style", style_years, s_coords, style_mat))

        # MODEL LAYER
        if emb_mat.shape[0] > 0:
            reducer_emb = umap.UMAP(
                n_neighbors=min(n_neighbors, max(2, emb_mat.shape[0] - 1)),
                min_dist=min_dist,
                metric=metric,
                n_components=2,
                random_state=42,
            )
            e_coords = reducer_emb.fit_transform(emb_mat)
            print(f"[embed UMAP] n={emb_mat.shape[0]}, neighbors={reducer_emb.n_neighbors}, min_dist={reducer_emb.min_dist}")
            objs.append(("embed", emb_years, e_coords, emb_mat))

        # Write rows
        for layer, years, coords, raw_mat in objs:
            for idx, year in enumerate(years):
                x, y = coords[idx]
                row = [year, layer, round(float(x), 6), round(float(y), 6)]
                if layer == "style":
                    row += [round(float(v), 4) for v in raw_mat[idx]]
                else:
                    row += [round(float(v), 6) for v in raw_mat[idx]]
                writer.writerow(row)

    print(f"\nSaved {OUT_PATH}")
    return objs


def describe(obj):
    """Print per-year positions for inspection."""
    for layer, years, coords, raw_mat in obj:
        print(f"\n--- {layer.upper()} ---")
        for y, (x, yy) in sorted(zip(years, coords.tolist())):
            print(f"  {y}: ({x:.4f}, {yy:.4f})")

    # Pairwise distances per layer
    for layer, years, coords, raw_mat in obj:
        if len(coords) < 3:
            continue
        from sklearn.metrics import pairwise_distances
        d = pairwise_distances(coords, metric="euclidean")
        print(f"\n{layer}: mean pairwise distance = {d[np.triu_indices(len(d), k=1)].mean():.4f}")
        # nearest-neighbor chain (chronological neighbors)
        chain = []
        for i in range(1, len(years)):
            chain.append(d[i, i - 1])
        print(f"  chronological neighbor distances: {[f'{v:.4f}' for v in chain]}")

    # Cross-layer same-year distances (style vs embed)
    if len(obj) == 2:
        (_, y_style, c_style, _), (_, y_emb, c_emb, _) = obj
        s_dict = {y: c for y, c in zip(y_style, c_style)}
        e_dict = {y: c for y, c in zip(y_emb, c_emb)}
        common = sorted(set(s_dict) & set(e_dict))
        if common:
            from sklearn.metrics import pairwise_distances
            sv = np.vstack([s_dict[y] for y in common])
            ev = np.vstack([e_dict[y] for y in common])
            d_cross = pairwise_distances(sv, ev, metric="euclidean").diagonal()
            print(f"\ncross-layer same-year distances: {[f'{y}: {v:.4f}' for y, v in zip(common, d_cross)]}")
            print(f"  mean cross-layer distance = {d_cross.mean():.4f}")

    # Light clustering summary per layer
    k = min(3, len(years))
    from sklearn.cluster import KMeans
    for layer, years, coords, raw_mat in obj:
        if len(coords) < 3:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(coords)
        print(f"\n{layer} KMeans(k={k}):")
        for cid in range(k):
            members = [y for y, l in zip(years, labels) if l == cid]
            print(f"  cluster {cid}: {members}")


if __name__ == "__main__":
    nn = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    md = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    metric = sys.argv[3] if len(sys.argv) > 3 else "cosine"
    print(f"Config: n_neighbors={nn}, min_dist={md}, metric={metric}\n")
    objs = build_combined(n_neighbors=nn, min_dist=md, metric=metric)
    if objs:
        describe(objs)
