#!/usr/bin/env python3
"""
Drift + trajectory-distance analysis.

Outputs:
  data/drift_by_year.tsv   — per-layer step length per year, drift_ratio
  data/probe_trajectory.tsv — probe projected into each UMAP space,
                               nearest-year match + distance-to-trajectory

Usage:
  python3 src/drift.py                          # drift report only
  python3 src/drift.py "text to probe"         # drift + probe
  python3 src/drift.py path/to/file.txt        # same, file input
"""
import argparse, csv, json, math, sys, hashlib
from pathlib import Path

import numpy as np
import umap
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TSV = DATA / "umap_clusters.tsv"
DRIFT_PATH = DATA / "drift_by_year.tsv"
PROBE_PATH = DATA / "probe_trajectory.tsv"
EMB_DIR = DATA / "embeddings"
RAW = DATA / "raw" / "vitalik_corpus.jsonl"

EMB_MODEL = "all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL)
    return _model


# ── Load UMAP coordinates ──────────────────────────────────────────────────

def load_umap():
    rows = list(csv.DictReader(open(TSV), delimiter="\t"))
    style = {int(r["year"]): (float(r["umap_x"]), float(r["umap_y"])) for r in rows if r["layer"] == "style"}
    embed = {int(r["year"]): (float(r["umap_x"]), float(r["umap_y"])) for r in rows if r["layer"] == "embed"}
    return style, embed


# ── Drift: year-on-year step length ───────────────────────────────────────

def compute_drift(style_coords, embed_coords):
    years_style = sorted(style_coords)
    years_embed = sorted(embed_coords)

    def steps(coords, years):
        out = []
        for i in range(1, len(years)):
            y0, y1 = years[i - 1], years[i]
            x0, a = coords[y0]
            x1, b = coords[y1]
            out.append((y0, y1, math.hypot(x1 - x0, b - a)))
        return out

    s_steps = steps(style_coords, years_style)
    e_steps = steps(embed_coords, years_embed)

    s_dict = {y1: s for _, y1, s in s_steps}
    e_dict = {y1: s for _, y1, s in e_steps}
    common = sorted(set(s_dict) & set(e_dict))

    rows = []
    for y in common:
        s = s_dict[y]
        e = e_dict[y]
        ratio = round(s / e, 4) if e > 1e-6 else None
        rows.append({"year": y, "meme_step": round(s, 4), "model_step": round(e, 4),
                     "drift_ratio_meme_model": ratio})

    with open(DRIFT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "meme_step", "model_step", "drift_ratio_meme_model"])
        w.writeheader()
        w.writerows(rows)
    return rows


# ── Cached projectors ─────────────────────────────────────────────────────

_projector_cache = None


def build_projectors(style_coords, embed_coords):
    """Fit UMAP reducers on annual aggregates. Cached within process."""
    global _projector_cache
    if _projector_cache is not None:
        return _projector_cache

    years = sorted(embed_coords)
    year_vecs = {}
    for p in sorted(EMB_DIR.glob("year_*.npy")):
        y = int(p.stem.split("_")[1])
        if y in years:
            year_vecs[y] = np.load(p)

    if not year_vecs:
        return None

    # Embed projector
    mat = np.vstack([year_vecs[y] for y in years])
    reducer_embed = umap.UMAP(
        n_neighbors=min(3, len(years) - 1),
        min_dist=0.3, metric="cosine", n_components=2, random_state=42,
    )
    reducer_embed.fit(mat)

    # Style projector
    reducer_style = None
    style_keys = [
        "compression_total", "meme_vocab_hits", "meme_vocab_density",
        "analogy_density", "parenthetical_density", "spread_score",
        "avg_sentence_len", "avg_char_per_word", "pct_questions",
        "pct_first_person", "pct_math_sentences",
        "compression_framing", "numbered_structure", "X_vs_y_frame",
        "possible_futures", "self_ref", "demystification",
    ]
    comp_cats = [
        "compression_framing", "numbered_structure", "X_vs_y_frame",
        "possible_futures", "self_ref", "demystification",
    ]

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from style import extract, STYLE_DIR

    full_rows = []
    for p in sorted(STYLE_DIR.glob("style_*.json")):
        data = json.loads(p.read_text())
        y = data["year"]
        if y not in years:
            continue
        row = [float(data.get(k, 0) or 0) for k in style_keys]
        comp = data.get("compression", {})
        for c in comp_cats:
            row.append(float(comp.get(c, 0)))
        full_rows.append(row)

    if full_rows:
        s_mat = np.array(full_rows, dtype=float)
        smin, smax = s_mat.min(axis=0), s_mat.max(axis=0)
        denom = smax - smin
        denom[denom == 0] = 1.0
        s_norm = (s_mat - smin) / denom
        reducer_style = umap.UMAP(
            n_neighbors=min(3, len(years) - 1),
            min_dist=0.3, metric="cosine", n_components=2, random_state=42,
        )
        reducer_style.fit(s_norm)

    _projector_cache = (reducer_embed, reducer_style, year_vecs,
                        smin if full_rows else None, denom if full_rows else None,
                        style_keys, comp_cats)
    return _projector_cache


def nearest_on_trajectory(probe_coord, traj_coords: dict):
    """Distance from probe to nearest year point and nearest segment of the trajectory."""
    years = sorted(traj_coords)
    best = {"year": None, "seg_from": None, "seg_to": None,
            "dist_to_year": float("inf"), "dist_to_segment": float("inf"),
            "era": None}
    for i, y in enumerate(years):
        px, py = traj_coords[y]
        d = math.hypot(probe_coord[0] - px, probe_coord[1] - py)
        if d < best["dist_to_year"]:
            best["dist_to_year"] = d
            best["year"] = y
            best["era"] = "early" if y <= 2018 else "mid" if y <= 2021 else "late"
        if i > 0:
            y0, y1 = years[i - 1], years[i]
            x0, a = traj_coords[y0]
            x1, b = traj_coords[y1]
            dx, dy = x1 - x0, b - a
            seg_len2 = dx * dx + dy * dy
            if seg_len2 == 0:
                ds = math.hypot(probe_coord[0] - x0, probe_coord[1] - a)
            else:
                t = max(0, min(1,
                    ((probe_coord[0] - x0) * dx + (probe_coord[1] - a) * dy) / seg_len2))
                ds = math.hypot(probe_coord[0] - (x0 + t * dx),
                                probe_coord[1] - (a + t * dy))
            if ds < best["dist_to_segment"]:
                best["dist_to_segment"] = ds
                best["seg_from"] = y0
                best["seg_to"] = y1
    return best


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", nargs="?", default=None)
    args = parser.parse_args()

    print("Loading UMAP coordinates …")
    style, embed = load_umap()

    print("\nDrift by year …")
    drift_rows = compute_drift(style, embed)
    for r in drift_rows:
        print(f"  {r['year']}: meme_step={r['meme_step']:.4f}  "
              f"model_step={r['model_step']:.4f}  "
              f"ratio={r['drift_ratio_meme_model']}")
    print(f"Saved {DRIFT_PATH}")

    if not args.probe:
        sys.exit(0)

    # Probe mode
    p = Path(args.probe)
    probe_text = p.read_text() if p.exists() else args.probe
    print(f"\nProbe: {'file ' + str(p) if p.exists() else probe_text[:80]}")
    print(f"  chars={len(probe_text)}")

    print("Building/fitting projectors …")
    result = build_projectors(style, embed)
    if result is None:
        print("No year vectors found."); sys.exit(1)
    (reducer_embed, reducer_style, year_vecs,
     smin, sdenom, style_keys, comp_cats) = result

    # Embed probe
    model = get_model()
    sents = [s.strip() for s in probe_text.split(".") if len(s.strip()) > 20]
    if not sents:
        sents = [probe_text[:500]]
    pv = np.mean(model.encode(sents, show_progress_bar=False), axis=0)
    pv2d_embed = reducer_embed.transform(pv.reshape(1, -1))[0]

    # Style probe
    sys.path.insert(0, str(ROOT / "src"))
    from style import extract
    pv2d_style = None
    if reducer_style is not None:
        fx = extract(probe_text)
        row = [float(fx.get(k, 0) or 0) for k in style_keys]
        comp = fx.get("compression", {})
        for c in comp_cats:
            row.append(float(comp.get(c, 0)))
        p_norm = (np.array([row], dtype=float) - smin) / sdenom
        pv2d_style = reducer_style.transform(p_norm)[0]

    print("\n=== Probe trajectory report ===")
    results = []
    for layer, pv2d in [("embed", pv2d_embed), ("style", pv2d_style)]:
        if pv2d is None:
            continue
        traj = embed if layer == "embed" else style
        b = nearest_on_trajectory(pv2d, traj)
        results.append({
            "layer": layer, "probe_x": round(float(pv2d[0]), 4),
            "probe_y": round(float(pv2d[1]), 4),
            "nearest_year": b["year"], "era": b["era"],
            "seg_from": b["seg_from"], "seg_to": b["seg_to"],
            "dist_to_year": round(b["dist_to_year"], 4),
            "dist_to_segment": round(b["dist_to_segment"], 4),
        })
        print(f"\n{layer.upper()}:")
        print(f"  position:      ({pv2d[0]:.4f}, {pv2d[1]:.4f})")
        print(f"  nearest year:  {b['year']} ({b['era']} era)")
        print(f"  dist to year:  {b['dist_to_year']:.4f}")
        print(f"  nearest seg:   {b['seg_from']}–{b['seg_to']}")
        print(f"  dist to seg:   {b['dist_to_segment']:.4f}")

    with open(PROBE_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "layer", "probe_x", "probe_y", "nearest_year", "era",
            "seg_from", "seg_to", "dist_to_year", "dist_to_segment",
        ])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {PROBE_PATH}")
