#!/usr/bin/env python3
"""
Stream 3 — Embeddings & probe comparison.

For each year checkpoint + each individual doc:
  * Build sentence-level embeddings via sentence-transformers
  * Pool to a single 384-dim vector
  * Cache to data/embeddings/

Comparison:
  * Any probe text → cosine similarity to each year / each doc
  * Returns ranked year + ranked nearest doc
"""
import json, hashlib, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EMB_DIR = DATA / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Return single pooled vector for a text block."""
    model = get_model()
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
    if not sentences:
        sentences = [text[:500]]
    vecs = model.encode(sentences, show_progress_bar=False)
    return np.mean(vecs, axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def year_cache_path(year: int) -> Path:
    return EMB_DIR / f"year_{year}.npy"


def doc_cache_path(doc_hash: str) -> Path:
    return EMB_DIR / f"doc_{doc_hash[:12]}.npy"


def build_year_vectors(corpus_path: Path = None) -> dict[int, np.ndarray]:
    """Pool each year's docs into one vector. Cache to disk."""
    if corpus_path is None:
        corpus_path = ROOT / "data" / "raw" / "vitalik_corpus.jsonl"

    corpora: dict[int, list[str]] = defaultdict(list)
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            date_str = row.get("date", "")
            try:
                y = int(date_str[:4])
            except ValueError:
                continue
            if y < 2010 or y > 2030:  # sanity filter
                continue
            corpora[y].append(row["text"])

    vectors: dict[int, np.ndarray] = {}
    model = get_model()
    for year in sorted(corpora):
        texts = corpora[year]
        sents = []
        for t in texts:
            sents.extend(s.strip() for s in t.split(".") if len(s.strip()) > 20)
        if not sents:
            sents = ["."]
        vecs = model.encode(sents, show_progress_bar=False)
        v = np.mean(vecs, axis=0)
        cache = year_cache_path(year)
        np.save(cache, v)
        vectors[year] = v
        print(f"  year {year}: {len(texts)} docs, {len(sents)} sents → vector cached")
    return vectors


def build_doc_vectors(corpus_path: Path = None) -> dict[str, tuple[np.ndarray, dict]]:
    """One vector per doc. Returns {doc_hash: (vector, metadata)}."""
    if corpus_path is None:
        corpus_path = ROOT / "data" / "raw" / "vitalik_corpus.jsonl"

    result: dict[str, tuple[np.ndarray, dict]] = {}
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            h = hashlib.sha256(line.encode()).hexdigest()[:12]
            cache = doc_cache_path(h)
            if cache.exists():
                v = np.load(cache)
            else:
                v = embed_text(row["text"])
                np.save(cache, v)
            result[h] = (v, {
                "title": row.get("title", ""),
                "date": row.get("date", ""),
                "url": row.get("url", ""),
                "source": row.get("source", ""),
            })
    return result


def probe(probe_text: str, year_vectors: dict[int, np.ndarray] = None,
          corpus_path: Path = None) -> dict:
    """Score a probe against all year vectors; return ranked results."""
    # Load + weight in probe path so cache-miss fallback doesn't bypass weighting
    if year_vectors is None:
        year_vectors = {}
        doc_counts: dict[int, int] = defaultdict(int)
        if corpus_path is None:
            corpus_path = ROOT / "data" / "raw" / "vitalik_corpus.jsonl"
        with open(corpus_path) as f:
            for line in f:
                row = json.loads(line)
                try:
                    y = int(row.get("date", "")[:4])
                    if 2010 < y < 2030:
                        doc_counts[y] += 1
                except ValueError:
                    pass
        for p in EMB_DIR.glob("year_*.npy"):
            y = int(p.stem.split("_")[1])
            v = np.load(p)
            w = math.sqrt(max(doc_counts.get(y, 1), 1))
            year_vectors[y] = v * w

    pv = embed_text(probe_text)
    scores = [(y, cosine_sim(pv, v)) for y, v in year_vectors.items()]
    scores.sort(key=lambda x: -x[1])
    return {
        "probe_chars": len(probe_text),
        "top_year": scores[0][0] if scores else None,
        "top_score": round(scores[0][1], 6) if scores else 0.0,
        "rankings": [{"year": y, "cosine": round(c, 6)} for y, c in scores],
    }


if __name__ == "__main__":
    import sys
    p = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(p))

    print("=== Building year vectors ===")
    years = build_year_vectors()
    print(f"\nCached {len(years)} year vectors to {EMB_DIR}")

    if len(sys.argv) > 1:
        probe_text = " ".join(sys.argv[1:])
    else:
        probe_text = "Polynomial commitments can replace state roots"
    print(f"\nProbe: {probe_text[:80]}")
    result = probe(probe_text, years)
    print(f"Top year: {result['top_year']}  cosine={result['top_score']}")
    print("\nAll years:")
    for r in result["rankings"]:
        print(f"  {r['year']}: {r['cosine']:.6f}")
