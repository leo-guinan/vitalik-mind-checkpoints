vitalik-mind-checkpoints
=======================

Mental model evolution tracker for Vitalik Buterin.

Hypothesis: anonymized Vitalik-era documents will show concept-graph
similarity to one of his time-bucketed mental models, not just surface
stylometry.

Split extraction
----------------
  Stream 1 — mental model: concept graph + co-occurrence edges (annual)
  Stream 2 — style/meme:  compression fingerprint + rhetorical features (annual)
  Stream 3 — embeddings : sentence-transformers pooled vectors (annual)

Plus UMAP 2D clustering on both layers to visualize decoupling / coupling.

Directory layout
----------------
data/
  raw/                      ingested source texts (183 docs, 2014-2026)
  checkpoints/              per-year concept graphs (JSON) — Stream 1
  style_profiles/           per-year style fingerprints (JSON) — Stream 2
  embeddings/               per-year pooled vectors (.npy) — Stream 3
  umap_clusters.tsv         UMAP 2D coordinates for style + embed layers
src/
  ingest.py                 vitlika.eth.limo discovery -> data/raw/vitalik_corpus.jsonl
  concepts.py               concept extractor -> data/checkpoints/checkpoint_YYYY.json
  match.py                  compare text against concept checkpoints
  style.py                  style/meme extraction -> data/style_profiles/style_YEAR.json
  embeddings.py             sentence-transformers -> data/embeddings/year_YEAR.npy
  pipeline.py               unified CLI (ingest / style / embed)
  umap_clusters.py          UMAP + KMeans on both layers -> data/umap_clusters.tsv

What the layers capture
-----------------------
Stream 1 (concepts): top-200 concept frequency + top-500 co-occurrence edges per year.
Stream 2 (style):   spread score, meme vocabulary density, readability fingerprint,
                    analogy density, compression signature counts, parenthetical rate.
Stream 3 (embeddings): 384-dim sentence-transformers vectors pooled per year,
                       weighted by sqrt(doc_count) to damp small-year bias.

UMAP clustering results
-----------------------
Style layer KMeans(k=3):
  cluster 0: [2020, 2022, 2024, 2025]
  cluster 1: [2016, 2017, 2018, 2023, 2026]
  cluster 2: [2019, 2021]

Embed layer KMeans(k=3):
  cluster 0: [2019, 2020, 2021]
  cluster 1: [2022, 2023, 2024, 2025, 2026]
  cluster 2: [2016, 2017, 2018]

Cross-layer same-year distance: mean = 16.34 (style vs embed spaces differ).

Running
-------
cd ~/Projects/vitalik-mind-checkpoints

# full pipeline (ingest + style + embeddings)
python3 src/pipeline.py

# embed probe
python3 src/pipeline.py embed "polynomial commitments replace state roots"

# style pass only
python3 src/pipeline.py style

# UMAP + clustering report
python3 src/umap_clusters.py 3 0.3 cosine
