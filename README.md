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
# UMAP clustering results
-----------------------


## Pseudonym search methodology
-------------------------------
The goal: find an anonymized Vitalik-era document (medium importance to Ethereum,
2016-2018 window, posted under a non-Vitalik identity for strategic reasons).

Three-layer matching:
  Layer 1 — 384-dim cosine proximity to year vectors (exact-year recovery: 32.1%,
             ±1yr: 55.1%, ±2yr: 65.4%)
  Layer 2 — topic overlap score against high-signal controversial topics:
             `issuance_minimum_viable`, `fee_market`, `casper_tfg_cbc`,
             `plasma_minimal`, `state_expiry_stateless`, `sharding_das`, 
             `consensus_pos`, `snarks_starks`
  Layer 3 — style fingerprint L2 distance to per-year profiles

Author fence: UMAP on author-level style+topic vectors from all corpora.
  Known Vitalik: vbuterin, vitalik buterin
  Fence centroid [-2.715, 4.754], radius 0.226
  Unknown authors inside fence: 0

Corpus stats:
  183-doc trained corpus (vitalik.eth.limo, 2016-2026)
  Pseudonym candidates: 1324 ethresearched + 108 reddit + 48 github + 41 bitcointalk
  38 unique authors evaluated in author fence
  Top combined hybrid hits cluster in 2016-2017 consensus/sharding topics

Key artifacts:
  src/targeted_probe.py       score against controversial-topic layer
  src/author_fence.py          author-level UMAP fence plot
  src/rescore_with_style.py   hybrid proximity+topic+style re-ranker
  src/rank_by_topic_overlap.py topic-overlap ranking
  docs/METHODOLOGY.md          full methodology write-up


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
