vitalik-mind-checkpoints
=======================

Mental model evolution tracker for Vitalik Buterin.

Hypothesis: anonymized Vitalik-era documents will show concept-graph
similarity to one of his time-bucketed mental models, not just surface
stylometry.

Directory layout
----------------
data/
  raw/                    ingested source texts with year+url
  checkpoints/            per-year concept graphs (JSON)
src/
  ingest.py               curated URL fetcher -> data/raw/vitalik_corpus.jsonl
  concepts.py             concept extractor -> data/checkpoints/checkpoint_YYYY.json
  match.py                compare arbitrary text against checkpoints

How checkpoints work
--------------------
Each checkpoint contains:
- top 200 concepts by frequency (with tech-term boosting)
- top 500 concept-pair co-occurrence edges (window=5 tokens)
- doc count and generation timestamp

What they capture
-----------------
2014: block, contract, transaction, storage, security — the substrate
2015: blockchain, system, problem — the abstraction layer
2021: proposer, builder, fee, bundle, slot — the market-mechanism layer
2026: state, storage, options, oracle, staking — the asset layer

Running
-------
cd ~/Projects/vitalik-mind-checkpoints

# 1. ingest known Vitalik writings
python3 src/ingest.py 300

# 2. generate checkpoints
python3 src/concepts.py

# 3. match text against checkpoints
python3 src/match.py some_text_or_file
