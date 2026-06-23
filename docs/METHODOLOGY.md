# Vitalik Pseudonym Search — Methodology

## Problem Statement
Vitalik Buterin posed: "At some point this decade, I wrote a published document of medium importance to Ethereum — I estimate ~200–2000 documents in Ethereum are as or more important — not under my name. Find it."

This artifact documents the reasoning behind the search pipeline and the hypotheses we are testing.

---

## Core Hypothesis

We assume the "hidden" document was written **strategically anonymously**, not just casually. The motivation for pseudonymous publication must outweigh the cost of losing authorial credit. For Vitalik specifically, that means the document likely falls into one of three categories:

1. **Ideas that undermined Ethereum's value prop if attributed to its founder** — e.g., "we should cut validator rewards," "issuance will drop to near-zero," "plasma is overhyped."
2. **Ideas that put him in a specific camp against a research consensus** — e.g., Casper TFG vs CBC, PoW vs PoS transition speed, rollup-first vs sharding-first.
3. **Ideas that were obviously right but unwelcome** — e.g., fee-burn before EIP-1559 consensus, state expiry before node operators accepted it.

Time window: **2016–2019**. After 2019 Vitalik's visibility made pseudonymous posting implausible; before 2016 Ethereum research was too small for "medium importance" to be hidden easily.

---

## Architecture

We built a three-stream extraction model:

| Stream | What | Implementation |
|--------|------|----------------|
| Mental model | Concepts, co-occurrence, temporal checkpoint graphs | `concepts.py` → year checkpoints |
| Style/meme | Compression signatures, analogy density, readability, meme-vocab density | `style.py` → 23 numeric features |
| Embedding model | 384-dim pooled sentence-transformers (all-MiniLM-L6-v2) per document | `embeddings.py` → year centroids |

We abandoned stylometric authorship attribution (the standard approach) because UMAP collapses generic documents to the 2017 anchor, making the 384-dim cosine similarity space the more reliable discriminator.

---

## Sources

| Source | Docs | Status | Notes |
|--------|------|--------|-------|
| vitalik.eth.limo | 156 | Live | ENS mirror, auto-discovered from index, 2016–2026 |
| reddit: r/ethereum (vbuterin) | 785 raw → 108 clean | Clean | Mostly already-known Vitalik posts; some cross-posts |
| ethresear.ch | 3003 topics → 1122 raw → 1324 clean | Live | Parallel Discourse API fetch; 1881 errors (deleted/gated) |
| GitHub: ethereum/* repos | 48 filtered | Scoring fixed | 2014–2020 window, min 120 chars |
| bitcointalk.org | 0 | Blocked | 403 Forbidden; jina.ai proxy returns guest page |

---

## Scoring Approach

### Primary: 384-dim Cosine Similarity to Year Vectors
- Each year in the Vitalik corpus gets a pooled embedding vector.
- Vectors are normalized to unit length.
- Scoring: `cos(doc, year_vec)` minus `cos(doc, mean_centroid)` = **proximity**.
- Higher proximity = closer to that year's writing style/topic distribution.

### Secondary: Topic-Overlap Weighted Matching
Regex-based topic dictionaries with per-topic weights reflecting strategic importance:

| Topic | Weight | Rationale |
|-------|--------|-----------|
| `snarks_starks` | 2.0 | Core crypto; pseudonymous posting plausible but author adds credibility |
| `sharding_das` | 2.0 | High-consensus topic; anonymous posting safe but signals alignment |
| `rollups` | 1.5 | Post-2019 dominant; less likely to be the "hidden" doc |
| `account_abstraction` | 2.0 | Later-era; lower probability |
| `consensus_pos` | 1.5 | Casper debates were actively factionalized |
| `verkle_merkle` | 1.2 | Implementation-specific |
| `state_expiry` | 1.0 | Unpopular with node operators; strong candidate |
| `mev` | 1.2 | Post-2020; less likely from early window |
| `validator_slashing` | 1.0 | Controversial inside consensus research |
| `eips` | 0.8 | Public process; pseudonymous posting unnecessary |
| `cryptoecon` | 1.2 | Issuance/validator economics; higher stakes |
| `languages` | 0.6 | Lower strategic value |
| `ec_paired` | 1.0 | Technical but narrow |
| `plasma_state` | 1.0 | Plasma optimism was mainstream |
| `defi` | 0.5 | Vitalik rarely wrote DeFi trading mechanism pseudonymously |

### Hybrid Score
`hybrid = 0.4 * proximity_norm + 0.6 * topic_norm`

Normalization is per-corpus so scores are comparable across sources.

---

## Validation

| Test | Result | Interpretation |
|------|--------|----------------|
| Known 2020 doc probes to 2020 | #1 | Recall within trained corpus |
| Polynomial commitments probe | 2019 | Near-miss; topic drifted across years |
| Account abstraction probe | 2022 | Post-shift correct; 1yr off |
| Leave-one-out exact-year | 32.1% | Above chance (1/11 = 9%) |
| Leave-one-out ±2yr | 65.4% | Reasonable for 11-year problem |
| Era buckets (pre/post/dacc) | 61% | Significantly above majority baseline (37%) |

---

## Known-Filter Heuristic

To avoid scoring already-identified Vitalik posts:
- Match `author` field against `vbuterin`, `vitalik buterin`
- Match titles: `keybase proof`, `AMA`, `official`
- Match URLs: vitalik.eth.limo, blog.ethereum.org
- Match body patterns: `x-post`, `written by vitalik`, `.blog.ca`

---

## Targeted Topic Hypothesis

The `targeted_probe.py` script re-ranks corpora using only the **strategically controversial** topic patterns above. Results so far:

**Ethresearched top-5 targeted:**
1. "Transaction fees in Casper FFG" (2017) — consensus_pos(2)
2. "Cartel formation incentive in full PoS - interest rates must rise with total deposit size" (2017) — consensus_pos(1), cryptoecon(1)
3. "A sketch for a STARK-based accumulator" (2016) — snarks_starks(1)
4. "SLONK—a simple universal SNARK" (2016) — snarks_starks(1)
5. "Batched cross-shard transaction fee payment" (2017) — sharding_das(1)

**Combined hybrid top-5 after known-filter:**
1. "Transaction fees in Casper FFG" (2017) — consensus_pos(2)
2. "Shard block staggering" (2017 reddit) — sharding_das(1)
3. "Cartel formation incentive in full PoS" (2017) — consensus_pos(1), cryptoecon(1)
4. "A sketch for a STARK-based accumulator" (2016) — snarks_starks(1)
5. "SLONK—a simple universal SNARK" (2016) — snarks_starks(1)

Observation: **6 of top-20 ethresearched combined hits are authored by vbuterin in the index**, meaning they are known and should be filtered out. Of the remaining 14, none carry the specific "minimum viable issuance" or "fee market before EIP-1559" framing we hypothesized as most strategically motivated.

---

## Next Steps

1. **Expand author metadata** — Pull GitHub commit logs for ethereum/research (PhABC, JustinDrake, ebuchman cluster appears to be a related pseudonymous network).
2. **Bitcointalk bypass** — The 403/jina.ai dead end needs a different vector (Wayback Machine, cached copies).
3. **Concept-graph distance** — Sparse graphs at window=5 yield 0.0 cross-year similarity. Needs larger context window or transitive-edge smoothing.
4. **UMAP collapse fix** — Generic corpus collapses to 2017 anchor. 384-dim cosine is the current workaround; possible solutions: UMAP on per-document residuals, or min-max normalization within era.
5. **Manual spot-check** — Read the actual candidate texts for "Transaction fees in Casper FFG" and "Cartel formation incentive" to verify if they match Vitalik's specific phrasing/compression style.

---

## Files

| File | Purpose |
|------|---------|
| `src/ingest.py` | vitalik.eth.limo discovery + batch curl |
| `src/concepts.py` | concept extractor → year checkpoints |
| `src/style.py` | style/meme fingerprint → 23 features |
| `src/embeddings.py` | sentence-transformers → year vectors |
| `src/pipeline.py` | unified CLI |
| `src/umap_clusters.py` | UMAP + KMeans on both layers |
| `src/plot_constellation.py` | style/embed constellation PNG |
| `src/drift.py` | drift trajectory + probe distance |
| `src/year_accuracy.py` | exact-year recovery metrics |
| `src/probe.py` | arbitrary text → year/proximity |
| `src/score_384.py` | cosine similarity in 384-dim space |
| `src/candidates.py` | bitcointalk + reddit fetcher |
| `src/ethresearched_fetch.py` | parallel Discourse API fetch |
| `src/clean.py` | dedup + filter corpora |
| `src/rank_by_topic_overlap.py` | topic-overlap ranking |
| `src/combined_rank.py` | hybrid proximity+topic ranking |
| `src/known_filter.py` | drop already-known posts |
| `src/github_source.py` | GitHub issue fetcher |
| `src/author_spotcheck.py` | ethresearched index author lookup |
| `src/targeted_probe.py` | re-rank by controversial topics only |
| `src/viz_dashboard.py` | pseudonym state dashboard |
| `data/pseudonym/ethresearched_clean_384_scored_topic_ranked.jsonl` | ethresearched ranked |
| `data/pseudonym/candidates_clean_384_scored_topic_ranked.jsonl` | reddit ranked |
| `data/pseudonym/combined_topic_ranked.jsonl` | hybrid combined |
| `data/pseudonym/combined_unknown_only.jsonl` | after known-filter |
| `data/pseudonym/github_384_scored_topic_ranked.jsonl` | GitHub ranked |
| `data/pseudonym_dashboard.png` | state visualization |
