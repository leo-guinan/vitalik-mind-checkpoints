"""
combined_rank.py — merge ethresearched + reddit, normalize proximity + topic score,
rerank by weighted hybrid score.
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'
OUT = PSEUDO / 'combined_topic_ranked.jsonl'

eth_path = PSEUDO / 'ethresearched_clean_384_scored_topic_ranked.jsonl'
red_path = PSEUDO / 'candidates_clean_384_scored_topic_ranked.jsonl'

def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

eth = load_rows(eth_path)
red = load_rows(red_path)

# normalize proximity and topic_score separately within each corpus (z-score-ish)
# so they're comparable across corpora
for rows in [eth, red]:
    prox_vals = [r['proximity'] for r in rows]
    topic_vals = [r['topic_score'] for r in rows]
    pmax, pmin = max(prox_vals), min(prox_vals)
    tmax, tmin = max(topic_vals), min(topic_vals)
    for r in rows:
        r['prox_norm'] = (r['proximity'] - pmin) / (pmax - pmin + 1e-9)
        r['topic_norm'] = (r['topic_score'] - tmin) / (tmax - tmin + 1e-9)
        r['hybrid'] = round(0.4 * r['prox_norm'] + 0.6 * r['topic_norm'], 4)

# label source
for r in eth: r['source'] = 'ethresearched'
for r in red: r['source'] = 'reddit'

combined = eth + red
combined.sort(key=lambda r: r['hybrid'], reverse=True)

with open(OUT, 'w') as f:
    for r in combined:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'wrote {len(combined)} combined docs -> {OUT}')

# report top-30
print(f'\ntop-30 combined (0.4 prox + 0.6 topic):')
print('='*100)
for i, r in enumerate(combined[:30], 1):
    print(f'{i:2d}. hybrid={r["hybrid"]:.4f} prox={r["proximity"]:.4f} ts={r["topic_score"]:.4f} year={r["nearest_year"]} [{r["source"]}] | {r["title"][:75]}')
    if r.get('topic_hits'):
        print('     hits:', ', '.join(f'{k}({v})' for k,v in r['topic_hits'].items()))
    if r.get('url'): print('     url:', r['url'][:100])
    print()
