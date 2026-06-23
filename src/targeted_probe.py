"""
targeted_probe.py — score corpus against high-signal controversial Vitalik topics.
These are topics where anonymous posting was strategically rational 2016-2018:
  - issuance/minimum_viable
  - fee_market/EIP1559_precursors
  - casper_tfg_vs_cbc
  - plasma_minimal
  - state_expiry_stateless
  - validator_slashing/penalties
"""
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'

TOPIC_PATTERNS = {
    'issuance_minimum_viable': [
        r'\b(minimum viable issuance|mvi|issuance curve|issuance rate|validator reward|staking reward)\b',
        r'\b(circulating supply equilibrium|minimum viableissuance|burn|issuance burn)\b',
    ],
    'fee_market': [
        r'\b(fee market|transaction fee|base fee|eip-1559|burn fee|priority fee)\b',
        r'\b(gas price|gas limit|gas market|fee burn)\b',
    ],
    'casper_tfg_cbc': [
        r'\b(casper tfg|friendly finality gadget|casper cbc|casper the friendly)\b',
        r'\b(finality gadget|checkpoint|lmd ghost|ffg)\b',
    ],
    'plasma_minimal': [
        r'\b(minimal viable plasma|minimal plasma|plasma chain|plasma design)\b',
        r'\b(plasma exit|plasma fraud|plasma challenge)\b',
    ],
    'state_expiry_stateless': [
        r'\b(state expiry|stateless client|weak statelessness|state rent|history expiry)\b',
        r'\b(state size|state bloat|state storage)\b',
    ],
    'validator_slashing': [
        r'\b(slashing|validator penalty|slash|penalty|inactivity leak|double vote|surround vote)\b',
        r'\b(validator deposit|withdrawal|exit queue|churn limit)\b',
    ],
}

WEIGHTS = {
    'issuance_minimum_viable': 2.0,
    'fee_market': 2.0,
    'casper_tfg_cbc': 1.8,
    'plasma_minimal': 1.5,
    'state_expiry_stateless': 1.2,
    'validator_slashing': 1.0,
}

def targeted_score(text):
    blob = text.lower()
    hits = Counter()
    for topic, pats in TOPIC_PATTERNS.items():
        matched = 0
        for pat in pats:
            matched += len(re.findall(pat, blob, flags=re.IGNORECASE))
        if matched:
            hits[topic] = matched
    raw = sum(hits[t] * WEIGHTS[t] for t in hits)
    norm = sum(WEIGHTS.values())
    return raw / (norm + 1e-9), hits

def load_rows(path):
    return [json.loads(l) for l in open(path)]

def rank_corpus(path, out_path=None, top_n=100):
    rows = load_rows(path)
    scored = []
    for r in rows:
        text = (r.get('title','') + ' ' + r.get('text','') + ' ' + r.get('body','')).strip()
        ts, hits = targeted_score(text)
        scored.append({
            'title': r.get('title','')[:90],
            'source': r.get('source',''),
            'nearest_year': r.get('nearest_year', r.get('year','?')),
            'proximity': r.get('proximity', 0),
            'topic_score': round(ts, 4),
            'topic_hits': dict(hits),
            'url': r.get('url',''),
            'author': r.get('author', r.get('username','')),
        })
    scored.sort(key=lambda x: x['topic_score'], reverse=True)
    scored = scored[:top_n]
    if out_path is None:
        stem = Path(path).stem
        out_path = PSEUDO / f'{stem}_targeted.jsonl'
    with open(out_path, 'w') as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'wrote {len(scored)} targeted docs -> {out_path}')
    return out_path

def report(path, top_n=30):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r['topic_score'], reverse=True)
    print(f'targeted ranked: {path}')
    print('='*100)
    for i,r in enumerate(rows[:top_n],1):
        hits = ', '.join(f'{k}({v})' for k,v in r['topic_hits'].items()) if r['topic_hits'] else 'none'
        print(f'{i:2d}. ts={r["topic_score"]:.4f} prox={r["proximity"]:.4f} year={r["nearest_year"]} [{r["source"]}]')
        print(f'    {r["title"][:85]}')
        print(f'    hits: {hits}')
        if r.get('url'): print(f'    url: {r["url"][:100]}')
        print()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', help='scored jsonl with prox/topic fields')
    ap.add_argument('--top', type=int, default=100)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()
    out = rank_corpus(args.corpus, args.out, args.top)
    if args.report:
        report(out, top_n=30)
