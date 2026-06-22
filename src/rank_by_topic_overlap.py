"""
rank_by_topic_overlap.py — re-rank scored candidates by overlap with known Vitalik signature topics.
"""
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'

TOPICS = {
    'snarks_starks': r'\b(snark|stark|zk-?snark|zero.?knowledge|groth16|plonk|bulletproof)\b',
    'sharding_das': r'\b(shard|data availability|das|danksharding|kzg|polynomial commitment)\b',
    'rollups': r'\b(rollup|optimistic rollup|zk.?rollup|arbitrum|optimism|zksync|loopring|starkex)\b',
    'account_abstraction': r'\b(account abstraction|erc.?4337|erc.?7702|entrypoint|paymaster|smart account)\b',
    'consensus_pos': r'\b(casper|proof of stake|pos|finality|beacon chain|lmd ghost|ffg)\b',
    'verkle_merkle': r'\b(verkle|merkle|tree|root|witness|state trie|trie)\b',
    'state_expiry': r'\b(state expiry|stateless|weak statelessness|state rent|history expiry)\b',
    'mev': r'\b(mev|maximal extractable value|proposer.?builder|pbs|mev.?boost|flashbots|relay|suave)\b',
    'eips': r'\b(eip.?[0-9]+|erc.?[0-9]+|rip.?[0-9]+)\b',
    'cryptoecon': r'\b(cryptoeconomics|double spent|incentive|slashing|validator reward|issuance|burn)\b',
    'languages': r'\b(vyper|serpent|solidity|eewasm|move|fe|huff)\b',
    'ec_paired': r'\b(elliptic curve|pairing|bn254|bls12|babyjubjub|alt.?bn128|curve25519)\b',
    'plasma_state': r'\b(plasma|state channel|payment channel|rugpull|fraud proof|validium)\b',
    'defi': r'\b(uniswap|amm|automated market maker|order.?book|dex|decentralized exchange)\b',
    'privacy': r'\b(stealth address|privacy|zk.?proof|mixer|tornado|semaphore|identity)\b',
}

TOPIC_WEIGHTS = {
    'snarks_starks': 2.0,
    'sharding_das': 2.0,
    'rollups': 1.5,
    'account_abstraction': 2.0,
    'consensus_pos': 1.5,
    'verkle_merkle': 1.2,
    'state_expiry': 1.0,
    'mev': 1.2,
    'eips': 0.8,
    'cryptoecon': 1.2,
    'languages': 0.6,
    'ec_paired': 1.0,
    'plasma_state': 1.0,
    'defi': 0.5,
    'privacy': 1.0,
}

def topic_score(text):
    text_lower = text.lower()
    hits = Counter()
    for topic, pat in TOPICS.items():
        matched = re.findall(pat, text_lower, flags=re.IGNORECASE)
        if matched:
            hits[topic] = len(matched)
    raw = sum(hits[t] * TOPIC_WEIGHTS[t] for t in hits)
    norm = sum(TOPIC_WEIGHTS.values())
    return raw / (norm + 1e-9), hits

def rerank(path, out_path=None, top_n=50):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r['proximity'], reverse=True)
    rows = rows[:top_n]
    out = []
    for r in rows:
        text = (r.get('title','') + ' ' + r.get('text','')).strip()
        ts, hits = topic_score(text)
        out.append({
            'title': r['title'][:90],
            'nearest_year': r['nearest_year'],
            'proximity': r['proximity'],
            'cos_sim': r['cos_sim'],
            'topic_score': round(ts, 4),
            'topic_hits': dict(hits),
            'url': r.get('url',''),
        })
    out.sort(key=lambda x: x['topic_score'], reverse=True)
    if out_path is None:
        stem = Path(path).stem
        out_path = PSEUDO / f'{stem}_topic_ranked.jsonl'
    with open(out_path, 'w') as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'wrote {len(out)} topic-ranked docs -> {out_path}')
    return out_path

def report(path, top_n=20):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: r['topic_score'], reverse=True)
    print(f'topic-ranked: {path}')
    print('=' * 90)
    for i, r in enumerate(rows[:top_n], 1):
        print(f'{i:2d}. ts={r["topic_score"]:.4f} prox={r["proximity"]:.4f} year={r["nearest_year"]} | {r["title"][:75]}')
        if r['topic_hits']:
            print('     hits:', ', '.join(f'{k}({v})' for k,v in r['topic_hits'].items()))
        if r.get('url'): print('     url:', r['url'][:100])
        print()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', nargs='?', default='data/pseudonym/ethresearched_clean_384_scored.jsonl')
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()
    out = rerank(args.corpus, args.out, args.top)
    if args.report:
        report(out, top_n=20)
