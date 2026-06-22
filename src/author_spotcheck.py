"""
author_spotcheck.py — look up author metadata from ethresearched index for top combined hits.
"""
import json, re
from pathlib import Path

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'

index_path = PSEUDO / 'ethresearched_index.json'
combined_path = PSEUDO / 'combined_topic_ranked.jsonl'

index_rows = []
with open(index_path) as f:
    for line in f:
        try:
            index_rows.append(json.loads(line))
        except Exception:
            pass
print(f'index rows loaded: {len(index_rows)}')
print('sample keys:', list(index_rows[0].keys())[:10] if index_rows else 'EMPTY')

lookup = {}
for row in index_rows:
    tid = row.get('id') or row.get('topic_id')
    if tid is not None:
        lookup[tid] = row

combined = [json.loads(l) for l in open(combined_path)]
eth_top = [r for r in combined[:30] if r.get('source') == 'ethresearched']
print(f'\nethresearched in combined top-30: {len(eth_top)}')
print('='*120)
for r in eth_top:
    url = r.get('url','')
    m = re.search(r'/(\d+)$', url)
    tid = int(m.group(1)) if m else None
    meta = lookup.get(tid, {})
    author = meta.get('author','?')
    username = meta.get('username','?')
    created = meta.get('created_at','') or meta.get('last_posted_at','')
    print(f'year={r["nearest_year"]} hybrid={r["hybrid"]:.4f} prox={r["proximity"]:.4f} ts={r["topic_score"]:.4f}')
    print(f'  title: {r["title"][:90]}')
    print(f'  url:   {url}')
    print(f'  id:    {tid} | author={author} username={username} created={created}')
    print()
