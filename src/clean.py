"""
clean.py — dedup and filter corpora before scoring.
"""
import json, hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'
PSEUDO.mkdir(exist_ok=True)

def content_hash(text):
    h = hashlib.md5(text.encode()).hexdigest()[:10]
    return h

def dedup(rows, key='title'):
    seen = set()
    out = []
    for r in rows:
        k = r.get(key, '').strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def clean_reddit():
    path = PSEUDO / 'candidates_raw.jsonl'
    out = PSEUDO / 'candidates_clean.jsonl'
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    print(f'reddit raw: {len(rows)}')
    # drop empty title + body
    rows = [r for r in rows if r.get('title') or r.get('text','').strip() or r.get('body','').strip()]
    rows = [r for r in rows if len((r.get('title','') + ' ' + r.get('text','') + ' ' + r.get('body','')).strip()) >= 80]
    rows = dedup(rows, 'title')
    print(f'reddit clean: {len(rows)}')
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return out

def clean_ethresearched():
    path = PSEUDO / 'ethresearched_raw.jsonl'
    out = PSEUDO / 'ethresearched_clean.jsonl'
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    print(f'ethresearched raw: {len(rows)}')
    # filter very short or malformed
    rows = [r for r in rows if len(r.get('text','').strip()) >= 200]
    rows = dedup(rows, 'text')
    print(f'ethresearched clean: {len(rows)}')
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return out

if __name__ == '__main__':
    r = clean_reddit()
    e = clean_ethresearched()
    print('wrote', r, e)
