"""
known_filter.py — drop likely-known Vitalik posts and re-hybridize.
All known-surface posts from the existing corpus with metadata clearing.
"""
import json, re
from pathlib import Path

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'
OUT = PSEUDO / 'combined_unknown_only.jsonl'

KNOWN_PATTERNS = {
    'keybase_proof': re.compile(r'keybase proof', re.I),
    'ama_announcement': re.compile(r'\bama\b|ask me anything', re.I),
    'official_blog_post': re.compile(r'blog\.ethereum\.org|vitalik\.eth\.limo|official', re.I),
    'xpost_marker': re.compile(r'x-post|xpost|repost|cross.?post', re.I),
    'personal_blog': re.compile(r'vbuterin\.blog|\.blog\.ca', re.I),
    'by_line_vitalik': re.compile(r'(written by|by )\s*vitalik', re.I),
}

KNOWN_TITLES_EXACT = {
    'my keybase proof',
    'eth 2.0 researchers ama',
    'request for public feedback',
}

def is_known(r):
    blob = (r.get('title','') + ' ' + r.get('url','') + ' ' + r.get('author','')).lower()
    for pat in KNOWN_PATTERNS.values():
        if pat.search(blob):
            return True
    if r.get('title','').strip().lower() in KNOWN_TITLES_EXACT:
        return True
    if r.get('author','').lower() in {'vbuterin', 'vitalik buterin', 'buterin'}:
        return True
    return False

rows = [json.loads(l) for l in open(PSEUDO / 'combined_topic_ranked.jsonl')]
unknown = [r for r in rows if not is_known(r)]
print(f'combined total: {len(rows)} -> after known filter: {len(unknown)}')

with open(OUT, 'w') as f:
    for r in unknown:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('\ntop-20 UNKNOWN only (re-hybridized):')
print('='*100)
for i,r in enumerate(unknown[:20], 1):
    print(f'{i:2d}. hybrid={r["hybrid"]:.4f} prox={r["proximity"]:.4f} ts={r["topic_score"]:.4f} year={r["nearest_year"]} [{r["source"]}] | {r["title"][:75]}')
    if r.get('topic_hits'):
        print('     hits:', ', '.join(f'{k}({v})' for k,v in r['topic_hits'].items()))
    if r.get('url'): print('     url:', r['url'][:110])
    print()
