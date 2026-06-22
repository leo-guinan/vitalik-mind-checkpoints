"""
github_source.py — fetch Ethereum-related GitHub issues/PRs as a third pseudonym corpus.
Uses GitHub search: repo:ethereum/go-ethereum, repo:ethereum/solidity, etc.
Then scores with cosine in 384-dim space via score_384 interface.
"""
import json, re, time
from pathlib import Path
from datetime import datetime

ROOT = Path('.').resolve()
PSEUDO = ROOT / 'data/pseudonym'
OUT = PSEUDO / 'github_raw.jsonl'

REPOS = [
    'ethereum/go-ethereum',
    'ethereum/solidity',
    'ethereum/EIPs',
    'ethereum/execution-apis',
    'ethereum/consensus-specs',
    'ethereum/research',
    'ethereum/ens-contracts',
    'ethereum-optimism/optimism',
]

# time window: Vitalik core pseudonymous period approx 2014-2018
# GitHub API: per_page max 100
def fetch_issues(repo):
    items = []
    for page in range(1, 11):
        url = f'https://api.github.com/repos/{repo}/issues?state=all&per_page=100&page={page}'
        # prefer curl fallback since `requests` may not be installed
        import subprocess
        r = subprocess.run(['curl', '-s', '-H', 'Accept: application/vnd.github.v3+json', url],
                           capture_output=True, text=True)
        data = json.loads(r.stdout or '[]')
        if isinstance(data, str):
            # rate limit or error message from GitHub
            print(f'  rate limit or error for {repo}: {data[:120]}')
            data = []
        if not data:
            break
        items.extend([it for it in data if isinstance(it, dict)])
        if len(data) < 100:
            break
        time.sleep(0.5)
    return items

all_rows = []
for repo in REPOS:
    print(f'fetching {repo} issues...')
    try:
        issues = fetch_issues(repo)
    except Exception as e:
        print(f'  ERROR {repo}: {e}')
        continue
    for it in issues:
        created = it.get('created_at', '')
        year = None
        if created:
            try:
                year = datetime.strptime(created, '%Y-%m-%dT%H:%M:%SZ').year
            except Exception:
                pass
        text = (it.get('title','') + ' ' + (it.get('body','') or '')).strip()
        all_rows.append({
            'source': f'github:{repo}',
            'id': it.get('number'),
            'url': it.get('html_url',''),
            'title': it.get('title',''),
            'text': text,
            'text_len': len(text),
            'date': created,
            'year': year,
            'author': it.get('user',{}).get('login',''),
        })
    print(f'  got {len(issues)} issues')

print(f'total github rows: {len(all_rows)}')

# filter to likely-relevant window with some min length
filtered = [r for r in all_rows if r.get('year') and 2014 <= r['year'] <= 2020 and r['text_len'] >= 120]
print(f'after 2014-2020 + len>=120: {len(filtered)}')

# drop exact title dupes
seen = set()
deduped = []
for r in filtered:
    k = r['title'].strip().lower()
    if k and k not in seen:
        seen.add(k)
        deduped.append(r)
print(f'after title dedup: {len(deduped)}')

with open(OUT, 'w') as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'wrote {len(deduped)} github corpus rows -> {OUT}')
