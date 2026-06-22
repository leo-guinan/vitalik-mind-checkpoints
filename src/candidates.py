"""
candidates.py -- fetch pseudonymous Vitalik candidates from bitcointalk + reddit.
Uses jina.ai summarizer as proxy to bypass 403 blocks.
Outputs a scored JSONL ready for review.
"""
import csv, json, os, re, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry

ROOT = Path('.').resolve()
OUT = ROOT / 'data/pseudonym/candidates_raw.jsonl'
OUT.parent.mkdir(exist_ok=True)

session = requests.Session()
session.headers['User-Agent'] = 'Mozilla/5.0 (research; vitalik-mind-checkpoints)'
session.headers['X-Retain-URLs'] = '1'
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
session.mount('https://', HTTPAdapter(max_retries=retries))

def write_rows(rows):
    with open(OUT, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('wrote {} rows -> {}'.format(len(rows), OUT))

class Bitcointalk:
    def fetch_user(self, username, max_pages=20, uid=None):
        rows = []
        uuids = set()
        if not uid:
            # search via Google for uid hint
            try:
                sr = session.get('https://r.jina.ai/http://bitcointalk.org/index.php',
                                 params={'action': 'search', 'keywords': username, 'search': 'Search'},
                                 timeout=30)
                m = re.search(r'profile;u=(\d+)', sr.text or '')
                uid = m.group(1) if m else None
            except Exception as e:
                print('bt uid lookup err:', e)
            if not uid:
                print('bt uid not found for {}'.format(username))
                return rows
        page = 0
        while page < max_pages:
            start = page * 20
            url = 'https://bitcointalk.org/index.php?action=profile;u={};sort=posts;start={}'.format(uid, start)
            try:
                jr = session.get('https://r.jina.ai/http://bitcointalk.org/index.php',
                                 params={'action': 'profile;u', 'u': uid, 'sort': 'posts', 'start': start},
                                 timeout=30)
            except Exception as e:
                print('bt fetch err:', e)
                break
            text = jr.text or ''
            posts_m = re.findall(r'\[(\d+)\]\s+(.*?)\s+\[Reply with quote\]', text, re.S)
            if not posts_m:
                # fallback: each section looks like "Subject ... \n <post body> … Reply with quote"
                parts = re.split(r'\n{2,}', text)
                if len(parts) < 2:
                    break
                # take all non-empty parts with title-like first line
                for part in parts:
                    sub = part.strip()
                    if not sub:
                        continue
                    lines = sub.splitlines()
                    title = lines[0][:120] if lines else ''
                    body = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                    rows.append({'source': 'bitcointalk', 'author': username,
                                 'title': title, 'text': body[:4000],
                                 'url': url})
            else:
                for mid, title in posts_m:
                    try:
                        jp = session.get('https://r.jina.ai/http://bitcointalk.org/index.php',
                                         params={'topic': '0', 'msg': mid}, timeout=30)
                        body = jp.text or ''
                        rows.append({'source': 'bitcointalk', 'author': username,
                                     'title': title.strip(), 'text': body[:4000],
                                     'url': 'https://bitcointalk.org/index.php?topic=0.msg{}#msg{}'.format(mid, mid)})
                    except Exception as e:
                        print('bt body fetch err:', e)
            print('  bt {} page {}: total {}'.format(username, page+1, len(rows)))
            if not posts_m:
                break
            page += 1
            time.sleep(0.3)
        return rows

class Reddit:
    def fetch_user(self, username, max_items=200):
        rows = []
        headers = {'User-Agent': 'research:vitalik-mind-checkpoints:v0.1 (by /u/leoguinan)'}
        for kind in ('submitted', 'comments'):
            url = 'https://www.reddit.com/user/{}/{}.json?limit=100'.format(username, kind)
            seen = 0
            while url and seen < max_items:
                try:
                    r = session.get(url, headers=headers, timeout=20)
                except Exception as e:
                    print('reddit err:', e)
                    break
                if r.status_code == 429:
                    time.sleep(2)
                    continue
                if r.status_code != 200:
                    print('reddit status {} page {} {}'.format(r.status_code, kind, username))
                    break
                try:
                    data = r.json()
                except Exception as e:
                    print('reddit json err:', e)
                    break
                children = data.get('data', {}).get('children', [])
                if not children:
                    break
                for c in children:
                    item = c['data']
                    text = item.get('selftext', '') or item.get('body', '') or ''
                    title = item.get('title', '')
                    rows.append({'source': 'reddit', 'author': item.get('author', username),
                                 'title': title, 'text': text,
                                 'url': 'https://www.reddit.com{}'.format(item.get('permalink', '')),
                                 'created_utc': item.get('created_utc')})
                seen += len(children)
                after = data['data'].get('after')
                url = 'https://www.reddit.com/user/{}/{}.json?limit=100&after={}'.format(username, kind, after) if after else None
                print('  reddit {} {}: total {}'.format(username, kind, seen))
                time.sleep(0.6)
        return rows

if __name__ == '__main__':
    btc = Bitcointalk()
    rdt = Reddit()
    rows = []
    print('=== bitcointalk random88 ===')
    rows.extend(btc.fetch_user('random88', max_pages=25, uid='13944'))
    print('=== reddit vbuterin ===')
    rows.extend(rdt.fetch_user('vbuterin', max_items=250))
    seen = set()
    deduped = []
    for r in rows:
        u = r.get('url') or r.get('text','')[:60]
        if u not in seen:
            seen.add(u)
            deduped.append(r)
    write_rows(deduped)
    print('done total:', len(deduped))
