"""
score_384.py -- score candidate docs against year vectors in original 384-dim embedding space.
Uses cosine similarity to year vectors (pre-weighted by sqrt(doc_count)).
"""
import json, hashlib, argparse, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path('.').resolve()
EMB_DIR = ROOT / 'data/embeddings'
CACHE_DIR = ROOT / 'data/doc_embeddings'
PSEUDO_DIR = ROOT / 'data/pseudonym'
CACHE_DIR.mkdir(exist_ok=True)
PSEUDO_DIR.mkdir(exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

year_vecs = {}
for p in sorted(EMB_DIR.glob('year_*.npy')):
    y = int(p.stem.split('_')[1])
    year_vecs[y] = np.load(p)

year_list = sorted(year_vecs)

era_map = {y: 'pre-merge' if y <= 2020 else 'post-merge' if y <= 2022 else 'd/acc-era' for y in year_vecs}

# normalize year vectors to unit length
year_norms = {}
for y, v in year_vecs.items():
    n = np.linalg.norm(v)
    year_norms[y] = n
    year_vecs[y] = v / (n + 1e-9)

mean_centroid = np.mean(np.vstack([year_vecs[y] for y in year_list]), axis=0)
mean_norm = np.linalg.norm(mean_centroid)
mean_centroid = mean_centroid / (mean_norm + 1e-9)

def doc_vec(text):
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    cache = CACHE_DIR / 'doc_{}.npy'.format(h)
    if cache.exists():
        return np.load(cache)
    sents = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if not sents:
        sents = [text[:500]]
    v = np.mean(model.encode(sents, show_progress_bar=False), axis=0)
    n = np.linalg.norm(v)
    v = v / (n + 1e-9)
    np.save(cache, v)
    return v

def score_doc(text, title=''):
    full = (title + ' ' + text).strip()
    v = doc_vec(full)
    cos = {}
    for y in year_list:
        cos[y] = float(np.dot(v, year_vecs[y]))
    ranked = sorted(cos, key=cos.get, reverse=True)
    best_y = ranked[0]
    cos_mean = float(np.dot(v, mean_centroid))
    proximity = (cos[best_y] - cos_mean)
    return {
        'nearest_year': best_y,
        'cos_sim': round(cos[best_y], 4),
        'cos_mean': round(cos_mean, 4),
        'proximity': round(proximity, 4),
        'top3_years': ranked[:3],
        'top3_eras': [era_map[y] for y in ranked[:3]],
        'cos_year': {str(y): round(cos[y], 4) for y in year_list}
    }

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def score_corpus(path, out_path=None):
    if out_path is None:
        out_path = PSEUDO_DIR / (Path(path).stem + '_384_scored.jsonl')
    rows = load_jsonl(path)
    scored = []
    for row in rows:
        text = row.get('text','') or row.get('body','') or row.get('content','')
        title = row.get('title','')
        if not text or len(text) < 30:
            continue
        s = score_doc(text, title)
        out = {
            'source': row.get('source', ''),
            'id': row.get('id') or row.get('url') or title[:80],
            'title': title[:140],
            'text_len': len(text),
            'nearest_year': s['nearest_year'],
            'cos_sim': s['cos_sim'],
            'proximity': s['proximity'],
            'top3_years': s['top3_years'],
            'top3_eras': s['top3_eras'],
            'cos_sims': s['cos_year']
        }
        for k in ('url','date','author','created_utc'):
            if k in row:
                out[k] = row[k]
        scored.append(out)
    scored.sort(key=lambda x: x['proximity'], reverse=True)
    with open(out_path, 'w') as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('wrote {} scored docs -> {}'.format(len(scored), out_path))
    return out_path

def report(path, topk=30, min_len=100):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get('text_len', 0) >= min_len]
    rows.sort(key=lambda r: r['proximity'], reverse=True)
    print('corpus: {}  docs>={}c: {}'.format(path, min_len, len(rows)))
    print('='*80)
    for i, r in enumerate(rows[:topk], 1):
        era = era_map[r['nearest_year']]
        print('{:3d}. year={} ({}) | prox={:.4f} cos={:.4f} | len={}'.format(
            i, r['nearest_year'], era, r['proximity'], r['cos_sim'], r.get('text_len')))
        print('     title:', r['title'][:100])
        if r.get('url'): print('     url:  ', r['url'][:110])
        if r.get('date'): print('     date: ', r['date'])
        if r.get('author'): print('     auth: ', r['author'][:40])
        if r.get('created_utc'): print('     cts:  ', r['created_utc'])
        print('     top3:  {} {}'.format(r['top3_years'], r['top3_eras']))
        print()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', nargs='?', default='', help='jsonl to score')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--topk', type=int, default=30)
    ap.add_argument('--out')
    ap.add_argument('--min-len', type=int, default=100)
    args = ap.parse_args()
    if not args.corpus:
        print('usage: python3 src/score_384.py <corpus.jsonl> [--report] [--topk N] [--min-len N]')
        raise SystemExit(0)
    out = score_corpus(args.corpus, args.out)
    if args.report:
        report(out, topk=args.topk, min_len=args.min_len)
