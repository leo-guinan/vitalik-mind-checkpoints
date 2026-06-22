"""
pseudonym.py -- score candidate documents against Vitalik's mental-model trajectory.
Outputs ranked proximity to each year + nearest-year label + era.
"""
import json, csv, hashlib, argparse, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import umap
from sentence_transformers import SentenceTransformer

ROOT = Path('.').resolve()
TSV = ROOT / 'data/umap_clusters.tsv'
EMB_DIR = ROOT / 'data/embeddings'
CACHE_DIR = ROOT / 'data/doc_embeddings'
PSEUDO_DIR = ROOT / 'data/pseudonym'
PSEUDO_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

year_vecs = {}
for p in sorted(EMB_DIR.glob('year_*.npy')):
    y = int(p.stem.split('_')[1])
    year_vecs[y] = np.load(p)

rows_umap = list(csv.DictReader(open(TSV), delimiter='\t'))
embed_coords = {int(r['year']): (float(r['umap_x']), float(r['umap_y'])) for r in rows_umap if r['layer']=='embed'}

year_list = sorted(year_vecs)
all_mat = np.vstack([year_vecs[y] for y in year_list])
reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, metric='cosine', n_components=2, random_state=42)
reducer.fit(all_mat)

era_map = {y: 'pre-merge' if y <= 2020 else 'post-merge' if y <= 2022 else 'd/acc-era' for y in year_vecs}

def doc_vec(text):
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    cache = CACHE_DIR / f'doc_{h}.npy'
    if cache.exists():
        return np.load(cache)
    sents = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if not sents:
        sents = [text[:500]]
    v = np.mean(model.encode(sents, show_progress_bar=False), axis=0)
    np.save(cache, v)
    return v

def score_doc(text, title=''):
    full = (title + ' ' + text).strip()
    v = doc_vec(full).reshape(1,-1)
    pj = reducer.transform(v)[0]
    year_dist = {}
    for y in year_list:
        cy, cx = embed_coords[y]
        d = float(np.linalg.norm(np.array([cx, cy]) - pj))
        year_dist[y] = d
    ranked = sorted(year_dist, key=year_dist.get)
    best_y = ranked[0]
    nearest_dist = year_dist[best_y]
    dist_to_mean = float(np.mean([year_dist[y] for y in year_list]))
    proximity = 1.0 - (nearest_dist / (dist_to_mean + 1e-9))
    # also report top-3
    top3 = ranked[:3]
    top3_eras = [era_map[y] for y in top3]
    return {
        'nearest_year': best_y,
        'nearest_dist': round(nearest_dist, 4),
        'proximity': round(proximity, 4),
        'top3_years': top3,
        'top3_eras': top3_eras,
        'year_dists': {str(y): round(year_dist[y], 4) for y in year_list}
    }

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def score_corpus(path, out_path=None):
    if out_path is None:
        out_path = PSEUDO_DIR / (Path(path).stem + '_scored.jsonl')
    rows = load_jsonl(path)
    scored = []
    for row in rows:
        text = row.get('text','') or row.get('body','') or row.get('content','')
        title = row.get('title','')
        if not text:
            continue
        s = score_doc(text, title)
        out = {
            'source': str(path),
            'id': row.get('id') or row.get('url') or title[:80],
            'title': title[:120],
            'text_snippet': text[:300].replace('\n',' '),
            'nearest_year': s['nearest_year'],
            'nearest_dist': s['nearest_dist'],
            'proximity': s['proximity'],
            'top3_years': s['top3_years'],
            'top3_eras': s['top3_eras'],
            'year_dists': s['year_dists']
        }
        if 'url' in row: out['url'] = row['url']
        if 'date' in row: out['date'] = row['date']
        if 'author' in row: out['author'] = row['author']
        scored.append(out)
    scored.sort(key=lambda x: x['proximity'], reverse=True)
    with open(out_path, 'w') as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'wrote {len(scored)} scored docs -> {out_path}')
    return out_path

def report(path, topk=20):
    rows = [json.loads(l) for l in open(path)]
    print(f'corpus: {path}  docs={len(rows)}')
    print('='*80)
    for i, r in enumerate(rows[:topk], 1):
        era = era_map[r['nearest_year']]
        print(f"{i:3d}. year={r['nearest_year']} ({era}) | prox={r['proximity']:.4f} | dist={r['nearest_dist']:.4f}")
        print(f"     title: {r['title'][:100]}")
        if r.get('url'): print(f"     url:   {r['url'][:120]}")
        if r.get('date'): print(f"     date:  {r['date']}")
        if r.get('author'): print(f"     author:{r['author'][:80]}")
        print(f"     text:  {r['text_snippet'][:180]}")
        print(f"     top3:  {r['top3_years']}  eras={r['top3_eras']}")
        print()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', nargs='?', default='', help='jsonl to score')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--out')
    args = ap.parse_args()
    if not args.corpus:
        print('usage: python3 src/pseudonym.py <corpus.jsonl> [--report] [--topk N] [--out path]')
        raise SystemExit(0)
    out = score_corpus(args.corpus, args.out)
    if args.report:
        report(out, topk=args.topk)
