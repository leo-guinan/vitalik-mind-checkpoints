import json, csv, hashlib, numpy as np, umap
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

ROOT = Path('.').resolve()
TSV = ROOT / 'data/umap_clusters.tsv'
RAW = ROOT / 'data/raw/vitalik_corpus.jsonl'
EMB_DIR = ROOT / 'data/embeddings'
CACHE_DIR = ROOT / 'data/doc_embeddings'
CACHE_DIR.mkdir(exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

year_vecs = {}
for p in sorted(EMB_DIR.glob('year_*.npy')):
    y = int(p.stem.split('_')[1])
    year_vecs[y] = np.load(p)

rows_umap = list(csv.DictReader(open(TSV), delimiter='\t'))
embed_coords_all = {int(r['year']): (float(r['umap_x']), float(r['umap_y'])) for r in rows_umap if r['layer']=='embed'}

year_docs = defaultdict(list)
with open(RAW) as f:
    for line in f:
        row = json.loads(line)
        ds = row.get('date','')
        if not ds[:4].isdigit(): continue
        y = int(ds[:4])
        if 2010 < y < 2030 and y in year_vecs:
            year_docs[y].append(row)

def doc_vec(row):
    h = hashlib.sha256(row['text'].encode()).hexdigest()[:12]
    cache = CACHE_DIR / f'doc_{h}.npy'
    if cache.exists():
        return np.load(cache)
    sents = [s.strip() for s in row['text'].split('.') if len(s.strip()) > 20]
    if not sents:
        sents = [row['text'][:500]]
    v = np.mean(model.encode(sents, show_progress_bar=False), axis=0)
    np.save(cache, v)
    return v

year_list = sorted(year_vecs)
all_mat = np.vstack([year_vecs[y] for y in year_list])
reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, metric='cosine', n_components=2, random_state=42)
reducer.fit(all_mat)

# nearest year on the trajectory: distance in projected 2D
cm_year = defaultdict(int)
nearest = defaultdict(int)
within1 = defaultdict(int)
within2 = defaultdict(int)
tot = 0
errors = []

for y in sorted(year_docs):
    for row in year_docs[y]:
        pv = doc_vec(row).reshape(1,-1)
        pj = reducer.transform(pv)[0]
        dists = {yy: float(np.linalg.norm(pj - embed_coords_all[yy])) for yy in year_list}
        best = min(dists, key=dists.get)
        nearest[best] += 1
        off = abs(best - y)
        if off == 0:
            cm_year[y] += 1
        if off <= 1:
            within1[y] += 1
        if off <= 2:
            within2[y] += 1
        if off > 0:
            errors.append((y, best, dists[best]))
        tot += 1

print(f'Total evaluated: {tot}')
print(f'Exact-year correct: {sum(cm_year.values())}/{tot} = {sum(cm_year.values())/tot:.1%}')
print(f'Within ±1 year:     {sum(within1.values())}/{tot} = {sum(within1.values())/tot:.1%}')
print(f'Within ±2 years:    {sum(within2.values())}/{tot} = {sum(within2.values())/tot:.1%}')
print('\nPer-year exact matches:')
for y in sorted(year_docs):
    n = len(year_docs[y])
    print(f'  {y}: {cm_year[y]}/{n} = {cm_year[y]/n:.0%}')
print('\nBiggest errors (true→predicted):')
for y, b, d in sorted(errors, key=lambda x: abs(x[1]-x[0]), reverse=True)[:10]:
    print(f'  true={y} pred={b} dist={d:.2f} off={abs(b-y)}')
