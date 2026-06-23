"""
author_fence.py — author-level UMAP fence with known set from trained corpus.
"""
import json, math, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '/Users/leoguinan/Projects/vitalik-mind-checkpoints/src')
from style import extract

import numpy as np
from sklearn.preprocessing import normalize
import umap
import matplotlib.pyplot as plt

ROOT = Path('/Users/leoguinan/Projects/vitalik-mind-checkpoints')
PSEUDO = ROOT / 'data/pseudonym'
RAW = ROOT / 'data/raw'

CORPORA = [
    PSEUDO / 'candidates_clean_384_scored.jsonl',
    PSEUDO / 'ethresearched_clean_384_scored.jsonl',
    PSEUDO / 'github_384_scored.jsonl',
    PSEUDO / 'bitcointalk_clean_384_scored.jsonl',
    RAW / 'vitalik_corpus.jsonl',
]

def load_rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]

def get_author(r):
    a = r.get('author') or r.get('username') or r.get('user')
    if not a:
        # trained corpus: all are Vitalik unless source marks otherwise
        if r.get('source','').startswith('vitalik') or r.get('date','').startswith('201'):
            a = 'vitalik'
        else:
            a = 'unknown'
    return str(a).lower().strip()

def flatten(fx):
    vec = []
    for cat in ['compression_framing', 'numbered_structure', 'X_vs_Y_frame', 'possible_futures', 'self_ref', 'demystification']:
        vec.append(fx['compression'].get(cat, 0))
    vec.append(fx['compression_total'])
    vec.append(fx['meme_vocab_hits'])
    vec.append(fx['meme_vocab_density'])
    vec.append(fx['readability']['avg_sentence_len'])
    vec.append(fx['readability']['avg_char_per_word'])
    vec.append(fx['readability']['pct_questions'])
    vec.append(fx['readability']['pct_first_person'])
    vec.append(fx['readability']['pct_math_sentences'])
    vec.append(fx['analogy_density'])
    vec.append(fx['parenthetical_density'])
    vec.append(fx['spread_score'])
    return vec

def pack_author_style(all_rows):
    texts = defaultdict(str)
    for r in all_rows:
        a = get_author(r)
        if a == 'unknown':
            continue
        txt = ((r.get('title','') or '') + ' ' + (r.get('text','') or '') + ' ' + (r.get('body','') or '')).strip()
        texts[a] += '\n' + txt
    out = {}
    for a, blob in texts.items():
        fx = extract(blob)
        out[a] = flatten(fx)
    return out

def pack_author_topic(all_rows):
    hits = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        a = get_author(r)
        if a == 'unknown':
            continue
        for k, v in (r.get('topic_hits') or {}).items():
            hits[a][k] += v
    out = {}
    for a, d in hits.items():
        total = sum(d.values()) or 1
        out[a] = {k: v/total for k, v in d.items()}
    return out

def build_author_vectors(all_rows):
    styles = pack_author_style(all_rows)
    topics = pack_author_topic(all_rows)
    authors = sorted(set(styles) | set(topics))
    print(f'authors with style/topic: {len(authors)}')
    vectors = {}
    known = set()
    for a in authors:
        s = styles.get(a, [0]*17)
        s = np.array(s, dtype=float)
        s = s / (np.linalg.norm(s) + 1e-9)
        t = topics.get(a, {})
        tkeys = ['snarks_starks','sharding_das','rollups','account_abstraction','consensus_pos',
                 'verkle_merkle','state_expiry','mev','validator_slashing','eips',
                 'cryptoecon','languages','ec_paired','plasma_state','defi']
        tv = np.array([t.get(k, 0) for k in tkeys], dtype=float)
        tv = tv / (np.linalg.norm(tv) + 1e-9)
        vec = np.concatenate([s, tv])
        vectors[a] = vec
        if a == 'vitalik' or a.startswith('vitalik') or a.endswith('buterin'):
            known.add(a)
    return vectors, topics, known

def umap_author_vectors(vectors, n_neighbors=None, min_dist=0.3):
    keys = sorted(vectors)
    n = len(keys)
    # UMAP requires n_neighbors < n_samples
    if n_neighbors is None:
        n_neighbors = max(2, min(5, n-1))
    mat = np.stack([vectors[k] for k in keys])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42)
    proj = reducer.fit_transform(mat)
    return {k: proj[i] for i, k in enumerate(keys)}

def plot_fence(projected, topics, author_docs, out_path, known):
    vitalik_pts = [projected[a] for a in projected if a in known]
    fig, ax = plt.subplots(figsize=(12, 9))
    xs, ys, colors, sizes = [], [], [], []
    for a, (x, y) in projected.items():
        xs.append(x); ys.append(y)
        if a in known:
            colors.append('crimson'); sizes.append(200)
        else:
            colors.append('steelblue'); sizes.append(60)
    ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.8, edgecolors='white', linewidth=0.5)
    for a, (x, y) in projected.items():
        if a in known:
            ax.annotate(a, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9, color='crimson', fontweight='bold')
    vitalik_topics = {'snarks_starks','sharding_das','consensus_pos','cryptoecon','rollups','plasma_state'}
    scores = {}
    for a, t in topics.items():
        scores[a] = sum(t.get(k, 0) for k in vitalik_topics)
    top10 = sorted(scores, key=scores.get, reverse=True)[:10]
    for a in top10:
        if a not in known:
            x, y = projected[a]
            ax.annotate(f'{a}\n({scores[a]:.2f})', (x, y), textcoords="offset points", xytext=(6, -8), fontsize=7, color='darkgreen')
    ax.set_title('Author UMAP fence — crimson=known Vitalik, green=top-10 topic match')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'wrote fence plot -> {out_path}')

def report_unknowns(projected, topics, author_docs, known):
    known_pts = [projected[a] for a in projected if a in known]
    if not known_pts:
        print('No known Vitalik authors in projected set')
        return
    kp = np.array(known_pts)
    centroid = kp.mean(axis=0)
    radius = max(np.linalg.norm(kp - centroid, axis=1)) * 1.5
    print(f'known Vitalik authors: {sorted(known)}')
    print(f'fence centroid: {centroid.round(4)} radius: {radius:.4f}')
    print()
    insiders = []
    for a, pt in projected.items():
        if a in known:
            continue
        d = np.linalg.norm(np.array(pt) - centroid)
        if d <= radius:
            insiders.append((a, d, pt))
    insiders.sort(key=lambda x: x[1])
    print(f'unknown authors INSIDE Vitalik fence: {len(insiders)}')
    for a, d, pt in insiders[:30]:
        t = topics.get(a, {})
        tv = sum(t.get(k, 0) for k in ['snarks_starks','sharding_das','consensus_pos','cryptoecon','rollups','plasma_state'])
        docs = author_docs.get(a, [])
        years = sorted(set(d.get('nearest_year','?') for d in docs))
        print(f'  {a:<30} dist={d:.4f} topic_v_score={tv:.2f} years={years} docs={len(docs)}')
        for doc in docs[:3]:
            print(f'    [{doc["source"]}] {doc["title"][:70]} | {doc.get("url","")[:60]}')

if __name__ == '__main__':
    rows = []
    for path in CORPORA:
        rows.extend(load_rows(path))
    print(f'total rows: {len(rows)}')
    vectors, topics, known = build_author_vectors(rows)
    if not known:
        known = {'vitalik'}
        print('using fallback known set: vitalik')
    proj = umap_author_vectors(vectors)
    author_docs = defaultdict(list)
    for r in rows:
        a = get_author(r)
        if a != 'unknown':
            author_docs[a].append({
                'title': r.get('title','')[:80],
                'url': r.get('url',''),
                'nearest_year': r.get('nearest_year','?'),
                'source': r.get('source',''),
            })
    plot_fence(proj, topics, author_docs, str(PSEUDO / 'author_fence.png'), known)
    report_unknowns(proj, topics, author_docs, known)
