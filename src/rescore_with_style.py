"""
rescore_with_style.py — add style fingerprint proximity to existing scored corpora.
For each scored doc, embed its text (or use stored text) and compute L2 distance
to each year's style profile. Convert to a style proximity score and re-rank.
"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, '/Users/leoguinan/Projects/vitalik-mind-checkpoints/src')
from style import extract

ROOT = Path('/Users/leoguinan/Projects/vitalik-mind-checkpoints')
STYLE_DIR = ROOT / 'data/style_profiles'

def load_profile(year):
    p = STYLE_DIR / f'style_{year}.json'
    if not p.exists():
        return None
    return json.load(open(p))

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

def style_proximity(text):
    fx = extract(text)
    cvec = flatten(fx)
    dists = {}
    for year in range(2014, 2027):
        p = load_profile(year)
        if not p:
            continue
        pvec = flatten(p)
        d = math.sqrt(sum((a-b)**2 for a,b in zip(cvec, pvec)))
        dists[year] = d
    if not dists:
        return 0.0, 0
    best_year = min(dists, key=dists.get)
    best_dist = dists[best_year]
    # convert to 0-1 proximity: small distance -> high proximity
    prox = max(0.0, 1.0 - best_dist / 300.0)
    return prox, best_year

def rescore(path, out_path):
    rows = [json.loads(l) for l in open(path)]
    out = []
    missed = 0
    for r in rows:
        txt = ((r.get('title','') or '') + ' ' + (r.get('text','') or '') + ' ' + (r.get('body','') or '')).strip()
        if not txt:
            missed += 1
            continue
        sp, sp_year = style_proximity(txt)
        r['style_proximity'] = round(sp, 4)
        r['style_nearest_year'] = sp_year
        out.append(r)
    with open(out_path, 'w') as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'rescored {len(out)} / {len(rows)} (missed {missed}) -> {out_path}')
    return out_path

def hybrid_rank(path, out_path, weights=(0.35, 0.35, 0.30)):
    rows = [json.loads(l) for l in open(path)]
    prox = [r.get('proximity', 0) for r in rows]
    ts = [r.get('topic_score', 0) for r in rows]
    sp = [r.get('style_proximity', 0) for r in rows]
    def norm(vals):
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        return [(v - lo) / span for v in vals]
    np = norm(prox)
    nt = norm(ts)
    ns = norm(sp)
    out = []
    for i, r in enumerate(rows):
        rr = dict(r)
        rr['hybrid'] = round(weights[0]*np[i] + weights[1]*nt[i] + weights[2]*ns[i], 4)
        rr['norm_proximity'] = round(np[i], 4)
        rr['norm_topic'] = round(nt[i], 4)
        rr['norm_style'] = round(ns[i], 4)
        out.append(rr)
    out.sort(key=lambda x: x['hybrid'], reverse=True)
    with open(out_path, 'w') as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'hybrid-ranked {len(out)} -> {out_path}')
    return out_path

def report(path, top_n=30):
    rows = [json.loads(l) for l in open(path)]
    print(f'hybrid ranked: {path}')
    print('='*140)
    for i,r in enumerate(rows[:top_n],1):
        print('{i:2d}. hybrid={hy:.4f} prox={pr:.4f} topic={tp:.4f} style={st:.4f} year={yr} [{src}]'.format(
            i=i, hy=r['hybrid'], pr=r.get('proximity',0), tp=r.get('topic_score',0), st=r.get('style_proximity',0), yr=r.get('nearest_year','?'), src=r.get('source','')))
        print('    ' + r['title'][:100])
        hits = r.get('topic_hits', {})
        if hits:
            print('    hits: ' + ', '.join(f'{k}({v})' for k,v in hits.items()))
        if r.get('url'): print('    url: ' + r['url'][:110])
        if r.get('style_nearest_year'): print('    style_nearest_year: ' + str(r['style_nearest_year']))
        print()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('scored_jsonl', help='existing scored jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--top', type=int, default=30)
    args = ap.parse_args()
    rescore(args.scored_jsonl, args.out)
    if args.report:
        rank_path = args.out
        hybrid_rank(args.out, rank_path)
        report(rank_path, args.top)
