"""
probe.py — score arbitrary text against Vitalik year vectors.
Usage:
  echo "some text" | python3 src/probe.py
  python3 src/probe.py "some text"
  python3 src/probe.py --file notes.txt
  # interactive
  python3 src/probe.py --interactive
"""
import sys, argparse, warnings
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path('.').resolve()
EMB_DIR = ROOT / 'data/embeddings'
CACHE_DIR = ROOT / 'data/doc_embeddings'

warnings.filterwarnings('ignore')

model = SentenceTransformer('all-MiniLM-L6-v2')

year_vecs = {}
for p in sorted(EMB_DIR.glob('year_*.npy')):
    y = int(p.stem.split('_')[1])
    year_vecs[y] = np.load(p)

year_list = sorted(year_vecs)
year_norms = {}
for y, v in year_vecs.items():
    n = np.linalg.norm(v)
    year_norms[y] = n
    year_vecs[y] = v / (n + 1e-9)

mean_centroid = np.mean(np.vstack([year_vecs[y] for y in year_list]), axis=0)
mean_centroid /= np.linalg.norm(mean_centroid) + 1e-9

era_map = {
    y: 'pre-merge' if y <= 2020 else 'post-merge' if y <= 2022 else 'd/acc-era'
    for y in year_vecs
}

CACHE_DIR.mkdir(exist_ok=True)

def doc_vec(text):
    h = hash(text.encode()) % (2**32)
    cache = CACHE_DIR / f'doc_{h}.npy'
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

def score(text):
    v = doc_vec(text)
    cos = {}
    for y in year_list:
        cos[y] = float(np.dot(v, year_vecs[y]))
    ranked = sorted(cos, key=cos.get, reverse=True)
    best_y = ranked[0]
    cos_mean = float(np.dot(v, mean_centroid))
    proximity = cos[best_y] - cos_mean
    return {
        'nearest_year': best_y,
        'era': era_map[best_y],
        'cos_sim': round(cos[best_y], 4),
        'proximity': round(proximity, 4),
        'top3': [(y, round(cos[y], 4), era_map[y]) for y in ranked[:3]]
    }

def format(r):
    lines = [
        f'nearest: {r["nearest_year"]} ({r["era"]})  cos={r["cos_sim"]}  prox={r["proximity"]}',
        'top 3:'
    ]
    for y, c, e in r['top3']:
        lines.append(f'  {y} {e}  cos={c}')
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser(description='Probe Vitalik trajectory with arbitrary text')
    ap.add_argument('text', nargs='?', default='', help='text to score')
    ap.add_argument('--file', '-f', help='read text from file')
    ap.add_argument('--interactive', '-i', action='store_true', help='interactive mode')
    args = ap.parse_args()

    if args.interactive:
        print('probe mode — paste text, ctrl-d to score, ctrl-c to quit')
        print('-' * 60)
        while True:
            try:
                text = input('> ')
                if not text.strip():
                    continue
                print(format(score(text)))
                print('-' * 60)
            except (EOFError, KeyboardInterrupt):
                print('\nexit.')
                break
        return

    if args.file:
        text = Path(args.file).read_text(errors='replace')
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        ap.print_help()
        return

    if not text.strip():
        print('empty input.')
        return
    print(format(score(text)))

if __name__ == '__main__':
    main()
