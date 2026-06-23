"""
style_probe.py — compute style fingerprint for arbitrary text and find nearest year profile.
Uses the same extraction logic as style.py + simple feature vector distance.
"""
import json, sys, math
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'src'))
from style import extract

STYLE_DIR = ROOT / 'data/style_profiles'

def load_profile(year):
    p = STYLE_DIR / f'style_{year}.json'
    if not p.exists():
        return None
    return json.load(open(p))

def flatten_features(fx):
    """Convert nested style features to a flat vector comparable across docs."""
    # order matters — keep consistent
    vec = []
    # compression totals by category
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

def l2(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def compare(text):
    fx = extract(text)
    vec = flatten_features(fx)
    print('candidate style features:')
    print(f'  compression_total={fx["compression_total"]}')
    print(f'  meme_vocab_hits={fx["meme_vocab_hits"]} density={fx["meme_vocab_density"]:.4f}')
    print(f'  avg_sent_len={fx["readability"]["avg_sentence_len"]:.1f} avg_char_word={fx["readability"]["avg_char_per_word"]:.2f}')
    print(f'  pct_questions={fx["readability"]["pct_questions"]:.4f} pct_first_person={fx["readability"]["pct_first_person"]:.4f}')
    print(f'  pct_math={fx["readability"]["pct_math_sentences"]:.4f}')
    print(f'  analogy_density={fx["analogy_density"]:.4f} paren_density={fx["parenthetical_density"]:.4f}')
    print(f'  spread_score={fx["spread_score"]:.4f}')
    print(f'  memes: {fx["meme_vocab_list"][:15]}')
    print()
    
    distances = []
    for year in range(2014, 2027):
        p = load_profile(year)
        if not p:
            continue
        pvec = flatten_features(p)
        d = l2(vec, pvec)
        distances.append((year, d))
    distances.sort(key=lambda x: x[1])
    print('nearest year profiles by L2 distance:')
    for year, d in distances[:5]:
        print(f'  {year}: {d:.4f}')
    print()
    return fx, distances

if __name__ == '__main__':
    text = sys.stdin.read() if not sys.stdin.isatty() else ' '.join(sys.argv[1:])
    if not text.strip():
        print('usage: cat file | python3 src/style_probe.py')
        sys.exit(1)
    compare(text)
