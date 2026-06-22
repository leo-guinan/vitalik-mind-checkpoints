#!/usr/bin/env python3
"""
Stream 2 — Style / Meme / Compression fingerprint.

Captures:
  * Compression patterns — how Vitalik packages ideas for spread
  * Meme vocabulary — coined or popularized term inventory
  * Rhetorical tics — sentence-openers, structural forms
  * Readability fingerprint — length, density, question density, etc.
  * Analogy density — metaphor rate as a rhetorical signal
"""
import re, math, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STYLE_DIR = DATA / "style_profiles"
STYLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Signature phrases grouped by category ────────────────────────────────────
SIGNATURE = {
    "compression_framing": [
        r"\ba (shallow|quick|simple) (dive|explanation|refresher|introduction)\b",
        r"\btrigger warning: math\b",
        r"\bthe purpose of this post is not\b",
        r"\bi used to (think|believe|prefer)\b",
        r"\bi was wrong about\b",
        r"\bi (genuinely|honestly) have no idea\b",
        r"\b(?:let me|i'll) explain by analogy\b",
    ],
    "numbered_structure": [
        r"(?<!\w)(?:\d+\.)\s+[A-Z]",               # "1. Something"
        r"step\s+\d+",
    ],
    "X_vs_Y_frame": [
        r"\b(is|versus|vs\.?)\b.*?\b(against|or|vs\.?)\b",
        r"\bx is (?:the new y|y for z)\b",
    ],
    "possible_futures": [
        r"\bpossible futures of the ethereum protocol\b",
        r"\bpart \d+:?\s*(?:the\s+)?\w+\b",
    ],
    "self_ref": [
        r"\bat some point this decade\b",
        r"\b~?\d+ to ~?\d+\s+(?:documents|words|posts)\b",
        r"\bincredibly oversized nutshell\b",
        r"\bi support(?:s)? (?:it )?only if\b",
        r"\bdegen communism\b",
        r"\bd/acc\b",
    ],
    "demystification": [
        r"\bthis (?:is|was) (?:a|my) (?:published|public|personal)\b",
        r"\bthe (?:real|actual|true)\b.*?\bis\b",
        r"\b(?:here is|here's)\s+(?:a|my|the)\s+(?:simple|one|quick)\b",
    ],
}

# Meme vocabulary — terms Vitalik coined or canonized, lowercased
MEME_VOCAB = {
    "d/acc", "acc", "plurality", "soulbound", "verkle", "kzg",
    "danksharding", "proto", "danksharding", "eip4844", "eip1559",
    "basefee", "blob",
    "possible futures", "the merge", "the surge", "the verge",
    "the purge", "the scourge", "the splurge",
    "network state", "exit game", "zk-evm",
    "open source", "copyleft", "permissive license",
    "community notes", "proof of personhood",
    "social recovery", "multi-sig", "account abstraction",
    "based rollup", "preconfirmation", "builder constraint",
    "multidimensional", "fee market", "state expiry",
    "stateless", "light client", "sync committee",
    "regenesis", "weak subjectivity",
    "layer 2", "layer 3", "validium", "volition",
    "stealth address", "stealth meta-address",
    "quadratic voting", "quadratic funding",
    "plutocracy", "lunar society", "moloch",
    "exit to community", "credibly neutral",
}

# Artifact noise — filter these before style analysis
ARTIFACT_NOISE = [
    r"bafybeig\w{40,}",             # IPFS CIDs
    r"//twitter\.com/\w+/status/\d+",
    r"https?://[^\s)]+",            # bare URLs
    r"\./general/\d{4}/\d{2}/\d{2}/\w+\.html",
]

# Analogy signals — metaphorical framing language
ANALOGY_SIGNALS = [
    r"\b(?:like|as if|as though|imagine)\b.*?\b(?:but|except|in|where)\b",
    r"\b(?:think of|compare to|analogous to|similar to)\b",
    r"\b(?:the x of y)\b",
    r"\b(?:y is (?:the|a|an)\s+)\b.*?\b(?:of|for)\b",
]

# ── Extraction ───────────────────────────────────────────────────────────────

def extract(text: str) -> dict:
    text_lower = text.lower()
    words = text_lower.split()
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    # Strip artifact patterns before analysis
    for pat in ARTIFACT_NOISE:
        text_lower = re.sub(pat, " ", text_lower)
    words = text_lower.split()

    features: dict = {}

    # 1. Compression signature counts
    compression = {}
    for cat, patterns in SIGNATURE.items():
        n = 0
        for p in patterns:
            n += len(re.findall(p, text_lower))
        compression[cat] = n
    features["compression"] = compression
    features["compression_total"] = sum(compression.values())

    # 2. Meme vocabulary density
    meme_hits = [v for v in MEME_VOCAB if v in text_lower]
    features["meme_vocab_hits"] = len(meme_hits)
    features["meme_vocab_density"] = len(meme_hits) / max(len(words), 1)
    features["meme_vocab_list"] = sorted(set(meme_hits))[:40]

    # 3. Readability fingerprint
    sentences = sents if sents else [text[:200]]
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    avg_char_per_word = sum(len(w) for w in words) / max(len(words), 1)
    pct_questions = sum(1 for s in sentences if s.endswith("?")) / max(len(sentences), 1)
    pct_first_person = sum(
        1 for w in words if w in {"i", "me", "my", "mine", "we", "us", "our"}
    ) / max(len(words), 1)
    pct_math = len(re.findall(r'\$[^$]+\$|\\\([^)]+\\\)', text)) / max(len(sentences), 1)

    features["readability"] = {
        "avg_sentence_len": round(avg_sent_len, 2),
        "avg_char_per_word": round(avg_char_per_word, 4),
        "pct_questions": round(pct_questions, 4),
        "pct_first_person": round(pct_first_person, 4),
        "pct_math_sentences": round(pct_math, 4),
        "sentence_count": len(sentences),
        "word_count": len(words),
    }

    # 4. Analogy density
    analogy_hits = []
    for p in ANALOGY_SIGNALS:
        analogy_hits.extend(re.findall(p, text_lower))
    features["analogy_density"] = len(analogy_hits) / max(len(sentences), 1)

    # 5. Parenthetical density (aside-signal)
    paren_hits = re.findall(r'\([^)]{10,}\)', text)
    features["parenthetical_density"] = len(paren_hits) / max(len(sentences), 1)

    # 6. Compression score (composite) — higher = more compressed, more spreadable
    features["spread_score"] = round(
        features["compression_total"] * 0.5
        + features["meme_vocab_density"] * 20
        + features["analogy_density"] * 3
        + pct_questions * 2
        + pct_first_person * 1.5
        + features["parenthetical_density"] * 1,
        4,
    )

    return features


def save_profile(features: dict, year: int, metadata: dict, out_dir: Path = STYLE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"style_{year}.json"
    payload = {
        "year": year,
        "generated_at": metadata.get("generated_at", ""),
        "doc_count": metadata.get("doc_count", 0),
        **features,
    }
    with open(path, "w") as f:
        import json
        json.dump(payload, f, indent=2)
    return path


if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ingest import OUT

    corpora: dict = collections.defaultdict(list)
    if OUT.exists():
        for line in open(OUT):
            row = json.loads(line)
            year = row.get("date", "")[:4]
            if year.isdigit():
                corpora[int(year)].append(row["text"])

    out_dir = STYLE_DIR
    for year in sorted(corpora):
        combined = "\n".join(corpora[year])
        fx = extract(combined)
        path = save_profile(fx, year, {"doc_count": len(corpora[year]), "generated_at": ""})
        top_memes = fx["meme_vocab_list"][:10]
        print(f"{year}: spread_score={fx['spread_score']:.3f}  memes={top_memes}  docs={len(corpora[year])}  -> {path.name}")
    print(f"\nSaved {len(corpora)} style profiles to {out_dir}")
