#!/usr/bin/env python3
"""
Match an arbitrary document against Vitalik mental-model checkpoints.
Outputs ranked years by structural similarity.
"""
import json, re, math, collections, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "data" / "checkpoints"

STOP = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each eg few
for from further get got had hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll
i'm i've if in into is isn't it it's its itself let's me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves
out over own per ph per/ perhaps re same sha she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those
through to too under until up us ve very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who
who's whom why why's will with won't would wouldn't yet you you'd you'll
you're you've your yours yourself yourselves ethereum also just may like
make many new now one see way well work year
""".split())

TECH_TERMS = {
    "evm", "eip", "pos", "pow", "dvt", "zk", "snark", "stark", "rollup",
    "shard", "beacon", "casper", "merge", "serenity", "frontier", "homestead",
    "byzantium", "constantinople", "istanbul", "muir", "glacier", "altair",
    "bellatrix", "capella", "deneb", "eip1559", "eip4844", "blob", "kzg",
    "verkle", "stateless", "witness", "merkle", "state", "account", "abi",
    "solidity", "viper", "vyper", "geth", "nethermind", "lighthouse",
    "prysm", "lodestar", "nimbus", "validator", "proposer", "builder",
    "sequencer", "l1", "l2", "arbitrum", "optimism", "zksync", "starknet",
    "polygon", "loopring", "aztec", "privacy", "anonymity", "signature",
    "ring", "stealth", "tornado", "mixer", "ecrecover",
    "predicate", "access", "list", "rlp", "ssz", "kzg",
    "danksharding", "blob", "fee", "market",
    "1559", "basefee", "priority", "tip", "burn", "issuance", "staking",
    "slash", "inactivity", "leak", "reward", "apy", "lido", "rocket",
    "rpl", "solo", "node", "client", "consensus", "execution",
    "graphql", "rest", "p2p", "discv5", "enr",
    "bootnode", "boot", "peer", "gossip", "subnet", "attestation",
    "proposal", "block", "slot", "epoch", "checkpoint", "finality",
    "justified", "finalized", "reorg", "orphan", "uncle",
    "parent", "child", "root", "hash", "digest", "sighash", "tx", "txn",
    "transaction", "call", "delegatecall", "staticcall", "create",
    "create2", "selfdestruct", "log", "event", "topic", "index",
    "storage", "calldata", "memory", "stack", "pc", "opcode", "gas",
    "gaslimit", "gasused", "intrinsic", "refund", "stipend",
    "precompile", "ecrecover", "sha256", "ripemd160", "identity",
    "modexp", "ecadd", "ecmul", "ecpairing", "blake2f",
    "compression", "decompression", "token", "erc20", "erc721", "erc1155",
    "dao", "defi", "amm", "liquidity", "swap", "pool", "yield", "lend",
    "borrow", "collateral", "oracle", "chainlink", "band",
}

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r'[-/]', ' ', text)
    tokens = re.findall(r'[a-z0-9_+/]+', text)
    return [t for t in tokens if len(t) > 2 and t not in STOP]

def extract_concepts(text: str, window: int = 5):
    tokens = tokenize(text)
    concepts = [t for t in tokens if t in TECH_TERMS or len(t) > 5]
    cooc = collections.defaultdict(int)
    for i, c1 in enumerate(concepts):
        for j in range(i + 1, min(i + window, len(concepts))):
            c2 = concepts[j]
            if c1 == c2:
                continue
            pair = tuple(sorted([c1, c2]))
            cooc[pair] += 1
    freq = collections.Counter(concepts)
    return freq, cooc

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def edge_similarity(a: dict, b: dict, top_n: int = 500) -> float:
    a_top = set(k for k, _ in sorted(a.items(), key=lambda x: -x[1])[:top_n])
    b_top = set(k for k, _ in sorted(b.items(), key=lambda x: -x[1])[:top_n])
    return jaccard(a_top, b_top)

def load_checkpoints():
    cps = {}
    for p in CHECKPOINT_DIR.glob("checkpoint_*.json"):
        d = json.load(open(p))
        cps[d["year"]] = {
            "concepts": set(d["concepts"].keys()),
            "cooccurrence": d["cooccurrence"],
            "source_docs": d["source_docs"],
            "path": str(p),
        }
    return cps

def main():
    if len(sys.argv) < 2:
        print("Usage: match.py <text|file> [--top-n N] [--window W]")
        sys.exit(1)

    arg = sys.argv[1]
    if arg.startswith("--"):
        print("Usage: match.py <text|file> [--top-n N] [--window W]")
        sys.exit(1)

    # Parse optional args
    top_n = 500
    window = 5
    if "--top-n" in sys.argv:
        i = sys.argv.index("--top-n")
        top_n = int(sys.argv[i + 1])
    if "--window" in sys.argv:
        i = sys.argv.index("--window")
        window = int(sys.argv[i + 1])

    if Path(arg).exists():
        text = open(arg).read()
        label = arg
    else:
        text = arg
        label = "stdin_text"

    if len(text) < 50:
        print("Text too short for meaningful matching.")
        sys.exit(1)

    freq, cooc = extract_concepts(text, window=window)
    concept_vec = set(freq.keys())

    cps = load_checkpoints()
    if not cps:
        print("No checkpoints found in", CHECKPOINT_DIR)
        sys.exit(1)

    scores = []
    for year, cp in sorted(cps.items()):
        c_jacc = jaccard(concept_vec, cp["concepts"])
        e_sim = edge_similarity(cooc, cp["cooccurrence"], top_n=top_n)
        combined = 0.4 * c_jacc + 0.6 * e_sim
        scores.append({
            "year": year,
            "combined": round(combined, 4),
            "concept_jaccard": round(c_jacc, 4),
            "edge_similarity": round(e_sim, 4),
            "source_docs": cp["source_docs"],
        })

    scores.sort(key=lambda x: x["combined"], reverse=True)

    print(f"\nMatched: {label} ({len(text)} chars, {len(freq)} concepts extracted)\n")
    print(f"{'Rank':<5} {'Year':<6} {'Combined':<10} {'Concept J':<12} {'Edge Sim':<10} {'Docs':<6}")
    print("-" * 55)
    for i, s in enumerate(scores, 1):
        print(f"{i:<5} {s['year']:<6} {s['combined']:<10} {s['concept_jaccard']:<12} {s['edge_similarity']:<10} {s['source_docs']:<6}")

    best = scores[0]
    print(f"\nBest match year: {best['year']} (combined={best['combined']})")

if __name__ == "__main__":
    main()
