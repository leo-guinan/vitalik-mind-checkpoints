#!/usr/bin/env python3
"""
Concept extraction and mental model graph builder.
Turns text into a concept-cooccurrence graph.
"""
import re, math, collections
from pathlib import Path
from datetime import datetime

# Stopwords + domain-specific noise
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
    "ring", "stealth", "tornado", "mixer", " dall", "ecrecover",
    "predicate", "access", "list", "rlp", "ssz", "kzg", "fiateleia",
    "danksharding", "proto", "danksharding", "blob", "fee", "market",
    "1559", "basefee", "priority", "tip", "burn", "issuance", "staking",
    "slash", "inactivity", "leak", "reward", "apy", "lido", "rocket",
    "rpl", "solo", "node", "client", "consensus", "execution", "engine",
    "api", "json", "rpc", "graphql", "rest", "p2p", "discv5", "enr",
    "bootnode", "boot", "peer", "gossip", "subnet", "attestation",
    "proposal", "block", "slot", "epoch", "checkpoint", "finality",
    "justified", "finalized", "reorg", "orphan", "uncle", "grandparent",
    "parent", "child", "root", "hash", "digest", "sighash", "tx", "txn",
    "transaction", "call", "delegatecall", "staticcall", "create",
    "create2", "selfdestruct", "log", "event", "topic", "index",
    "storage", "calldata", "memory", "stack", "pc", "opcode", "gas",
    "gaslimit", "gasused", "intrinsic", "refund", "stipend",
    "precompile", "ecrecover", "sha256", "ripemd160", "identity",
    "modexp", "ecadd", "ecmul", "ecpairing", "blake2f",
    "compression", "decompression",
}

def tokenize(text: str):
    text = text.lower()
    # Keep technical terms intact if they appear in compound
    text = re.sub(r'[-/]', ' ', text)
    tokens = re.findall(r'[a-z0-9_+/]+', text)
    return [t for t in tokens if len(t) > 2 and t not in STOP]

def extract_concepts(text: str, window: int = 5) -> dict:
    tokens = tokenize(text)
    # Boost known tech terms
    concepts = []
    for t in tokens:
        if t in TECH_TERMS:
            concepts.append(t)
        elif len(t) > 5:
            concepts.append(t)
    
    # Build co-occurrence graph
    cooc = collections.defaultdict(int)
    for i, c1 in enumerate(concepts):
        for j in range(i+1, min(i+window, len(concepts))):
            c2 = concepts[j]
            if c1 == c2:
                continue
            pair = tuple(sorted([c1, c2]))
            cooc[pair] += 1
    
    # Concept frequency
    freq = collections.Counter(concepts)
    
    return {
        "concepts": dict(freq.most_common(200)),
        "cooccurrence": {f"{a}|{b}": c for (a,b), c in sorted(cooc.items(), key=lambda x: -x[1])[:500]},
    }

def save_checkpoint(concepts: dict, year: int, source_count: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"checkpoint_{year}.json"
    payload = {
        "year": year,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_docs": source_count,
        "concept_count": len(concepts["concepts"]),
        "edge_count": len(concepts["cooccurrence"]),
        "concepts": concepts["concepts"],
        "cooccurrence": concepts["cooccurrence"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ingest import OUT
    
    corpora = collections.defaultdict(list)
    if OUT.exists():
        for line in open(OUT):
            row = json.loads(line)
            year = row.get("year")
            if year:
                corpora[year].append(row["text"])
    
    out_dir = Path(__file__).resolve().parents[1] / "data" / "checkpoints"
    for year in sorted(corpora):
        combined = "\n".join(corpora[year])
        concepts = extract_concepts(combined)
        path = save_checkpoint(concepts, year, len(corpora[year]), out_dir)
        print(f"Saved {path}")
