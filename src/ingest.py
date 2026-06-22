#!/usr/bin/env python3
"""
Ingest Vitalik's known writings with temporal markers.
Direct URL strategy (avoids unreliable author-tag scraping).
"""
import json, re, time, os, sys
from pathlib import Path
from datetime import datetime, timezone
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "vitalik_corpus.jsonl"

UA = "Mozilla/5.0 (compatible; mind-checkpoint/0.2)"

# Curated seed URLs: Vitalik Buterin verified writings
SEED_URLS = [
    # ethresear.ch vbuterin posts (from search results + known canonical posts)
    ("https://ethresear.ch/t/hyper-scaling-state-by-creating-new-forms-of-state/24052", 2026),
    ("https://ethresear.ch/t/combining-preconfirmations-with-based-rollups-for-synchronous-composability/23863/1", 2026),
    ("https://ethresear.ch/t/building-index-tracking-assets-on-top-of-options-instead-of-debt/25036", 2026),
    ("https://ethresear.ch/t/native-dvt-for-ethereum-staking/23894/1", 2026),
    ("https://ethresear.ch/t/a-local-node-favoring-delta-to-the-scaling-roadmap/22368", 2025),
    ("https://ethresear.ch/t/sticking-to-8192-signatures-per-slot-post-ssf-how-and-why/17989", 2023),
    ("https://ethresear.ch/t/supporting-decentralized-staking-through-more-anti-correlation-incentives/19116/1", 2024),
    ("https://ethresear.ch/t/proposer-block-builder-separation-friendly-fee-market-designs/9725", 2021),
    ("https://ethresear.ch/t/future-proof-shard-and-history-access-precompiles/9781", 2021),
    ("https://ethresear.ch/t/maximally-simple-account-abstraction-without-gas-refunds/5007", 2019),
    # Historical Ethereum blog posts (known Vitalik authorship, from blog index)
    ("https://blog.ethereum.org/2014/01/15/slasher-a-punitive-proof-of-stake-algorithm/", 2014),
    ("https://blog.ethereum.org/2014/03/20/the-latest-evm-ethereum-is-a-trust-free-closure-system/", 2014),
    ("https://blog.ethereum.org/2014/05/15/long-range-attacks-the-serious-problem-with-adaptive-proof-of-work/", 2014),
    ("https://blog.ethereum.org/2014/09/17/scalability-part-1-building-top/", 2014),
    ("https://blog.ethereum.org/2014/11/25/proof-stake-learned-love-weak-subjectivity/", 2014),
    ("https://blog.ethereum.org/2015/04/13/visions-part-1-the-value-of-blockchain-technology/", 2015),
    ("https://blog.ethereum.org/2015/04/27/visions-part-2-the-problem-of-trust/", 2015),
    ("https://blog.ethereum.org/2015/08/28/on-anti-pre-revelation-games/", 2015),
    ("https://blog.ethereum.org/2016/07/15/to-fork-or-not-to-fork/", 2016),
    ("https://blog.ethereum.org/2017/09/14/on-transaction-fees-smart-contract-platforms-and-why-im-slightly-worried-about-the-future-of-ethereum/", 2017),
    # vitalik.ca (may fail DNS in sandbox; keep as attempt)
    ("https://vitalik.ca/general/2021/01/06/toll.html", 2021),
    ("https://vitalik.ca/general/2020/05/08/medical.html", 2020),
    ("https://vitalik.ca/general/2019/11/22/progress.html", 2019),
    ("https://vitalik.ca/general/2018/12/31/2018.html", 2018),
    ("https://vitalik.ca/general/2017/12/31/2017.html", 2017),
    ("https://vitalik.ca/general/2016/12/30/2016.html", 2016),
    ("https://vitalik.ca/general/2015/12/31/2015.html", 2015),
]

def fetch(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

def clean_html(html: str) -> str:
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.S)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.S)
    html = re.sub(r'<[^>]+>', ' ', html)
    return re.sub(r'\s+', ' ', html).strip()

def seen(url: str) -> bool:
    if not OUT.exists():
        return False
    return any(url in line for line in open(OUT))

def save(rows: list):
    with open(OUT, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def run(target_docs: int = 300):
    print(f"=== Vitalik Mind Corpus | seed URLs: {len(SEED_URLS)} ===")
    rows = []
    for url, year in SEED_URLS:
        if seen(url):
            print(f"[skip] {url}")
            continue
        try:
            html = fetch(url)
            text = clean_html(html)
            if len(text) < 200:
                print(f"[short] {url} -> {len(text)} chars")
                continue
            rows.append({
                "source": "vitalik_seed",
                "url": url,
                "title": url.split("/")[-1],
                "text": text,
                "author": "Vitalik Buterin",
                "year": year,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[ok] {year} | {len(text)} chars | {url[:60]}")
        except Exception as e:
            print(f"[err] {url}: {e}")
        time.sleep(0.3)
    save(rows)
    total = sum(1 for _ in open(OUT)) if OUT.exists() else 0
    print(f"\nCorpus total: {total} docs")
    if OUT.exists():
        print(f"File: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
