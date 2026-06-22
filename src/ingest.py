#!/usr/bin/env python3
"""
Ingest Vitalik's verified writings via vitalik.eth.limo (the live mirror of vitalik.ca).

Discovers all posts from the index, fetches full text, writes a single JSONL corpus.
Falls back to the pre-fetched vitalik_corpus_limo.jsonl if network is unavailable.
"""
import json, re, time, os, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "vitalik_corpus.jsonl"
LIMO_FALLBACK = RAW / "vitalik_corpus_limo.jsonl"

BASE = "https://vitalik.eth.limo"


def curl_get(url: str, timeout: int = 25) -> str:
    r = subprocess.run(
        ["curl", "-sL", "--max-time", str(timeout), url],
        capture_output=True, text=True, timeout=timeout + 5,
    )
    return r.stdout


def fetch_index_urls() -> dict:
    """Return { (year, slug): {date, path, category} } from the limo index."""
    html = curl_get(BASE)
    links = re.findall(
        r'href="(\./(\w+)/(\d{4})/(\d{2})/(\d{2})/([^"]+)\.html)"', html
    )
    seen = {}
    for href, cat, y, m, d, slug in links:
        key = (y, slug)
        if key not in seen:
            seen[key] = {
                "path": f"./{cat}/{y}/{m}/{d}/{slug}.html",
                "date": f"{y}-{m}-{d}",
                "category": cat,
                "slug": slug,
            }
    return seen


def fetch_post(url_path: str) -> dict | None:
    rel = url_path.lstrip("./")
    url = f"{BASE}/{rel}"
    html = curl_get(url)
    if not html or len(html) < 500 or "failed to resolve" in html[:200]:
        return None

    # Title
    title_m = re.search(r"<title>\s*\[Mirror\]\s*(.*?)\s*</title>", html)
    title = title_m.group(1).strip() if title_m else ""
    if not title:
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
        if h1:
            title = re.sub(r"<[^>]+>", "", h1.group(1)).strip()

    # Content from <p> tags inside #doc
    start_doc = html.find('<div id="doc"')
    if start_doc == -1:
        return None
    chunks = []
    sp = start_doc
    while True:
        po = html.find("<p>", sp)
        if po == -1:
            break
        pc = html.find("</p>", po)
        if pc == -1:
            break
        chunk = html[po + 3 : pc]
        chunk = re.sub(r"<[^>]+>", " ", chunk)
        chunk = chunk.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        chunk = re.sub(r"&#\d+;", "", chunk)
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if len(chunk) > 5:
            chunks.append(chunk)
        sp = pc + 4

    if not chunks:
        return None

    # Drop mirror preamble if present
    if chunks[0].startswith("This is a mirror of the post"):
        chunks = chunks[1:]

    text = " ".join(chunks)
    return {
        "title": title,
        "url": url_path,
        "source": url,
        "author": "vbuterin",
        "word_count": len(text.split()),
        "char_count": len(text),
        "text": text,
    }


def run(target_docs: int = 500) -> None:
    print(f"=== Vitalik Mind Corpus | target ≥{target_docs} docs ===\n")

    # Use pre-fetched limo corpus if it exists and we don't need to re-fetch
    if LIMO_FALLBACK.exists() and LIMO_FALLBACK.stat().st_size > 10_000:
        print(f"[reuse] {LIMO_FALLBACK} ({LIMO_FALLBACK.stat().st_size / 1024:.0f} KB)")
        existing_urls = set()
        if OUT.exists():
            existing_urls = {
                json.loads(l).get("url", "")
                for l in open(OUT)
            }
        new_rows = []
        for line in open(LIMO_FALLBACK):
            row = json.loads(line)
            if row.get("url", "") not in existing_urls:
                new_rows.append(row)
        if new_rows:
            with open(OUT, "a") as f:
                for row in new_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total = sum(1 for _ in open(OUT))
            print(f"[merged] added {len(new_rows)} docs, total now {total}")
            print(f"File: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
        else:
            total = sum(1 for _ in open(OUT))
            print(f"[skip] corpus already up to date ({total} docs)")
        return

    # Fresh discovery mode
    print("[fetch] discovering posts from index …")
    urls = fetch_index_urls()
    print(f"[index] {len(urls)} unique posts")
    years: dict[str, int] = {}
    for v in urls.values():
        years[v["date"][:4]] = years.get(v["date"][:4], 0) + 1
    for y in sorted(years):
        print(f"  {y}: {years[y]} posts")

    rows, written, errors = [], 0, 0
    skip_urls = set()
    if OUT.exists():
        skip_urls = {
            json.loads(l).get("url", "")
            for l in open(OUT)
        }

    for key, meta in sorted(urls.items(), key=lambda x: x[1]["date"]):
        path = meta["path"]
        if path in skip_urls:
            continue
        try:
            row = fetch_post(path)
            if not row:
                errors += 1
                continue
            row["date"] = meta["date"]
            row["fetched_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
            written += 1
            print(
                f"  [ok] {row['date']} | {row['word_count']:5d} w | {row['title'][:60]}"
            )
            time.sleep(0.15)
        except Exception as e:
            errors += 1
            print(f"  [err] {path}: {e}")

    if rows:
        with open(OUT, "a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(1 for _ in open(OUT)) if OUT.exists() else 0
    print(f"\nDone: {written} new, {errors} errors, total {total} docs")
    if OUT.exists():
        print(f"File: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
