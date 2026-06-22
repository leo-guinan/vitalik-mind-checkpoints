#!/usr/bin/env python3
"""
Two-stream unified pipeline.

  Stream 1 — mental model: concept graph  → data/checkpoints/checkpoint_YEAR.json
  Stream 2 — style/meme:  compression fingerprint → data/style_profiles/style_YEAR.json
  Stream 3 — embeddings:  vector cache        → data/embeddings/year_YEAR.npy

Usage:
  python3 src/pipeline.py                  # full run (ingest + all 3 streams)
  python3 src/pipeline.py embed "probe"    # just embed a probe
  python3 src/pipeline.py style            # just run style
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def full_run():
    print("=== STREAM 1: ingest ===")
    from ingest import run as ingest_run
    ingest_run(500)

    print("\n=== STREAM 2: style/meme ===")
    from style import main as style_main
    style_main()

    print("\n=== STREAM 3: embeddings ===")
    from embeddings import main as embed_main
    embed_main()

    print("\nPipeline complete.")


def probe_mode(query: str):
    from embeddings import probe, build_year_vectors
    years = build_year_vectors()
    result = probe(query, years)
    print(f"Probe: {query[:80]}")
    print(f"Best year: {result['top_year']}  cosine={result['top_score']}")
    for r in result["rankings"]:
        print(f"  {r['year']}: {r['cosine']:.6f}")


def style_mode():
    from style import main as style_main
    style_main()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "embed" or cmd == "probe":
            probe_mode(" ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Polynomial commitments replace state roots")
        elif cmd == "style":
            style_mode()
        elif cmd == "ingest":
            from ingest import run as ingest_run
            ingest_run(int(sys.argv[2]) if len(sys.argv) > 2 else 500)
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python3 pipeline.py [embed|style|ingest] [args]")
    else:
        full_run()
