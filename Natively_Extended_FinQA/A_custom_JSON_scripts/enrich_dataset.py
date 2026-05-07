"""
This was used to create the longer "enriched" versions of the datasets, 
which were then used for the training, dev, and test runs. 
-----------------
ENRICH DATASET
-----------------
Reads the original flat finance QA dataset JSON and produces a new JSON file
identical in structure, except each question's "pre_text" and "post_text" fields
are expanded with content drawn from 9 other entries that share the same
company ticker (same ticker, any year). If the company pool is too small,
the remainder is filled randomly from the global pool. All 9 sources must come from
distinct filenames, and none may match the current entry's own filename.

The insertion position of the 9 extra blocks (before vs. after the original text) 
is randomised independently for pre_text and post_text, so the model cannot 
learn to identify relevant context from location.

Usage:
    python enrich_dataset.py <input_file> <output_file> [--seed SEED]

Examples:
    python enrich_dataset.py train.json train_enriched.json
    
    # or with a fixed seed for reproducibility
    python enrich_dataset.py train.json train_enriched.json --seed 42
"""

import json
import random
import argparse
import copy
from collections import defaultdict


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enrich a finance QA JSON dataset with expanded pre_text and post_text fields."
    )
    parser.add_argument("input_file",  help="Path to the original flat JSON file.")
    parser.add_argument("output_file", help="Path for the new enriched JSON file.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional integer seed for full reproducibility."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Lookup structure builder
# ---------------------------------------------------------------------------

def build_pools(data: list[dict]) -> dict[str, list[dict]]:
    """
    Build a dict mapping each company ticker to all entries for that ticker.
    E.g., { "IP": [...], "ADI": [...], ... }
    The global pool is just `data` itself and is handled at call sites.
    """
    company_pool = defaultdict(list)
    for entry in data:
        ticker = entry["filename"].split("/")[0]
        company_pool[ticker].append(entry)
    return company_pool


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_sources(
    current_filename: str,
    ticker: str,
    company_pool: dict[str, list[dict]],
    global_pool: list[dict],
    rng: random.Random,
    n: int = 9,
) -> list[dict]:
    """
    Return n entries with filenames that are:
      - not equal to current_filename
      - all distinct from each other
    Priority: same-ticker entries first, global pool as fallback.
    """
    used_filenames = {current_filename}
    selected = []

    # --- Phase 1: same-ticker pool (any year) ---
    company_candidates = [
        e for e in company_pool[ticker]
        if e["filename"] != current_filename
    ]
    # Shuffle so we draw randomly without replacement
    rng.shuffle(company_candidates)
    for entry in company_candidates:
        if len(selected) >= n:
            break
        if entry["filename"] not in used_filenames:
            selected.append(entry)
            used_filenames.add(entry["filename"])

    # --- Phase 2: global fallback if company pool was not large enough ---
    if len(selected) < n:
        global_candidates = [
            e for e in global_pool
            if e["filename"] not in used_filenames
        ]
        rng.shuffle(global_candidates)
        for entry in global_candidates:
            if len(selected) >= n:
                break
            if entry["filename"] not in used_filenames:
                selected.append(entry)
                used_filenames.add(entry["filename"])

    return selected


# ---------------------------------------------------------------------------
# Field expansion
# ---------------------------------------------------------------------------

def expand_field(
    original: list[str],
    blocks: list[list[str]],
    k: int,
) -> list[str]:
    """
    Interleave 9 extra blocks around the original list.

    k  : number of blocks placed BEFORE the original (0-9 inclusive)
    The remaining (9 - k) blocks go AFTER the original.

    All blocks are flattened into a single list of strings.
    """
    before = []
    for block in blocks[:k]:
        before.extend(block)

    after = []
    for block in blocks[k:]:
        after.extend(block)

    return before + original + after


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

def enrich(data: list[dict], company_pool: dict, rng: random.Random) -> list[dict]:
    enriched = []

    for entry in data:
        # Deep-copy so the original dict is never mutated
        new_entry = copy.deepcopy(entry)

        current_filename = entry["filename"]
        ticker = current_filename.split("/")[0]

        # Sample 9 unique-filename sources
        sources = sample_sources(
            current_filename=current_filename,
            ticker=ticker,
            company_pool=company_pool,
            global_pool=data,
            rng=rng,
        )

        # Pull pre_text and post_text blocks from each source
        pre_blocks  = [s["pre_text"]  for s in sources]
        post_blocks = [s["post_text"] for s in sources]

        # Two independent random split points, freshly drawn per entry
        k_pre  = rng.randint(0, 9)
        k_post = rng.randint(0, 9)

        new_entry["pre_text"]  = expand_field(entry["pre_text"],  pre_blocks,  k_pre)
        new_entry["post_text"] = expand_field(entry["post_text"], post_blocks, k_post)

        enriched.append(new_entry)

    return enriched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seed the RNG (None = non-deterministic)
    rng = random.Random(args.seed)

    print(f"Loading input file: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries.")

    print("Building lookup pools...")
    company_pool = build_pools(data)
    print(f"  Found {len(company_pool)} unique tickers.")

    print("Enriching entries...")
    enriched = enrich(data, company_pool, rng)

    print(f"Writing output file: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f)

    print(f"Done. {len(enriched)} enriched entries written to {args.output_file}")


if __name__ == "__main__":
    main()
