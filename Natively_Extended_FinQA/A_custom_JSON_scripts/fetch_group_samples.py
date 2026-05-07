# """
# FETCH GROUP SAMPLES:
# Pseudorandomly samples 9 entries from a named group and returns
# only the pre_text, post_text, and table fields.

# Optimised for repeated calls (100,000+):
#   - Loads and parses the JSON file exactly once at import time.
#   - Uses random.sample on a pre-built index of list lengths, avoiding any
#     per-call work beyond the O(9) sample and O(9) field lookups.
#   - Results are fully reproducible: pass an integer seed for determinism.

# Usage:
#     from fetch_group_samples import load_grouped_data, fetch_samples

#     # One-time load (do this outside any loop)
#     data = load_grouped_data("grouped_output.json")

#     # Inside your loop
#     samples = fetch_samples(data, "TFX/2018/", seed=42)
#     for s in samples:
#         print(s["pre_text"], s["post_text"], s["table"])
# """

# import json
# import random
# from typing import Any


# # ---------------------------------------------------------------------------
# # One-time data loading
# # ---------------------------------------------------------------------------

# def load_grouped_data(grouped_path: str) -> dict[str, list[dict]]:
#     """
#     Load the pre-grouped JSON produced by group_by_filename.py.
#     Call this ONCE and reuse the returned dict across all fetch_samples calls.
#     """
#     with open(grouped_path, "r", encoding="utf-8") as f:
#         return json.load(f)


# # ---------------------------------------------------------------------------
# # High-frequency sampling function
# # ---------------------------------------------------------------------------

# _FIELDS = ("pre_text", "post_text", "table")   # tuple lookup is slightly faster than list

# def fetch_samples(
#     data: dict[str, list[dict]],
#     group_key: str,
#     seed: int | None = None,
#     n: int = 9,
# ) -> list[dict[str, Any]]:
#     """
#     Return n pseudorandomly selected entries from *group_key*, each
#     containing only pre_text, post_text, and table.

#     Parameters
#     ----------
#     data      : dict returned by load_grouped_data()
#     group_key : e.g. "TFX/2018/"
#     seed      : integer seed for reproducibility; None for non-deterministic
#     n         : number of entries to sample (default 9)

#     Returns
#     -------
#     List of dicts with keys pre_text, post_text, table.
#     Raises ValueError if the group has fewer than n entries.
#     """
#     group = data[group_key]          # O(1) dict lookup; KeyError if missing
#     rng = random.Random(seed)        # local RNG — does not disturb global state
#     indices = rng.sample(range(len(group)), n)
#     return [{f: group[i][f] for f in _FIELDS} for i in indices]


# # ---------------------------------------------------------------------------
# # Quick smoke-test when run directly
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import sys

#     grouped_file = sys.argv[1] if len(sys.argv) > 1 else "grouped_output.json"
#     target_group = sys.argv[2] if len(sys.argv) > 2 else None
#     seed_arg     = int(sys.argv[3]) if len(sys.argv) > 3 else 42

#     data = load_grouped_data(grouped_file)

#     if target_group is None:
#         target_group = next(iter(data))   # first available group
#         print(f"No group specified; using first group: {target_group!r}")

#     samples = fetch_samples(data, target_group, seed=seed_arg)
#     print(f"Sampled {len(samples)} entries from {target_group!r} (seed={seed_arg}):\n")
#     for idx, s in enumerate(samples, 1):
#         print(f"--- Entry {idx} ---")
#         print("pre_text :", s["pre_text"][:2], "...")
#         print("post_text:", s["post_text"][:2], "...")
#         print("table    :", s["table"][:2], "...")
#         print()
