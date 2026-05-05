"""
validate_dataset.py
-------------------
Validates enriched (long) dataset JSON files against their original counterparts.

For each of the 4 file pairs, the following checks are run:
  1. Both JSON files load without errors
  2. Entry counts match
  3. Every original id exists in the enriched file (and vice versa)
  4. For every entry (matched by id), these fields are byte-for-byte identical:
       table, id, qa.question, qa.program, qa.gold_inds, qa.exe_ans, qa.program_re

Failure reporting:
  - 0 failures       -> single pass confirmation per check
  - 1-25 failures    -> print each failing entry's id and the mismatched fields
  - 26+ failures     -> print only the total failure count

Usage:
    python validate_dataset.py
"""

import json

# ---------------------------------------------------------------------------
# Hardcoded file pairs: (enriched, original)
# ---------------------------------------------------------------------------

FILE_PAIRS = [
    ("long_dataset/long_dev.json",          "dataset/dev.json"),
    ("long_dataset/long_private_test.json", "dataset/private_test.json"),
    ("long_dataset/long_test.json",         "dataset/test.json"),
    ("long_dataset/long_train.json",        "dataset/train.json"),
]

# Fields inside qa that must be identical
QA_FIELDS = ("question", "program", "gold_inds", "exe_ans", "program_re")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> tuple[list | None, str | None]:
    """
    Attempt to load a JSON file.
    Returns (data, None) on success, (None, error_message) on failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def report_failures(failures: list[dict]) -> None:
    """
    Print failure details according to the count threshold.
    Each item in failures is:
      { "id": str, "mismatches": [ {"field": str, "original": ..., "enriched": ...} ] }
    """
    count = len(failures)
    if count == 0:
        return

    if count > 25:
        print(f"    FAIL — {count} entries failed (too many to list individually).")
        return

    print(f"    FAIL — {count} entr{'y' if count == 1 else 'ies'} failed:")
    for failure in failures:
        print(f"\n      Entry id: {failure['id']}")
        for m in failure["mismatches"]:
            print(f"        Field     : {m['field']}")
            print(f"        Original  : {m['original']}")
            print(f"        Enriched  : {m['enriched']}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_counts(original: list, enriched: list) -> bool:
    orig_count = len(original)
    enr_count  = len(enriched)
    if orig_count == enr_count:
        print(f"  [PASS] Entry counts match: {orig_count}")
        return True
    else:
        print(f"  [FAIL] Entry count mismatch — original: {orig_count}, enriched: {enr_count}")
        return False


def check_ids(original: list, enriched: list) -> bool:
    orig_ids = {e["id"] for e in original}
    enr_ids  = {e["id"] for e in enriched}

    missing  = orig_ids - enr_ids   # in original but not in enriched
    extra    = enr_ids  - orig_ids  # in enriched but not in original

    if not missing and not extra:
        print(f"  [PASS] All {len(orig_ids)} ids match exactly.")
        return True

    if missing:
        count = len(missing)
        if count > 25:
            print(f"  [FAIL] {count} ids from original are missing in enriched (too many to list).")
        else:
            print(f"  [FAIL] {count} id(s) from original missing in enriched:")
            for i in sorted(missing):
                print(f"           {i}")

    if extra:
        count = len(extra)
        if count > 25:
            print(f"  [FAIL] {count} ids in enriched have no match in original (too many to list).")
        else:
            print(f"  [FAIL] {count} id(s) in enriched have no match in original:")
            for i in sorted(extra):
                print(f"           {i}")

    return False


def check_fields(original: list, enriched: list) -> bool:
    """
    Match every entry by id and verify that table and qa subfields are identical.
    The entire entry is counted as one failure if any field mismatches.
    """
    # Index enriched entries by id for O(1) lookup
    enriched_by_id = {e["id"]: e for e in enriched}

    failures = []

    for orig_entry in original:
        entry_id = orig_entry["id"]
        enr_entry = enriched_by_id.get(entry_id)

        if enr_entry is None:
            # Already caught by check_ids; skip here to avoid duplicate noise
            continue

        mismatches = []

        # Check table
        if orig_entry.get("table") != enr_entry.get("table"):
            mismatches.append({
                "field":    "table",
                "original": orig_entry.get("table"),
                "enriched": enr_entry.get("table"),
            })

        # Check qa subfields
        orig_qa = orig_entry.get("qa", {})
        enr_qa  = enr_entry.get("qa", {})
        for field in QA_FIELDS:
            if orig_qa.get(field) != enr_qa.get(field):
                mismatches.append({
                    "field":    f"qa.{field}",
                    "original": orig_qa.get(field),
                    "enriched": enr_qa.get(field),
                })

        if mismatches:
            failures.append({"id": entry_id, "mismatches": mismatches})

    if not failures:
        print(f"  [PASS] All field-level checks passed across {len(original)} entries.")
        return True

    report_failures(failures)
    return False


# ---------------------------------------------------------------------------
# Per-pair validation
# ---------------------------------------------------------------------------

def validate_pair(enriched_path: str, original_path: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  Enriched : {enriched_path}")
    print(f"  Original : {original_path}")
    print(f"{'=' * 70}")

    # Check 1: JSON loads
    print("\n  [CHECK 1] Loading JSON files...")
    original, orig_err = load_json(original_path)
    enriched, enr_err  = load_json(enriched_path)

    if orig_err:
        print(f"  [FAIL] Could not load original file — {orig_err}")
        return False
    if enr_err:
        print(f"  [FAIL] Could not load enriched file — {enr_err}")
        return False
    print(f"  [PASS] Both files loaded successfully.")

    pair_passed = True

    # Check 2: Entry counts
    print("\n  [CHECK 2] Comparing entry counts...")
    if not check_counts(original, enriched):
        pair_passed = False

    # Check 3: ID sets
    print("\n  [CHECK 3] Comparing id sets...")
    if not check_ids(original, enriched):
        pair_passed = False

    # Check 4: Field-level identity
    print("\n  [CHECK 4] Validating field-level integrity...")
    if not check_fields(original, enriched):
        pair_passed = False

    return pair_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\nDataset Validation Report")
    print("=" * 70)

    results = {}
    for enriched_path, original_path in FILE_PAIRS:
        passed = validate_pair(enriched_path, original_path)
        results[(enriched_path, original_path)] = passed

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    all_passed = True
    for (enr_path, orig_path), passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {enr_path}")

    print()
    if all_passed:
        print("All file pairs passed validation.")
    else:
        print("One or more file pairs failed validation. See details above.")
    print()


if __name__ == "__main__":
    main()
