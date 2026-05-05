import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------
# Simple text matching helpers
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for stable substring matching."""
    return " ".join(str(text).lower().split())


def gold_text_is_present(gold_text: str, retrieved_chunks):
    """
    Check whether a full normalized gold evidence string appears inside
    any retrieved chunk text.

    Returns:
        (found: bool, matches: list[dict])
    """
    norm_gold = normalize_text(gold_text)

    if not norm_gold:
        return False, []

    matches = []

    for position, chunk in enumerate(retrieved_chunks, start=1):
        norm_chunk = normalize_text(chunk.get("text", ""))

        if norm_gold in norm_chunk:
            matches.append({
                "retrieved_position": position,
                "chunk_id": chunk.get("chunk_id"),
                "chunk_type": chunk.get("chunk_type"),
                "location": chunk.get("location"),
                "selected_by": chunk.get("selected_by"),
                "gold_injected": chunk.get("gold_injected", False),
                "original_retrieved_position": chunk.get("original_retrieved_position"),
            })

    return len(matches) > 0, matches


# ---------------------------------------------------------------------
# Split checking
# ---------------------------------------------------------------------

def check_split(dataset_file: Path, retrieved_file: Path, output_file: Path):
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(retrieved_file, "r", encoding="utf-8") as f:
        retrieved_records = json.load(f)

    retrieved_by_id = {record["id"]: record for record in retrieved_records}

    examples_report = []
    missing_examples = []

    examples_scored = 0
    total_gold_items = 0
    matched_gold_items = 0
    missing_gold_items = 0

    text_gold_items = 0
    text_gold_matched = 0
    table_gold_items = 0
    table_gold_matched = 0

    missing_retrieved_records = []

    for example in dataset:
        example_id = example.get("id")
        gold_inds = example.get("qa", {}).get("gold_inds", {})

        if not gold_inds:
            continue

        retrieved_record = retrieved_by_id.get(example_id)

        if retrieved_record is None:
            missing_retrieved_records.append(example_id)
            continue

        retrieved_chunks = retrieved_record.get("retrieved_chunks", [])

        gold_items_report = []
        example_gold_total = 0
        example_gold_matched = 0
        example_missing_gold_ids = []

        for gold_id, gold_text in gold_inds.items():
            found, matches = gold_text_is_present(gold_text, retrieved_chunks)

            example_gold_total += 1
            total_gold_items += 1

            if gold_id.startswith("text_"):
                text_gold_items += 1
            elif gold_id.startswith("table_"):
                table_gold_items += 1

            if found:
                example_gold_matched += 1
                matched_gold_items += 1

                if gold_id.startswith("text_"):
                    text_gold_matched += 1
                elif gold_id.startswith("table_"):
                    table_gold_matched += 1
            else:
                missing_gold_items += 1
                example_missing_gold_ids.append(gold_id)

            gold_items_report.append({
                "gold_id": gold_id,
                "found": found,
                "matches": matches,
                "gold_text": gold_text,
            })

        examples_scored += 1

        example_overlap = (
            example_gold_matched / example_gold_total
            if example_gold_total else None
        )

        example_report = {
            "id": example_id,
            "gold_items_total": example_gold_total,
            "gold_items_matched": example_gold_matched,
            "gold_items_missing": example_gold_total - example_gold_matched,
            "gold_overlap": example_overlap,
            "missing_gold_ids": example_missing_gold_ids,
            "retrieved_chunk_count": len(retrieved_chunks),
            "gold_items": gold_items_report,
        }

        examples_report.append(example_report)

        if example_missing_gold_ids:
            missing_examples.append({
                "id": example_id,
                "gold_items_total": example_gold_total,
                "gold_items_matched": example_gold_matched,
                "gold_items_missing": example_gold_total - example_gold_matched,
                "gold_overlap": example_overlap,
                "missing_gold_ids": example_missing_gold_ids,
            })

    summary = {
        "dataset_file": str(dataset_file),
        "retrieved_file": str(retrieved_file),
        "output_file": str(output_file),
        "matching_policy": "strict full normalized qa.gold_inds text must appear inside retrieved chunk text",
        "examples_scored": examples_scored,
        "examples_with_missing_gold": len(missing_examples),
        "missing_retrieved_records_count": len(missing_retrieved_records),
        "missing_retrieved_records": missing_retrieved_records[:50],
        "total_gold_items": total_gold_items,
        "matched_gold_items": matched_gold_items,
        "missing_gold_items": missing_gold_items,
        "average_gold_overlap": matched_gold_items / total_gold_items if total_gold_items else None,
        "text_gold_items": text_gold_items,
        "text_gold_matched": text_gold_matched,
        "text_gold_overlap": text_gold_matched / text_gold_items if text_gold_items else None,
        "table_gold_items": table_gold_items,
        "table_gold_matched": table_gold_matched,
        "table_gold_overlap": table_gold_matched / table_gold_items if table_gold_items else None,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "missing_examples": missing_examples,
            "examples": examples_report,
        }, f, indent=2)

    print(f"\nFinished: {dataset_file.name}")
    print(f"Checked: {retrieved_file}")
    print(f"Saved report: {output_file}")
    print(f"Examples scored: {examples_scored}")
    print(f"Total gold items: {total_gold_items}")
    print(f"Matched gold items: {matched_gold_items}")
    print(f"Missing gold items: {missing_gold_items}")
    print(f"Average gold overlap: {summary['average_gold_overlap']}")
    print(f"Examples with missing gold: {len(missing_examples)}")

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check strict qa.gold_inds overlap for GOLD_INJECTED_SORTED retrieval files."
    )

    parser.add_argument("--dataset_dir", default="long_dataset")
    parser.add_argument(
        "--retrieved_dir",
        default="rag_retrieved_interleaved_plus_GOLD_INJECTED_SORTED"
    )
    parser.add_argument(
        "--output_dir",
        default="rag_retrieved_interleaved_plus_GOLD_INJECTED_SORTED_overlap_check"
    )
    parser.add_argument(
        "--retrieved_suffix",
        default="_interleaved_plus_GOLD_INJECTED_SORTED_retrieved"
    )
    parser.add_argument(
        "--report_suffix",
        default="_GOLD_INJECTED_STRICT_gold_overlap_check"
    )
    parser.add_argument(
        "--summary_filename",
        default="summary_GOLD_INJECTED_STRICT_gold_overlap_check.json"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["long_train", "long_dev", "long_test"],
        help="Splits to check. Private test is skipped by default because it may not have gold labels."
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    retrieved_dir = Path(args.retrieved_dir)
    output_dir = Path(args.output_dir)

    all_summaries = {}

    for split in args.splits:
        dataset_file = dataset_dir / f"{split}.json"
        retrieved_file = retrieved_dir / f"{split}{args.retrieved_suffix}.json"
        output_file = output_dir / f"{split}{args.report_suffix}.json"

        if not dataset_file.exists():
            print(f"Skipping missing dataset file: {dataset_file}")
            continue

        if not retrieved_file.exists():
            print(f"Skipping missing retrieved file: {retrieved_file}")
            continue

        summary = check_split(
            dataset_file=dataset_file,
            retrieved_file=retrieved_file,
            output_file=output_file,
        )

        all_summaries[split] = summary

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / args.summary_filename
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved overall summary: {summary_file}")


if __name__ == "__main__":
    main()
