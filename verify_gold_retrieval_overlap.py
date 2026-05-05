"""Verify retrieved chunks against actual FinQA gold evidence text.

This interleaved-version verifier expects retrieved files named like:
    long_train_interleaved_retrieved.json

Important: the overlap score is based on the text stored in qa.gold_inds values,
not on old text_X/table_X IDs. IDs are kept only for reporting and type totals.
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for matching.

    This keeps matching simple:
    - lowercase
    - collapse repeated whitespace
    - strip leading/trailing spaces

    It does not remove punctuation because punctuation and numbers may matter
    in financial text.
    """
    return " ".join(str(text).lower().split())


def split_front_back(text: str):
    """
    Split a normalized gold text string into front and back halves.

    The split tries to avoid cutting through the middle of a word by moving
    to the nearest space around the midpoint.
    """
    text = normalize_text(text)

    if len(text) <= 1:
        return text, ""

    midpoint = len(text) // 2

    left_space = text.rfind(" ", 0, midpoint)
    right_space = text.find(" ", midpoint)

    if left_space == -1 and right_space == -1:
        split_at = midpoint
    elif left_space == -1:
        split_at = right_space
    elif right_space == -1:
        split_at = left_space
    else:
        split_at = left_space if (midpoint - left_space) <= (right_space - midpoint) else right_space

    front = text[:split_at].strip()
    back = text[split_at:].strip()

    return front, back


# ---------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------

def find_substring_in_chunks(search_text, normalized_chunks):
    """
    Search for one normalized string inside all retrieved chunks.

    Returns every retrieved chunk that contains the string.
    """
    search_text = normalize_text(search_text)

    if not search_text:
        return []

    matches = []

    for chunk in normalized_chunks:
        if search_text in chunk["norm_text"]:
            matches.append({
                "chunk_id": chunk["chunk_id"],
                "chunk_type": chunk.get("chunk_type"),
                "location": chunk.get("location")
            })

    return matches


def find_shortened_front(front_text, normalized_chunks, trim_chars):
    """
    Search the front half after repeatedly removing characters from the right.

    Example:
    original front: "the total revenue was $ 500 million"
    first shortened: remove 8 rightmost characters
    """
    current = normalize_text(front_text)
    trim_steps = 0

    while len(current) > trim_chars:
        current = current[:-trim_chars].strip()
        trim_steps += 1

        if not current:
            break

        matches = find_substring_in_chunks(current, normalized_chunks)

        if matches:
            return {
                "found": True,
                "matched_text": current,
                "trim_steps": trim_steps,
                "matches": matches
            }

    return {
        "found": False,
        "matched_text": None,
        "trim_steps": trim_steps,
        "matches": []
    }


def find_shortened_back(back_text, normalized_chunks, trim_chars):
    """
    Search the back half after repeatedly removing characters from the left.

    Example:
    original back: "cash flow hedges denominated in euros"
    first shortened: remove 8 leftmost characters
    """
    current = normalize_text(back_text)
    trim_steps = 0

    while len(current) > trim_chars:
        current = current[trim_chars:].strip()
        trim_steps += 1

        if not current:
            break

        matches = find_substring_in_chunks(current, normalized_chunks)

        if matches:
            return {
                "found": True,
                "matched_text": current,
                "trim_steps": trim_steps,
                "matches": matches
            }

    return {
        "found": False,
        "matched_text": None,
        "trim_steps": trim_steps,
        "matches": []
    }


def score_gold_item(gold_id, gold_text, normalized_chunks, trim_chars):
    """
    Score one gold evidence item against the retrieved chunks.

    Scoring policy:
    - full gold text found: 1.0
    - front half found: +0.5
    - back half found: +0.5
    - shortened front found: +0.35
    - shortened back found: +0.35
    - cap total score at 1.0
    """
    norm_gold = normalize_text(gold_text)

    # First try full-string match.
    full_matches = find_substring_in_chunks(norm_gold, normalized_chunks)

    if full_matches:
        return {
            "gold_id": gold_id,
            "gold_text": gold_text,
            "score": 1.0,
            "match_type": "full",
            "full_match": {
                "found": True,
                "matches": full_matches
            },
            "front_match": None,
            "back_match": None
        }

    # If full match fails, try front/back partial matching.
    front, back = split_front_back(norm_gold)

    score = 0.0

    front_result = {
        "found": False,
        "match_type": None,
        "matched_text": None,
        "score_added": 0.0,
        "matches": [],
        "trim_steps": 0
    }

    back_result = {
        "found": False,
        "match_type": None,
        "matched_text": None,
        "score_added": 0.0,
        "matches": [],
        "trim_steps": 0
    }

    # Front half exact search.
    front_matches = find_substring_in_chunks(front, normalized_chunks)

    if front_matches:
        score += 0.5
        front_result = {
            "found": True,
            "match_type": "front_half",
            "matched_text": front,
            "score_added": 0.5,
            "matches": front_matches,
            "trim_steps": 0
        }
    else:
        # Front shortened search.
        shortened_front = find_shortened_front(front, normalized_chunks, trim_chars)

        if shortened_front["found"]:
            score += 0.35
            front_result = {
                "found": True,
                "match_type": "front_shortened",
                "matched_text": shortened_front["matched_text"],
                "score_added": 0.35,
                "matches": shortened_front["matches"],
                "trim_steps": shortened_front["trim_steps"]
            }

    # Back half exact search.
    back_matches = find_substring_in_chunks(back, normalized_chunks)

    if back_matches:
        score += 0.5
        back_result = {
            "found": True,
            "match_type": "back_half",
            "matched_text": back,
            "score_added": 0.5,
            "matches": back_matches,
            "trim_steps": 0
        }
    else:
        # Back shortened search.
        shortened_back = find_shortened_back(back, normalized_chunks, trim_chars)

        if shortened_back["found"]:
            score += 0.35
            back_result = {
                "found": True,
                "match_type": "back_shortened",
                "matched_text": shortened_back["matched_text"],
                "score_added": 0.35,
                "matches": shortened_back["matches"],
                "trim_steps": shortened_back["trim_steps"]
            }

    score = min(score, 1.0)

    if score == 0.0:
        match_type = "none"
    elif score >= 1.0:
        match_type = "partial_full_credit"
    else:
        match_type = "partial"

    return {
        "gold_id": gold_id,
        "gold_text": gold_text,
        "score": score,
        "match_type": match_type,
        "full_match": {
            "found": False,
            "matches": []
        },
        "front_match": front_result,
        "back_match": back_result
    }


# ---------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------

def normalize_retrieved_chunks(retrieved_record):
    """
    Prepare retrieved chunks for matching.

    The retrieved files created by the hybrid retriever contain:
    retrieved_chunks[*].text
    """
    normalized_chunks = []

    for chunk in retrieved_record.get("retrieved_chunks", []):
        normalized_chunks.append({
            "chunk_id": chunk.get("chunk_id"),
            "chunk_type": chunk.get("chunk_type"),
            "location": chunk.get("location"),
            "text": chunk.get("text", ""),
            "norm_text": normalize_text(chunk.get("text", ""))
        })

    return normalized_chunks


def process_split(dataset_file, retrieved_file, output_file, trim_chars):
    """
    Process one split and save a detailed gold-overlap report.

    This does not modify the dataset.
    This does not modify the retrieved file.
    This does not rerun retrieval.
    """
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(retrieved_file, "r", encoding="utf-8") as f:
        retrieved = json.load(f)

    retrieved_by_id = {record["id"]: record for record in retrieved}

    report_records = []

    total_examples = 0
    total_gold_items = 0
    total_gold_score = 0.0

    full_item_matches = 0
    partial_item_matches = 0
    zero_item_matches = 0

    text_gold_items = 0
    text_gold_score = 0.0

    table_gold_items = 0
    table_gold_score = 0.0

    missing_retrieved_ids = []

    for example in dataset:
        example_id = example["id"]
        gold_inds = example.get("qa", {}).get("gold_inds", {})

        # Skip examples without gold evidence.
        if not gold_inds:
            continue

        retrieved_record = retrieved_by_id.get(example_id)

        if retrieved_record is None:
            missing_retrieved_ids.append(example_id)
            continue

        normalized_chunks = normalize_retrieved_chunks(retrieved_record)

        gold_items_report = []
        example_gold_score = 0.0

        for gold_id, gold_text in gold_inds.items():
            item_report = score_gold_item(
                gold_id=gold_id,
                gold_text=gold_text,
                normalized_chunks=normalized_chunks,
                trim_chars=trim_chars
            )

            gold_items_report.append(item_report)

            item_score = item_report["score"]
            example_gold_score += item_score

            total_gold_items += 1
            total_gold_score += item_score

            # Use the gold ID only to separate text/table summary totals.
            # Matching correctness above is based on gold_text, not the old text_X/table_X position.
            if gold_id.startswith("text_"):
                text_gold_items += 1
                text_gold_score += item_score

            if gold_id.startswith("table_"):
                table_gold_items += 1
                table_gold_score += item_score

            if item_report["match_type"] == "full":
                full_item_matches += 1
            elif item_report["score"] == 0.0:
                zero_item_matches += 1
            else:
                partial_item_matches += 1

        gold_count = len(gold_inds)
        example_overlap = example_gold_score / gold_count if gold_count else None

        report_records.append({
            "id": example_id,
            "gold_count": gold_count,
            "gold_score": example_gold_score,
            "gold_overlap": example_overlap,
            "gold_items": gold_items_report,
            "retrieved_chunk_count": len(normalized_chunks),
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in normalized_chunks]
        })

        total_examples += 1

    split_summary = {
        "dataset_file": str(dataset_file),
        "retrieved_file": str(retrieved_file),
        "output_file": str(output_file),
        "matching_basis": "actual qa.gold_inds evidence text values",
        "id_usage_note": "gold_id is used only for reporting and text/table totals, not for matching correctness",
        "examples_scored": total_examples,
        "missing_retrieved_ids_count": len(missing_retrieved_ids),
        "missing_retrieved_ids": missing_retrieved_ids[:50],
        "total_gold_items": total_gold_items,
        "total_gold_score": total_gold_score,
        "average_gold_overlap": total_gold_score / total_gold_items if total_gold_items else None,
        "full_item_matches": full_item_matches,
        "partial_item_matches": partial_item_matches,
        "zero_item_matches": zero_item_matches,
        "text_gold_items": text_gold_items,
        "text_gold_score": text_gold_score,
        "text_gold_overlap": text_gold_score / text_gold_items if text_gold_items else None,
        "table_gold_items": table_gold_items,
        "table_gold_score": table_gold_score,
        "table_gold_overlap": table_gold_score / table_gold_items if table_gold_items else None
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": split_summary,
            "examples": report_records
        }, f, indent=2)

    print(f"\nFinished: {dataset_file.name}")
    print(f"Saved report: {output_file}")
    print(f"Examples scored: {total_examples}")
    print(f"Total gold items: {total_gold_items}")
    print(f"Average gold overlap: {split_summary['average_gold_overlap']}")
    print(f"Text gold overlap: {split_summary['text_gold_overlap']}")
    print(f"Table gold overlap: {split_summary['table_gold_overlap']}")
    print(f"Full matches: {full_item_matches}")
    print(f"Partial matches: {partial_item_matches}")
    print(f"Zero matches: {zero_item_matches}")

    return split_summary


def main():
    parser = argparse.ArgumentParser(
        description="Verify interleaved retrieved chunks against actual FinQA gold evidence text after long-context expansion."
    )

    parser.add_argument("--dataset_dir", default="long_dataset")
    parser.add_argument("--retrieved_dir", default="rag_retrieved_interleaved")
    parser.add_argument("--output_dir", default="rag_retrieved_interleaved_validation")

    parser.add_argument(
        "--retrieved_suffix",
        default="_interleaved_retrieved",
        help="Suffix used by the retrieved files, excluding .json."
    )

    parser.add_argument(
        "--report_suffix",
        default="_interleaved_gold_overlap_report",
        help="Suffix used by the validation report files, excluding .json."
    )

    parser.add_argument(
        "--summary_filename",
        default="summary_interleaved.json",
        help="Name of the overall validation summary file."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["long_train", "long_dev", "long_test"],
        help="Splits to evaluate. Private test is usually skipped because it may not have gold labels."
    )

    parser.add_argument(
        "--trim_chars",
        type=int,
        default=8,
        help="Number of characters removed from inside-facing substring edges during shortened matching."
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

        summary = process_split(
            dataset_file=dataset_file,
            retrieved_file=retrieved_file,
            output_file=output_file,
            trim_chars=args.trim_chars
        )

        all_summaries[split] = summary

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / args.summary_filename, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved overall summary: {output_dir / args.summary_filename}")


if __name__ == "__main__":
    main()