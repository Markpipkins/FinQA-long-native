"""Postprocess interleaved_plus FinQA retrieval files.

This script does NOT rerun FAISS, BM25, embeddings, or retrieval.

It creates two derived retrieval-result sets from existing interleaved_plus files:

1. SORTED files
   - Same retrieved chunks as the input.
   - Chunks are sorted into readable document order.

2. GOLD_INJECTED_SORTED files
   - Missing qa.gold_inds evidence text is injected directly into the retrieved set.
   - Injection replaces the lowest-ranked non-gold retrieved chunk first.
   - After injection, chunks are sorted into readable document order.

Important experimental policy:
- GOLD_INJECTED files are oracle / hypothetical files.
- They are not valid for the final practical RAG comparison.
- They are intentionally labeled with GOLD_INJECTED in folder and file names.
- Model prompt builders should include only chunk text, not metadata.
"""

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------
# Text normalization and matching
# ---------------------------------------------------------------------


def normalize_text(text: Any) -> str:
    """Lowercase text and collapse whitespace for exact substring matching."""
    return " ".join(str(text).lower().split())


def normalized_gold_items(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return gold evidence items with normalized text, excluding empty strings."""
    gold_inds = example.get("qa", {}).get("gold_inds", {}) or {}
    items = []

    for gold_id, gold_text in gold_inds.items():
        norm_text = normalize_text(gold_text)
        if not norm_text:
            continue

        items.append({
            "gold_id": str(gold_id),
            "gold_text": str(gold_text),
            "norm_text": norm_text,
            "gold_type": infer_gold_type(str(gold_id)),
        })

    return items


def text_contains(norm_container: str, norm_needle: str) -> bool:
    """Return True if a normalized evidence string appears in normalized chunk text."""
    return bool(norm_needle) and norm_needle in norm_container


def gold_present_in_chunks(gold_item: Dict[str, str], chunks: List[Dict[str, Any]]) -> bool:
    """Check whether a full gold evidence text appears in any retrieved chunk."""
    for chunk in chunks:
        if text_contains(normalize_text(chunk.get("text", "")), gold_item["norm_text"]):
            return True
    return False


def chunk_contains_any_gold(chunk: Dict[str, Any], gold_items: List[Dict[str, str]]) -> bool:
    """Protect chunks that already contain any full gold evidence text."""
    if chunk.get("gold_injected"):
        return True

    norm_chunk_text = normalize_text(chunk.get("text", ""))
    return any(text_contains(norm_chunk_text, item["norm_text"]) for item in gold_items)


# ---------------------------------------------------------------------
# Chunk ID parsing and sorting
# ---------------------------------------------------------------------


LOCATION_ORDER = {
    "pre_text": 0,
    "table": 1,
    "post_text": 2,
    "gold_injected": 3,
    "unknown": 4,
}


def infer_gold_type(gold_id: str) -> str:
    """Infer whether a gold evidence item refers to text or table evidence."""
    if str(gold_id).startswith("table_"):
        return "table"
    if str(gold_id).startswith("text_"):
        return "text"
    return "gold_injected"


def parse_chunk_id(chunk_id: Any) -> Tuple[str, int, int]:
    """Parse IDs like text_121, table_2, or text_121__part0.

    Returns:
        (prefix, numeric_id, part_number)

    Unknown or unparsable IDs are pushed toward the end while preserving stability
    through original_retrieved_position in the final sort key.
    """
    chunk_id = str(chunk_id or "")
    match = re.match(r"^(text|table)_(\d+)(?:__part(\d+))?$", chunk_id)

    if not match:
        return "unknown", 10**9, 10**9

    prefix = match.group(1)
    number = int(match.group(2))
    part = int(match.group(3)) if match.group(3) is not None else 0

    return prefix, number, part


def sort_key(chunk: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Sort chunks by readable document order.

    The visible order should be:
        pre_text -> table -> post_text -> fallback/unknown

    Within a section, sort numerically by chunk ID and part number.
    The original retrieval position is used only as a stable tie breaker.
    """
    location = chunk.get("location") or "unknown"
    location_rank = LOCATION_ORDER.get(str(location), LOCATION_ORDER["unknown"])

    if "sort_numeric_override" in chunk:
        numeric_id = int(chunk.get("sort_numeric_override", 10**9))
        part_number = int(chunk.get("sort_part_override", 0))
    else:
        sort_source_id = chunk.get("sort_source_id", chunk.get("chunk_id"))
        _prefix, numeric_id, part_number = parse_chunk_id(sort_source_id)

    original_position = int(chunk.get("original_retrieved_position", 10**9))
    return location_rank, numeric_id, part_number, original_position


def add_original_positions(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add 1-indexed original retrieval position before sorting or injection."""
    positioned = []

    for index, chunk in enumerate(chunks, start=1):
        new_chunk = copy.deepcopy(chunk)
        new_chunk.setdefault("original_retrieved_position", index)
        positioned.append(new_chunk)

    return positioned


def sorted_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return chunks sorted into readable document order."""
    return sorted(chunks, key=sort_key)


# ---------------------------------------------------------------------
# Gold injection
# ---------------------------------------------------------------------


def choose_replacement_index(chunks: List[Dict[str, Any]], gold_items: List[Dict[str, str]]) -> Optional[int]:
    """Choose the lowest-ranked chunk that does not already contain gold evidence.

    Original ranking is represented by list position before sorting. The search
    starts at the bottom of the list: chunk 10, then 9, then 8, etc.
    """
    for index in range(len(chunks) - 1, -1, -1):
        if not chunk_contains_any_gold(chunks[index], gold_items):
            return index

    return None


def apply_gold_type_to_replacement(
    replacement_chunk: Dict[str, Any],
    gold_item: Dict[str, str],
) -> Dict[str, Any]:
    """Update chunk type/location to match the injected gold evidence type.

    If table gold replaces a text chunk, the result becomes table/table.
    If text gold replaces a table chunk, the result becomes text/post_text by
    default because the shifted expanded text IDs are not safe enough to locate
    exact pre_text vs post_text position.
    """
    updated = copy.deepcopy(replacement_chunk)
    gold_type = gold_item["gold_type"]
    old_location = replacement_chunk.get("location")

    if gold_type == "table":
        updated["chunk_type"] = "table"
        updated["location"] = "table"
        updated["sort_source_id"] = gold_item["gold_id"]
        updated["injected_location_assumption"] = "table_gold_id_used_for_table_order"
        return updated

    if gold_type == "text":
        updated["chunk_type"] = "text"

        if old_location in {"pre_text", "post_text"}:
            # Keep the replaced text chunk's broad document region.
            updated["location"] = old_location
            updated["sort_source_id"] = replacement_chunk.get("chunk_id")
            updated["injected_location_assumption"] = "kept_replaced_text_chunk_location"
        else:
            # Text IDs shifted during expansion, so do not pretend we know the exact location.
            # Put text fallback after the table in the post_text region.
            updated["location"] = "post_text"
            updated["sort_numeric_override"] = 1_000_000 + int(replacement_chunk.get("original_retrieved_position", 0))
            updated["sort_part_override"] = 0
            updated["injected_location_assumption"] = "post_text_default_for_text_gold_replacing_non_text_chunk"

        return updated

    updated["chunk_type"] = "gold_injected"
    updated["location"] = "gold_injected"
    updated["sort_numeric_override"] = 2_000_000 + int(replacement_chunk.get("original_retrieved_position", 0))
    updated["sort_part_override"] = 0
    updated["injected_location_assumption"] = "unknown_gold_id_type"
    return updated


def inject_missing_gold(chunks: List[Dict[str, Any]], gold_items: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Inject missing full gold evidence text by replacing low-ranked non-gold chunks."""
    working_chunks = copy.deepcopy(chunks)

    replacements = []
    already_present = []
    unresolved = []

    for gold_item in gold_items:
        if gold_present_in_chunks(gold_item, working_chunks):
            already_present.append(gold_item["gold_id"])
            continue

        replacement_index = choose_replacement_index(working_chunks, gold_items)

        if replacement_index is None:
            unresolved.append({
                "gold_id": gold_item["gold_id"],
                "reason": "no_non_gold_replacement_chunk_available",
            })
            continue

        old_chunk = working_chunks[replacement_index]
        new_chunk = apply_gold_type_to_replacement(old_chunk, gold_item)

        replaced_selected_by = old_chunk.get("selected_by", [])
        if isinstance(replaced_selected_by, str):
            replaced_selected_by = [replaced_selected_by]

        new_chunk.update({
            "text": gold_item["gold_text"],
            "selected_by": ["GOLD_INJECTED"],
            "gold_injected": True,
            "original_gold_id": gold_item["gold_id"],
            "gold_evidence_type": gold_item["gold_type"],
            "replaced_chunk_id": old_chunk.get("chunk_id"),
            "replaced_chunk_type": old_chunk.get("chunk_type"),
            "replaced_location": old_chunk.get("location"),
            "replaced_selected_by": replaced_selected_by,
            "replaced_faiss_rank": old_chunk.get("faiss_rank"),
            "replaced_faiss_score": old_chunk.get("faiss_score"),
            "replaced_bm25_rank": old_chunk.get("bm25_rank"),
            "replaced_bm25_score": old_chunk.get("bm25_score"),
            "replacement_original_retrieved_position": old_chunk.get("original_retrieved_position"),
            "model_visible_note": "Only the text field should be placed in prompts; this metadata is for audit only.",
        })

        working_chunks[replacement_index] = new_chunk

        replacements.append({
            "gold_id": gold_item["gold_id"],
            "gold_type": gold_item["gold_type"],
            "replaced_chunk_id": old_chunk.get("chunk_id"),
            "replaced_chunk_type": old_chunk.get("chunk_type"),
            "replaced_location": old_chunk.get("location"),
            "replacement_original_retrieved_position": old_chunk.get("original_retrieved_position"),
            "new_chunk_type": new_chunk.get("chunk_type"),
            "new_location": new_chunk.get("location"),
        })

    still_missing = []
    for gold_item in gold_items:
        if not gold_present_in_chunks(gold_item, working_chunks):
            still_missing.append(gold_item["gold_id"])

    debug = {
        "gold_items_total": len(gold_items),
        "gold_items_already_present": len(already_present),
        "gold_items_injected": len(replacements),
        "gold_items_unresolved": len(still_missing),
        "already_present_gold_ids": already_present,
        "still_missing_gold_ids": still_missing,
        "unresolved_details": unresolved,
        "replacements": replacements,
    }

    return working_chunks, debug


# ---------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def process_split(
    split: str,
    dataset_file: Path,
    retrieved_file: Path,
    sorted_output_file: Path,
    gold_output_file: Path,
) -> Dict[str, Any]:
    """Create sorted-only and GOLD_INJECTED_SORTED files for one split."""
    dataset = load_json(dataset_file)
    retrieved_records = load_json(retrieved_file)
    dataset_by_id = {example["id"]: example for example in dataset}

    sorted_records = []
    gold_records = []

    summary = {
        "split": split,
        "dataset_file": str(dataset_file),
        "retrieved_file": str(retrieved_file),
        "sorted_output_file": str(sorted_output_file),
        "gold_injected_output_file": str(gold_output_file),
        "records_processed": 0,
        "records_missing_dataset_example": 0,
        "records_without_gold_items": 0,
        "records_requiring_injection": 0,
        "total_gold_items": 0,
        "gold_items_already_present": 0,
        "gold_items_injected": 0,
        "gold_items_still_missing_after_injection": 0,
        "chunks_replaced": 0,
        "max_injections_in_one_example": 0,
        "missing_dataset_ids_sample": [],
        "examples_with_unresolved_gold_sample": [],
    }

    for record in retrieved_records:
        example_id = record.get("id")
        example = dataset_by_id.get(example_id)

        base_chunks = add_original_positions(record.get("retrieved_chunks", []))

        # Sorted-only record. This version does not use gold information.
        sorted_record = copy.deepcopy(record)
        sorted_record["retrieved_chunks"] = sorted_chunks(base_chunks)
        sorted_record.setdefault("postprocess_debug", {})
        sorted_record["postprocess_debug"].update({
            "sorted": True,
            "gold_injected": False,
            "source_retrieval_order_preserved_in_original_retrieved_position": True,
        })
        sorted_records.append(sorted_record)

        # GOLD_INJECTED record.
        gold_record = copy.deepcopy(record)

        if example is None:
            summary["records_missing_dataset_example"] += 1
            if len(summary["missing_dataset_ids_sample"]) < 25:
                summary["missing_dataset_ids_sample"].append(example_id)

            gold_record["retrieved_chunks"] = sorted_chunks(base_chunks)
            gold_record.setdefault("postprocess_debug", {})
            gold_record["postprocess_debug"].update({
                "sorted": True,
                "gold_injected": False,
                "gold_injection_skipped_reason": "matching_dataset_example_not_found",
            })
            gold_records.append(gold_record)
            continue

        gold_items = normalized_gold_items(example)
        summary["total_gold_items"] += len(gold_items)

        if not gold_items:
            summary["records_without_gold_items"] += 1

        injected_chunks, injection_debug = inject_missing_gold(base_chunks, gold_items)
        sorted_injected_chunks = sorted_chunks(injected_chunks)

        if injection_debug["gold_items_injected"] > 0:
            summary["records_requiring_injection"] += 1

        summary["gold_items_already_present"] += injection_debug["gold_items_already_present"]
        summary["gold_items_injected"] += injection_debug["gold_items_injected"]
        summary["gold_items_still_missing_after_injection"] += injection_debug["gold_items_unresolved"]
        summary["chunks_replaced"] += injection_debug["gold_items_injected"]
        summary["max_injections_in_one_example"] = max(
            summary["max_injections_in_one_example"],
            injection_debug["gold_items_injected"],
        )

        if injection_debug["gold_items_unresolved"] > 0 and len(summary["examples_with_unresolved_gold_sample"]) < 25:
            summary["examples_with_unresolved_gold_sample"].append({
                "id": example_id,
                "still_missing_gold_ids": injection_debug["still_missing_gold_ids"],
            })

        gold_record["retrieved_chunks"] = sorted_injected_chunks
        gold_record.setdefault("postprocess_debug", {})
        gold_record["postprocess_debug"].update({
            "sorted": True,
            "gold_injected": True,
            "oracle_gold_injected": True,
            "valid_for_practical_final_comparison": False,
            "intended_use": "hypothetical retrieval with 100 percent gold-context availability plus distractor chunks",
            "source_retrieval_order_preserved_in_original_retrieved_position": True,
            **injection_debug,
        })
        gold_records.append(gold_record)

        summary["records_processed"] += 1

    save_json(sorted_output_file, sorted_records)
    save_json(gold_output_file, gold_records)

    print(f"\nFinished split: {split}")
    print(f"Saved SORTED file: {sorted_output_file}")
    print(f"Saved GOLD_INJECTED_SORTED file: {gold_output_file}")
    print(f"Records processed: {summary['records_processed']}")
    print(f"Total gold items: {summary['total_gold_items']}")
    print(f"Gold already present: {summary['gold_items_already_present']}")
    print(f"Gold injected: {summary['gold_items_injected']}")
    print(f"Gold still missing after injection: {summary['gold_items_still_missing_after_injection']}")

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SORTED and GOLD_INJECTED_SORTED versions of interleaved_plus retrieval files."
    )

    parser.add_argument("--dataset_dir", default="long_dataset")
    parser.add_argument("--retrieved_dir", default="rag_retrieved_interleaved_plus")
    parser.add_argument("--sorted_output_dir", default="rag_retrieved_interleaved_plus_SORTED")
    parser.add_argument("--gold_output_dir", default="rag_retrieved_interleaved_plus_GOLD_INJECTED_SORTED")

    parser.add_argument("--input_suffix", default="_interleaved_plus_retrieved")
    parser.add_argument("--sorted_suffix", default="_interleaved_plus_SORTED_retrieved")
    parser.add_argument("--gold_suffix", default="_interleaved_plus_GOLD_INJECTED_SORTED_retrieved")

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["long_train", "long_dev", "long_test"],
        help="Splits to postprocess. Default intentionally excludes private_test because it usually lacks gold labels.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    retrieved_dir = Path(args.retrieved_dir)
    sorted_output_dir = Path(args.sorted_output_dir)
    gold_output_dir = Path(args.gold_output_dir)

    config = {
        "script": "postprocess_interleaved_plus_retrieval.py",
        "dataset_dir": args.dataset_dir,
        "retrieved_dir": args.retrieved_dir,
        "sorted_output_dir": args.sorted_output_dir,
        "gold_output_dir": args.gold_output_dir,
        "input_suffix": args.input_suffix,
        "sorted_suffix": args.sorted_suffix,
        "gold_suffix": args.gold_suffix,
        "splits": args.splits,
        "reruns_retrieval": False,
        "rebuilds_candidate_chunks": False,
        "sorted_files_policy": "Same chunks as interleaved_plus, sorted by location and numeric chunk_id.",
        "gold_injected_policy": "Missing qa.gold_inds evidence text replaces the lowest-ranked non-gold retrieved chunk before sorting.",
        "gold_matching_policy": "Full normalized qa.gold_inds text must appear inside retrieved chunk text to count as already present.",
        "replacement_policy": "Start from the bottom of original retrieval rank order and skip chunks that already contain any gold evidence.",
        "type_correction_policy": "Injected chunks take chunk_type/location from the gold evidence type rather than blindly keeping replaced chunk type.",
        "oracle_gold_injected": True,
        "valid_for_practical_final_comparison": False,
        "model_visible_fields_policy": "Prompt builders should pass only retrieved_chunks[*].text to models; metadata is for audit only.",
    }

    save_json(sorted_output_dir / "retrieval_postprocess_config.json", config)
    save_json(gold_output_dir / "retrieval_postprocess_config_GOLD_INJECTED.json", config)

    all_summaries = {}

    for split in args.splits:
        dataset_file = dataset_dir / f"{split}.json"
        retrieved_file = retrieved_dir / f"{split}{args.input_suffix}.json"
        sorted_output_file = sorted_output_dir / f"{split}{args.sorted_suffix}.json"
        gold_output_file = gold_output_dir / f"{split}{args.gold_suffix}.json"

        if not dataset_file.exists():
            print(f"Skipping missing dataset file: {dataset_file}")
            continue

        if not retrieved_file.exists():
            print(f"Skipping missing retrieved file: {retrieved_file}")
            continue

        summary = process_split(
            split=split,
            dataset_file=dataset_file,
            retrieved_file=retrieved_file,
            sorted_output_file=sorted_output_file,
            gold_output_file=gold_output_file,
        )

        all_summaries[split] = summary

    save_json(sorted_output_dir / "postprocess_summary.json", all_summaries)
    save_json(gold_output_dir / "postprocess_summary_GOLD_INJECTED.json", all_summaries)

    print(f"\nSaved SORTED summary: {sorted_output_dir / 'postprocess_summary.json'}")
    print(f"Saved GOLD_INJECTED summary: {gold_output_dir / 'postprocess_summary_GOLD_INJECTED.json'}")


if __name__ == "__main__":
    main()
