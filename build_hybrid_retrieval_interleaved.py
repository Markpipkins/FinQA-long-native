import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def remove_extra_spaces(text: str) -> str:
    return " ".join(str(text).split())


def ensure_list(value):
    """FinQA usually stores pre_text/post_text as lists, but this keeps the script safe."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return [str(value)]


def table_row_to_text(header, row) -> str:
    """Convert one FinQA table row into readable text, similar to FinQA's utility function."""
    if not header or not row:
        return ""

    # Fallback if row/header lengths are unusual.
    if len(header) < 2 or len(row) < 2:
        return remove_extra_spaces(" ".join(map(str, row)))

    result = ""

    if header[0]:
        result += str(header[0]) + " "

    row_name = str(row[0])

    for head, cell in zip(header[1:], row[1:]):
        result += f"the {row_name} of {head} is {cell} ; "

    return remove_extra_spaces(result)


def split_long_text(text: str, max_words: int, overlap_words: int):
    """
    Split only if a chunk is very long.
    Most FinQA rows/sentences will stay unchanged.
    """
    words = remove_extra_spaces(text).split()

    if max_words <= 0 or len(words) <= max_words:
        return [remove_extra_spaces(text)]

    chunks = []
    step = max(1, max_words - overlap_words)

    for start in range(0, len(words), step):
        part = words[start:start + max_words]
        if part:
            chunks.append(" ".join(part))

        if start + max_words >= len(words):
            break

    return chunks


def add_text_chunks(chunks, texts, start_index, location, max_words, overlap_words):
    """Add pre_text or post_text chunks while preserving FinQA-style text indices."""
    for local_i, text in enumerate(texts):
        global_i = start_index + local_i
        parts = split_long_text(str(text), max_words, overlap_words)

        for part_i, part_text in enumerate(parts):
            chunk_id = f"text_{global_i}" if len(parts) == 1 else f"text_{global_i}__part{part_i}"

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_type": "text",
                "location": location,
                "text": part_text
            })


def add_table_chunks(chunks, table, max_words, overlap_words):
    """Add one chunk per table row. Table header row is skipped."""
    if not isinstance(table, list) or len(table) < 2:
        return

    header = table[0]

    for row_i in range(1, len(table)):
        row_text = table_row_to_text(header, table[row_i])
        if not row_text:
            continue

        parts = split_long_text(row_text, max_words, overlap_words)

        for part_i, part_text in enumerate(parts):
            chunk_id = f"table_{row_i}" if len(parts) == 1 else f"table_{row_i}__part{part_i}"

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_type": "table",
                "location": "table",
                "text": part_text
            })


def build_chunks(example, max_words, overlap_words):
    """
    Build candidate chunks from only the model-visible context:
    pre_text + table + post_text.
    """
    chunks = []

    pre_text = ensure_list(example.get("pre_text", []))
    post_text = ensure_list(example.get("post_text", []))
    table = example.get("table", [])

    add_text_chunks(
        chunks=chunks,
        texts=pre_text,
        start_index=0,
        location="pre_text",
        max_words=max_words,
        overlap_words=overlap_words
    )

    add_table_chunks(
        chunks=chunks,
        table=table,
        max_words=max_words,
        overlap_words=overlap_words
    )

    add_text_chunks(
        chunks=chunks,
        texts=post_text,
        start_index=len(pre_text),
        location="post_text",
        max_words=max_words,
        overlap_words=overlap_words
    )

    return chunks


def bm25_tokenize(text: str):
    """Simple keyword tokenization that keeps numbers, percentages, and finance terms."""
    return re.findall(r"[a-zA-Z0-9_%.$/-]+", str(text).lower())


# ---------------------------------------------------------------------
# BM25 query expansion
# ---------------------------------------------------------------------

FINANCE_QUERY_EXPANSIONS = {
    # Keep this intentionally small and finance-focused.
    # These terms are added only to the BM25 query, not to the FAISS query.
    "revenue": ["sales", "net sales", "total revenue", "net revenue"],
    "revenues": ["sales", "net sales", "total revenues", "net revenues"],
    "sales": ["revenue", "revenues", "net sales"],
    "income": ["earnings", "profit", "profits", "net income", "operating income"],
    "earnings": ["income", "profit", "profits", "net income"],
    "profit": ["income", "earnings", "net income", "operating income"],
    "profits": ["income", "earnings", "net income", "operating income"],
    "expense": ["cost", "costs", "expenses"],
    "expenses": ["cost", "costs", "expense"],
    "cost": ["expense", "expenses", "costs"],
    "costs": ["expense", "expenses", "cost"],
    "debt": ["borrowings", "liabilities", "loans", "indebtedness"],
    "liability": ["debt", "liabilities", "obligation", "obligations"],
    "liabilities": ["debt", "liability", "obligations", "borrowings"],
    "cash": ["cash equivalents", "liquidity", "cash and cash equivalents"],
    "assets": ["total assets", "asset"],
    "equity": ["shareholders equity", "stockholders equity", "shareholder equity", "stockholder equity"],
    "inventory": ["inventories"],
    "inventories": ["inventory"],
    "tax": ["taxes", "income tax", "tax expense", "tax benefit"],
    "taxes": ["tax", "income tax", "tax expense", "tax benefit"],
}


def expand_finance_query(question: str) -> str:
    """
    Add a small set of finance synonyms to the BM25 query.

    This helps BM25 when the question and evidence use different common
    finance wording, such as "revenue" in the question but "net sales"
    in the table. The original question is preserved at the front.
    """
    question = str(question)
    question_lower = question.lower()
    expansion_terms = []
    seen = set()

    for trigger, synonyms in FINANCE_QUERY_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(trigger)}\b", question_lower):
            for synonym in synonyms:
                normalized = synonym.lower().strip()

                if not normalized:
                    continue

                # Avoid adding a phrase that is already present in the question
                # or that has already been added by another trigger.
                if normalized in question_lower or normalized in seen:
                    continue

                seen.add(normalized)
                expansion_terms.append(synonym)

    if not expansion_terms:
        return question

    return remove_extra_spaces(question + " " + " ".join(expansion_terms))


def retrieve_faiss(model, question, chunks, top_k):
    """Dense semantic retrieval using SentenceTransformers + FAISS."""
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)

    scores, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx == -1:
            continue

        results.append({
            "chunk_index": int(idx),
            "rank": rank,
            "score": float(scores[0][rank - 1])
        })

    return results


def retrieve_bm25(question, chunks, top_k):
    """Keyword retrieval using BM25."""
    if not chunks:
        return []

    tokenized_chunks = [bm25_tokenize(chunk["text"]) for chunk in chunks]
    tokenized_query = bm25_tokenize(question)

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = np.argsort(scores)[::-1][:min(top_k, len(chunks))]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "chunk_index": int(idx),
            "rank": rank,
            "score": float(scores[idx])
        })

    return results


def text_index_from_chunk_id(chunk_id: str):
    """Return the numeric index from text_N or text_N__partM chunk IDs."""
    match = re.match(r"^text_(\d+)(?:__part\d+)?$", str(chunk_id))
    return int(match.group(1)) if match else None


def build_text_neighbor_lookup(chunks):
    """Map each FinQA-style text index to its chunk indices."""
    lookup = {}

    for chunk_index, chunk in enumerate(chunks):
        if chunk.get("chunk_type") != "text":
            continue

        text_index = text_index_from_chunk_id(chunk.get("chunk_id"))

        if text_index is None:
            continue

        lookup.setdefault(text_index, []).append(chunk_index)

    return lookup


def make_retrieved_chunk_record(chunk, selected_by=None, neighbor_of=None):
    """Create the saved retrieved-chunk record with consistent metadata fields."""
    return {
        "chunk_id": chunk["chunk_id"],
        "chunk_type": chunk["chunk_type"],
        "location": chunk["location"],
        "text": chunk["text"],
        "selected_by": list(selected_by or []),
        "neighbor_of": neighbor_of,
        "faiss_rank": None,
        "faiss_score": None,
        "bm25_rank": None,
        "bm25_score": None
    }


def merge_results(
    chunks,
    faiss_results,
    bm25_results,
    max_chunks,
    include_text_neighbors=True,
    text_neighbor_window=1,
    max_primary_chunks=None
):
    """
    Interleaved hybrid merge with optional nearby text chunks.

    Primary retrieval order:
    1. Add FAISS rank 1.
    2. Add BM25 rank 1.
    3. Add FAISS rank 2.
    4. Add BM25 rank 2.
    5. Continue until max_primary_chunks unique primary chunks are selected.

    If text-neighbor inclusion is enabled, the script then adds nearby text
    chunks such as text_22 and text_24 around a retrieved text_23 when space
    remains under max_chunks. This gives the generator a little surrounding
    context without using gold evidence labels.
    """
    merged = {}

    if max_primary_chunks is None:
        max_primary_chunks = max_chunks

    max_primary_chunks = max(0, min(max_primary_chunks, max_chunks))

    def add_chunk(chunk_index, method_name=None, result=None, neighbor_of=None, limit=None):
        if limit is None:
            limit = max_chunks

        chunk = chunks[chunk_index]
        chunk_id = chunk["chunk_id"]

        if chunk_id not in merged:
            if len(merged) >= limit:
                return False

            selected_by = [method_name] if method_name else []
            merged[chunk_id] = make_retrieved_chunk_record(
                chunk=chunk,
                selected_by=selected_by,
                neighbor_of=neighbor_of
            )
        elif method_name and method_name not in merged[chunk_id]["selected_by"]:
            merged[chunk_id]["selected_by"].append(method_name)

        if method_name == "faiss" and result is not None:
            merged[chunk_id]["faiss_rank"] = result["rank"]
            merged[chunk_id]["faiss_score"] = result["score"]

        if method_name == "bm25" and result is not None:
            merged[chunk_id]["bm25_rank"] = result["rank"]
            merged[chunk_id]["bm25_score"] = result["score"]

        return True

    def add_result(result, method_name, limit):
        return add_chunk(
            chunk_index=result["chunk_index"],
            method_name=method_name,
            result=result,
            limit=limit
        )

    max_len = max(len(faiss_results), len(bm25_results))

    for i in range(max_len):
        if i < len(faiss_results):
            add_result(faiss_results[i], "faiss", max_primary_chunks)

        if i < len(bm25_results):
            add_result(bm25_results[i], "bm25", max_primary_chunks)

        if len(merged) >= max_primary_chunks:
            break

    primary_chunk_ids = list(merged.keys())

    if include_text_neighbors and text_neighbor_window > 0 and len(merged) < max_chunks:
        text_neighbor_lookup = build_text_neighbor_lookup(chunks)

        for primary_chunk_id in primary_chunk_ids:
            primary_text_index = text_index_from_chunk_id(primary_chunk_id)

            if primary_text_index is None:
                continue

            for distance in range(1, text_neighbor_window + 1):
                for neighbor_text_index in (primary_text_index - distance, primary_text_index + distance):
                    for neighbor_chunk_index in text_neighbor_lookup.get(neighbor_text_index, []):
                        if len(merged) >= max_chunks:
                            break

                        add_chunk(
                            chunk_index=neighbor_chunk_index,
                            method_name="neighbor",
                            neighbor_of=primary_chunk_id,
                            limit=max_chunks
                        )

                    if len(merged) >= max_chunks:
                        break

                if len(merged) >= max_chunks:
                    break

            if len(merged) >= max_chunks:
                break

    return list(merged.values())


def base_chunk_id(chunk_id: str) -> str:
    """Used only for optional retrieval-overlap analysis."""
    return chunk_id.split("__part")[0]


def gold_overlap(example, retrieved_chunks):
    """
    Optional analysis only.
    This does not affect retrieval.
    """
    gold_inds = example.get("qa", {}).get("gold_inds", {})

    if not gold_inds:
        return None

    gold_ids = set(gold_inds.keys())
    retrieved_ids = {base_chunk_id(chunk["chunk_id"]) for chunk in retrieved_chunks}

    matched = gold_ids.intersection(retrieved_ids)

    return {
        "gold_count": len(gold_ids),
        "matched_gold_count": len(matched),
        "matched_gold_ids": sorted(matched),
        "gold_recall": len(matched) / len(gold_ids) if gold_ids else 0.0
    }


def process_split(input_file, output_file, model, args):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_records = []

    total_candidate_chunks = 0
    total_retrieved_chunks = 0
    gold_recalls = []

    for example in tqdm(data, desc=f"Processing {input_file.name}"):
        question = example["qa"]["question"]

        chunks = build_chunks(
            example=example,
            max_words=args.max_chunk_words,
            overlap_words=args.overlap_words
        )

        faiss_results = retrieve_faiss(
            model=model,
            question=question,
            chunks=chunks,
            top_k=args.faiss_top_k
        )

        bm25_question = expand_finance_query(question) if args.expand_bm25_query else question

        bm25_results = retrieve_bm25(
            question=bm25_question,
            chunks=chunks,
            top_k=args.bm25_top_k
        )

        retrieved_chunks = merge_results(
            chunks=chunks,
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            max_chunks=args.max_final_chunks,
            include_text_neighbors=args.include_text_neighbors,
            text_neighbor_window=args.text_neighbor_window,
            max_primary_chunks=args.max_primary_chunks
        )

        overlap = gold_overlap(example, retrieved_chunks)

        if overlap is not None:
            gold_recalls.append(overlap["gold_recall"])

        total_candidate_chunks += len(chunks)
        total_retrieved_chunks += len(retrieved_chunks)

        output_records.append({
            "id": example["id"],
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_debug": {
                "num_candidate_chunks": len(chunks),
                "num_retrieved_chunks": len(retrieved_chunks),
                "num_primary_chunk_limit": args.max_primary_chunks,
                "num_neighbor_chunks": sum(1 for chunk in retrieved_chunks if "neighbor" in chunk.get("selected_by", [])),
                "bm25_query_expanded": bm25_question != question,
                "bm25_query_used": bm25_question,
                "gold_overlap_analysis": overlap
            }
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2)

    avg_candidates = total_candidate_chunks / len(data) if data else 0
    avg_retrieved = total_retrieved_chunks / len(data) if data else 0
    avg_gold_recall = sum(gold_recalls) / len(gold_recalls) if gold_recalls else None

    print(f"\nFinished: {input_file}")
    print(f"Saved to: {output_file}")
    print(f"Examples: {len(data)}")
    print(f"Average candidate chunks/example: {avg_candidates:.2f}")
    print(f"Average retrieved chunks/example: {avg_retrieved:.2f}")

    if avg_gold_recall is not None:
        print(f"Average gold overlap recall: {avg_gold_recall:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Build interleaved hybrid FAISS + BM25 retrieved context files for expanded FinQA."
    )

    parser.add_argument("--input_dir", default="long_dataset")
    parser.add_argument("--output_dir", default="rag_retrieved_interleaved_plus")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")

    parser.add_argument("--faiss_top_k", type=int, default=8)
    parser.add_argument("--bm25_top_k", type=int, default=8)
    parser.add_argument("--max_final_chunks", type=int, default=10)
    parser.add_argument(
        "--max_primary_chunks",
        type=int,
        default=8,
        help="Maximum interleaved FAISS/BM25 chunks before adding optional neighboring text chunks."
    )

    parser.add_argument("--max_chunk_words", type=int, default=180)
    parser.add_argument("--overlap_words", type=int, default=30)

    parser.add_argument(
        "--expand_bm25_query",
        dest="expand_bm25_query",
        action="store_true",
        default=True,
        help="Add a small finance synonym expansion to the BM25 query."
    )
    parser.add_argument(
        "--no_expand_bm25_query",
        dest="expand_bm25_query",
        action="store_false",
        help="Disable finance synonym expansion for the BM25 query."
    )

    parser.add_argument(
        "--include_text_neighbors",
        dest="include_text_neighbors",
        action="store_true",
        default=True,
        help="Add nearby text chunks around retrieved text chunks when space remains."
    )
    parser.add_argument(
        "--no_text_neighbors",
        dest="include_text_neighbors",
        action="store_false",
        help="Disable nearby text chunk inclusion."
    )
    parser.add_argument(
        "--text_neighbor_window",
        type=int,
        default=1,
        help="Number of neighboring text indices to try on each side of a retrieved text chunk."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["long_train", "long_dev", "long_test", "long_private_test"],
        help="Dataset splits to process."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    config = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "embedding_model": args.embedding_model,
        "faiss_top_k": args.faiss_top_k,
        "bm25_top_k": args.bm25_top_k,
        "max_final_chunks": args.max_final_chunks,
        "max_primary_chunks": args.max_primary_chunks,
        "max_chunk_words": args.max_chunk_words,
        "overlap_words": args.overlap_words,
        "expand_bm25_query": args.expand_bm25_query,
        "include_text_neighbors": args.include_text_neighbors,
        "text_neighbor_window": args.text_neighbor_window,
        "splits": args.splits,
        "merge_strategy": "interleaved_faiss_bm25_with_optional_text_neighbors",
        "output_file_suffix": "interleaved_plus_retrieved",
        "note": "Gold fields are not used to select chunks. Gold overlap is analysis only."
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "retrieval_config_interleaved_plus.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for split in args.splits:
        input_file = input_dir / f"{split}.json"
        output_file = output_dir / f"{split}_interleaved_plus_retrieved.json"

        if not input_file.exists():
            print(f"Skipping missing file: {input_file}")
            continue

        process_split(
            input_file=input_file,
            output_file=output_file,
            model=model,
            args=args
        )


if __name__ == "__main__":
    main()