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


def merge_results(chunks, faiss_results, bm25_results, max_chunks):
    """
    Simple merge:
    1. Add FAISS chunks first.
    2. Add BM25 chunks next.
    3. Deduplicate by chunk_id.
    4. Stop after max_chunks.
    """
    merged = {}

    def add_result(result, method_name):
        chunk = chunks[result["chunk_index"]]
        chunk_id = chunk["chunk_id"]

        if chunk_id not in merged:
            if len(merged) >= max_chunks:
                return

            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_type": chunk["chunk_type"],
                "location": chunk["location"],
                "text": chunk["text"],
                "selected_by": [],
                "faiss_rank": None,
                "faiss_score": None,
                "bm25_rank": None,
                "bm25_score": None
            }

        if method_name not in merged[chunk_id]["selected_by"]:
            merged[chunk_id]["selected_by"].append(method_name)

        if method_name == "faiss":
            merged[chunk_id]["faiss_rank"] = result["rank"]
            merged[chunk_id]["faiss_score"] = result["score"]

        if method_name == "bm25":
            merged[chunk_id]["bm25_rank"] = result["rank"]
            merged[chunk_id]["bm25_score"] = result["score"]

    for result in faiss_results:
        add_result(result, "faiss")

    for result in bm25_results:
        add_result(result, "bm25")

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

        bm25_results = retrieve_bm25(
            question=question,
            chunks=chunks,
            top_k=args.bm25_top_k
        )

        retrieved_chunks = merge_results(
            chunks=chunks,
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            max_chunks=args.max_final_chunks
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
        description="Build hybrid FAISS + BM25 retrieved context files for expanded FinQA."
    )

    parser.add_argument("--input_dir", default="long_dataset")
    parser.add_argument("--output_dir", default="rag_retrieved")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")

    parser.add_argument("--faiss_top_k", type=int, default=5)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--max_final_chunks", type=int, default=8)

    parser.add_argument("--max_chunk_words", type=int, default=180)
    parser.add_argument("--overlap_words", type=int, default=30)

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
        "max_chunk_words": args.max_chunk_words,
        "overlap_words": args.overlap_words,
        "splits": args.splits,
        "note": "Gold fields are not used to select chunks. Gold overlap is analysis only."
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "retrieval_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for split in args.splits:
        input_file = input_dir / f"{split}.json"
        output_file = output_dir / f"{split}_retrieved.json"

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