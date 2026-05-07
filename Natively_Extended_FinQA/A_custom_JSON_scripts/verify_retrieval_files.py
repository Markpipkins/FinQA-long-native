import json
from pathlib import Path

pairs = [
    ("long_dataset/long_train.json", "rag_retrieved/long_train_retrieved.json"),
    ("long_dataset/long_dev.json", "rag_retrieved/long_dev_retrieved.json"),
    ("long_dataset/long_test.json", "rag_retrieved/long_test_retrieved.json"),
    ("long_dataset/long_private_test.json", "rag_retrieved/long_private_test_retrieved.json"),
]

for source_path, retrieved_path in pairs:
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    retrieved = json.loads(Path(retrieved_path).read_text(encoding="utf-8"))

    source_ids = [x["id"] for x in source]
    retrieved_ids = [x["id"] for x in retrieved]

    print("\n", retrieved_path)
    print("source count:", len(source))
    print("retrieved count:", len(retrieved))
    print("ids match:", source_ids == retrieved_ids)
    print("empty retrieved:", sum(len(x["retrieved_chunks"]) == 0 for x in retrieved))
    print("min chunks:", min(len(x["retrieved_chunks"]) for x in retrieved))
    print("max chunks:", max(len(x["retrieved_chunks"]) for x in retrieved))