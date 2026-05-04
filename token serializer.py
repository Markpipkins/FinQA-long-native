"""
----------------
TOKEN SERIALIZER
----------------
This script calculates the number of tokens for each example in the dataset after serialization, 
and reports statistics such as average tokens, maximum tokens, and how many examples exceed a 
specified token limit. It uses the Hugging Face Transformers library to tokenize the serialized 
text according to the specified model's tokenizer.

Usage:
# change the elements in the FILES list to point to your dataset files first, 
# then run:

python validate_token_counts.py

"""

import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LIMIT = 12000
FILES = [
    "dataset/file1.json",
    "dataset/file2.json",
    "dataset/file3.json",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def serialize_example(ex):
    return f"""Question:
{ex["qa"]["question"]}

Pre-text:
{" ".join(ex.get("pre_text", [])) if isinstance(ex.get("pre_text"), list) else ex.get("pre_text", "")}

Table:
{ex.get("table", "")}

Post-text:
{" ".join(ex.get("post_text", [])) if isinstance(ex.get("post_text"), list) else ex.get("post_text", "")}

Answer with a FinQA program ending in EOF.
"""


for file in FILES:
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    counts = []

    for ex in data:
        text = serialize_example(ex)
        n_tokens = len(tokenizer.encode(text))
        counts.append((ex["id"], n_tokens))

    over = [x for x in counts if x[1] > LIMIT]

    print(f"\n{file}")
    print(f"Examples: {len(counts)}")
    print(f"Average tokens: {sum(c for _, c in counts) / len(counts):.1f}")
    print(f"Max tokens: {max(c for _, c in counts)}")
    print(f"Over {LIMIT}: {len(over)}")

    for ex_id, count in over[:20]:
        print(f"  {ex_id}: {count}")
