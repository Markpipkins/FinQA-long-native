# """
# GROUP BY FILENAME:
# Groups finance QA dataset entries by the two leftmost path components
# of their "filename" field (e.g., "TFX/2018/page_47.pdf" -> "TFX/2018/").

# Usage:
#     python group_by_filename.py <input_file> <output_file>

# Defaults to 'input.json' and 'grouped_output.json' if not provided.
# """

# import json
# import sys
# from collections import defaultdict


# def group_by_filename(input_path: str, output_path: str) -> None:
#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     groups = defaultdict(list)

#     for entry in data:
#         filename = entry.get("filename", "")
#         parts = filename.split("/")
#         # Key is "TICKER/YEAR/" using the two leftmost slash-separated components
#         key = f"{parts[0]}/{parts[1]}/" if len(parts) >= 2 else filename
#         groups[key].append(entry)

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(groups, f)

#     total_groups = len(groups)
#     total_entries = sum(len(v) for v in groups.values())
#     print(f"Done. {total_entries} entries grouped into {total_groups} groups -> {output_path}")


# if __name__ == "__main__":
#     input_file  = sys.argv[1] if len(sys.argv) > 1 else "input.json"
#     output_file = sys.argv[2] if len(sys.argv) > 2 else "grouped_output.json"
#     group_by_filename(input_file, output_file)
