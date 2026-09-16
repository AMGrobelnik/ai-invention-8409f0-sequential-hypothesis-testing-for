#!/usr/bin/env python3
"""Generate full/mini/preview variants for a JSON array file."""

import json
from pathlib import Path

MAX_ARRAY_ITEMS = 3
MAX_STRING_LENGTH = 200
TRUNCATE_MARKER = "..."


def truncate_value(value):
    if isinstance(value, list):
        return [truncate_value(item) for item in value[:MAX_ARRAY_ITEMS]]
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + TRUNCATE_MARKER
        return value
    if isinstance(value, dict):
        return {key: truncate_value(val) for key, val in value.items()}
    return value


def main():
    input_path = Path("results/method_out_k128.json")
    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")
    data = json.loads(input_path.read_text())

    base = input_path.stem
    input_path.with_name(f"full_{base}.json").write_text(json.dumps(data, indent=2))
    mini = data.copy()
    if isinstance(mini, dict):
        for key, val in list(mini.items()):
            if isinstance(val, list):
                mini[key] = val[:MAX_ARRAY_ITEMS]
    elif isinstance(mini, list):
        mini = mini[:MAX_ARRAY_ITEMS]
    input_path.with_name(f"mini_{base}.json").write_text(json.dumps(mini, indent=2))
    preview = truncate_value(mini)
    input_path.with_name(f"preview_{base}.json").write_text(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
