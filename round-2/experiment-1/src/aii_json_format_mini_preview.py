#!/usr/bin/env python
"""
JSON Formatter - Generate full, mini, and preview versions

Creates three versions of a JSON file:
- full: Identical to original
- mini: Only first 3 items from primary array
- preview: Mini version with all strings truncated to 200 chars

Supports both bare arrays and dict-wrapped arrays (e.g. {"examples": [...]}).
Use --format to specify which schema format to use (determines the array key).

Usage:
    python aii_json_format_mini_preview.py --input data.json
    python aii_json_format_mini_preview.py --format exp_sel_data_out --input data.json
    python aii_json_format_mini_preview.py --format exp_sel_data_out --input data.json --output-dir ./output
"""

import argparse
import sys
from pathlib import Path

try:
    from aii_lib.abilities.aii_ability import aii_ability
except ImportError:  # standalone use: aii_lib / ability server not installed

    def aii_ability(*_args, **_kwargs):
        """No-op decorator fallback (the real one only attaches server metadata)."""

        def _decorator(func):
            return func

        return _decorator


SERVER_NAME = "aii_json__format"
DEFAULT_TIMEOUT = 60.0

# Configuration
MAX_ARRAY_ITEMS = 3
MAX_STRING_LENGTH = 200
TRUNCATE_MARKER = "..."

# Format → primary array key mapping (must match schemas in ../schemas/)
# For datasets-grouped schemas, the top-level key is "datasets" and each
# entry contains an "examples" array that also needs slicing.
FORMAT_ARRAY_KEY: dict[str, str] = {
    "exp_sel_data_out": "datasets",
    "exp_gen_sol_out": "datasets",
    "exp_eval_sol_out": "datasets",
    "exp_proof_out": "lemmas",
}

# Schemas that use datasets-grouped structure (need nested example slicing)
DATASETS_GROUPED_FORMATS = {"exp_sel_data_out", "exp_gen_sol_out", "exp_eval_sol_out"}


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================


def init_json_format():
    """Initialize JSON format environment with warmup."""
    import json

    # Warmup: parse/serialize a small JSON to warm up the module
    json.loads(json.dumps({"warmup": True}))


@aii_ability(
    name="aii_json__format",
    description="Generate full, mini, and preview versions of a JSON file.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_json_format",
)
def core_json_format(
    input_file: str = "", output_dir: str | None = None, format_type: str | None = None
) -> dict:
    """
    Generate full, mini, and preview versions of JSON file.

    Args:
        input_file: Path to input JSON file
        output_dir: Optional output directory (defaults to same as input)
        format_type: Optional schema format (e.g. "exp_sel_data_out") to determine array key

    Returns:
        Dict with success status and output file paths
    """
    import json

    def truncate_value(value):
        """Recursively truncate JSON data for preview."""
        if isinstance(value, list):
            return [truncate_value(item) for item in value[:MAX_ARRAY_ITEMS]]
        if isinstance(value, str):
            if len(value) > MAX_STRING_LENGTH:
                return value[:MAX_STRING_LENGTH] + TRUNCATE_MARKER
            return value
        if isinstance(value, dict):
            return {key: truncate_value(val) for key, val in value.items()}
        return value

    if not input_file:
        return {"success": False, "error": "input_file is required"}
    _project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    _resolved = Path(input_file).resolve()
    if not any(_resolved == d or d in _resolved.parents for d in [_project_root, Path("/tmp")]):
        return {
            "success": False,
            "error": "input_file must be under the project directory or /tmp",
        }
    # Use the absolute resolved path everywhere downstream — agents pass
    # relative paths from their own CWD which differs from the server's;
    # the security check above already confirmed the resolved location.
    input_path = _resolved

    if not input_path.exists():
        return {"success": False, "error": f"Input file does not exist: {input_path}"}

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = input_path.parent

    base_name = input_path.stem

    # Load JSON data
    try:
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in input file: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read input file: {e}"}

    # Determine the primary array to slice
    wrapper_key = None
    is_datasets_grouped = format_type in DATASETS_GROUPED_FORMATS if format_type else False
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if format_type and format_type in FORMAT_ARRAY_KEY:
            wrapper_key = FORMAT_ARRAY_KEY[format_type]
        # Auto-detect: check for "datasets" first, then fall back to "examples"
        elif "datasets" in data:
            wrapper_key = "datasets"
            is_datasets_grouped = True
        elif "examples" in data:
            wrapper_key = "examples"
        else:
            return {
                "success": False,
                "error": f"No 'datasets' or 'examples' key found in JSON (keys: {', '.join(data.keys())}). Pass --format to specify the schema (one of: {', '.join(sorted(FORMAT_ARRAY_KEY))})",
            }
        if wrapper_key not in data:
            return {
                "success": False,
                "error": f"Key '{wrapper_key}' not found in JSON (keys: {', '.join(data.keys())}). Pass --format to specify the schema (one of: {', '.join(sorted(FORMAT_ARRAY_KEY))})",
            }
        if not isinstance(data[wrapper_key], list):
            return {"success": False, "error": f"Key '{wrapper_key}' is not an array"}
        items = data[wrapper_key]
    else:
        return {
            "success": False,
            "error": f"Input JSON must be an array or object, got {type(data).__name__}",
        }

    def _slice_dataset_examples(datasets_arr, max_per_dataset):
        """For datasets-grouped schemas, slice examples within each dataset."""
        return [
            {**ds, "examples": ds.get("examples", [])[:max_per_dataset]}
            if isinstance(ds, dict)
            else ds
            for ds in datasets_arr
        ]

    def _wrap(arr):
        """Re-wrap array in original dict structure if input was a dict."""
        if wrapper_key is not None:
            return {**data, wrapper_key: arr}
        return arr

    def _count_total_examples(datasets_arr):
        """Count total examples across all datasets."""
        return sum(len(ds.get("examples", [])) for ds in datasets_arr if isinstance(ds, dict))

    # Generate versions
    full_data = _wrap(items)
    full_file = out_dir / f"full_{base_name}.json"

    if is_datasets_grouped:
        # Keep all datasets, slice examples within each to MAX_ARRAY_ITEMS
        mini_items = _slice_dataset_examples(items, MAX_ARRAY_ITEMS)
        mini_data = _wrap(mini_items)
        mini_count = _count_total_examples(mini_items)

        preview_items = _slice_dataset_examples(items, MAX_ARRAY_ITEMS)
        preview_data = truncate_value(_wrap(preview_items))
        preview_count = _count_total_examples(preview_items)

        full_count = _count_total_examples(items)
    else:
        mini_data = _wrap(items[:MAX_ARRAY_ITEMS])
        mini_count = min(MAX_ARRAY_ITEMS, len(items))

        preview_data = truncate_value(_wrap(items[:MAX_ARRAY_ITEMS]))
        preview_count = min(MAX_ARRAY_ITEMS, len(items))

        full_count = len(items)

    mini_file = out_dir / f"mini_{base_name}.json"
    preview_file = out_dir / f"preview_{base_name}.json"

    # Save all three versions
    try:
        with open(full_file, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)

        with open(mini_file, "w", encoding="utf-8") as f:
            json.dump(mini_data, f, indent=2, ensure_ascii=False)

        with open(preview_file, "w", encoding="utf-8") as f:
            json.dump(preview_data, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "full_file": str(full_file),
            "mini_file": str(mini_file),
            "preview_file": str(preview_file),
            "full_count": full_count,
            "mini_count": mini_count,
            "preview_count": preview_count,
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to save output files: {e}"}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate full, mini, and preview versions of a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aii_json_format_mini_preview.py --input data.json
  python aii_json_format_mini_preview.py --format exp_sel_data_out --input data.json
  python aii_json_format_mini_preview.py --format exp_sel_data_out --input data.json --output-dir ./output
        """,
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=list(FORMAT_ARRAY_KEY.keys()),
        help="Schema format (required for object-type JSONs)",
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    args = parser.parse_args()

    params = {
        "input_file": args.input,
        "output_dir": args.output_dir,
        "format_type": args.format,
    }

    result = None
    try:
        from aii_lib.abilities.ability_server import call_server

        result = call_server(SERVER_NAME, params, timeout=DEFAULT_TIMEOUT)
    except Exception:
        result = None

    if result is None:
        # Standalone fallback: run the core logic locally (no ability server needed).
        init_json_format()
        result = core_json_format(**params)

    if result.get("success"):
        print("Generated 3 versions:")
        print(f"  Full ({result['full_count']} items): {result['full_file']}")
        print(f"  Mini ({result['mini_count']} items): {result['mini_file']}")
        print(f"  Preview ({result['preview_count']} items, truncated): {result['preview_file']}")
        sys.exit(0)
    else:
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
