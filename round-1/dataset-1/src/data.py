#!/usr/bin/env python3
"""Load duplicate-question datasets into exp_sel_data_out.json schema."""

import json
import os
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path("/home/adrian/projects/ai-inventor-runtest/aii_data/users/admin/runs/run_kDDM2LtKR7zJ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"


def load_dataset(name: str, json_dir: Path, filename: str, feature_fields: list[str], label_field: str) -> dict:
    """Load a JSON dataset and convert to exp_sel_data_out schema group."""
    json_path = json_dir / filename
    logger.info(f"Loading {name} from {json_path}")
    with open(json_path) as f:
        records = json.load(f)
    examples = []
    for i, row in enumerate(records):
        input_obj = {k: row[k] for k in feature_fields}
        example = {
            "input": json.dumps(input_obj, ensure_ascii=False),
            "output": str(row[label_field]),
            "metadata_fold": 0,
            "metadata_feature_names": feature_fields,
            "metadata_task_type": "binary_classification",
            "metadata_n_classes": 2,
            "metadata_row_index": i,
        }
        examples.append(example)
    return {"dataset": name, "examples": examples}


def main():
    logger.info("Starting data preparation")
    datasets = []
    datasets.append(
        load_dataset(
            "sentence-transformers_quora-duplicates",
            DATASETS_DIR / "full_sentence-transformers_quora-duplicates_pair-class_train",
            "full_sentence-transformers_quora-duplicates_pair-class_train_1.json",
            feature_fields=["sentence1", "sentence2"],
            label_field="label",
        )
    )
    output = {"datasets": datasets}
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False))
    logger.info(f"Saved {OUTPUT_PATH} with {sum(len(d['examples']) for d in datasets)} examples across {len(datasets)} datasets")


if __name__ == "__main__":
    main()
