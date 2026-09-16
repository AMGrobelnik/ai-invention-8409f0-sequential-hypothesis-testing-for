#!/usr/bin/env python3
"""MinHash + SPRT experiment on Quora duplicate pairs.

Compares sequential SPRT stopping against exhaustive MinHash comparison,
threshold truncation, and Chernoff consecutive-k rule.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import resource
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ijson
import numpy as np
from loguru import logger

# =============================================================================
# Constants / Paths
# =============================================================================
WORKSPACE = Path(
    "/home/adrian/projects/ai-inventor-runtest/aii_data/users/admin/runs/run_kDDM2LtKR7zJ/3_invention_loop/iter_2/gen_art/gen_art_experiment_1"
)
DEP_DATA = Path(
    "/home/adrian/projects/ai-inventor-runtest/aii_data/users/admin/runs/run_kDDM2LtKR7zJ/3_invention_loop/iter_1/gen_art/gen_art_dataset_1"
)
DATA_PATH = DEP_DATA / "mini_data_out.json"
OUTPUT_DIR = WORKSPACE / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_PAIRS = 5000
LIMIT_PER_CHUNK = 2000

# =============================================================================
# Logging
# =============================================================================
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss}|{level:<7}|{message}",
)
logger.add(
    WORKSPACE / "logs" / "run.log",
    rotation="30 MB",
    level="DEBUG",
)

# =============================================================================
# Resource limits
# =============================================================================
def set_resource_limits() -> None:
    """Set conservative RAM/CPU limits for container safety."""
    try:
        _avail = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ram_budget = min(4 * 1024**3, max(_avail, 768 * 1024**2))
        resource.setrlimit(resource.RLIMIT_AS, (ram_budget, ram_budget))
        resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
        logger.info(f"Resource limits set: RAM={ram_budget/1e9:.2f}GB, CPU=3600s")
    except Exception as exc:  # pragma: no cover - platform-specific
        logger.warning(f"Could not set resource limits: {exc}")


set_resource_limits()

# =============================================================================
# Data structures
# =============================================================================
@dataclass
class Pair:
    sentence1: str
    sentence2: str
    label: int
    row_index: int
    estimated_jaccard: float = 0.0
    sig1: list[int] = field(default_factory=list)
    sig2: list[int] = field(default_factory=list)


@dataclass
class MethodResult:
    prediction: int
    components_examined: int
    final_llr: float | None = None
    decision: str | None = None
    boundary_crossed: bool | None = None
    stopped_by: str | None = None


# =============================================================================
# Helpers
# =============================================================================
def deterministic_hash(idx: int, shingle: str) -> int:
    h = hashlib.sha256(f"{idx}:{shingle}".encode("utf-8"))
    return int(h.hexdigest()[:8], 16) & 0xFFFFFFFF


def get_shingles(text: str, k: int = 3) -> set[str]:
    tokens = text.lower().split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def estimate_jaccard(s1: str, s2: str, k: int = 3) -> float:
    set1 = get_shingles(s1, k)
    set2 = get_shingles(s2, k)
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def minhash_signature(text: str, k: int) -> list[int]:
    shingles = get_shingles(text)
    if not shingles:
        return [0] * k
    signature = []
    for i in range(k):
        min_h = 0xFFFFFFFF
        for s in shingles:
            h = deterministic_hash(i, s)
            if h < min_h:
                min_h = h
        signature.append(min_h)
    return signature


# =============================================================================
# Method implementations
# =============================================================================
def exhaustive_comparison(sig1: list[int], sig2: list[int], threshold: float = 0.5) -> MethodResult:
    k = len(sig1)
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    estimate = matches / k
    return MethodResult(
        prediction=1 if estimate >= threshold else 0,
        components_examined=k,
        final_llr=float(matches),
        decision="exhaustive",
        boundary_crossed=False,
        stopped_by="exhaustive",
    )


def sprt_comparison(
    sig1: list[int],
    sig2: list[int],
    t: float,
    delta: float,
    alpha: float,
    beta: float,
) -> MethodResult:
    k = len(sig1)
    p0 = max(t - delta, 1e-10)
    p1 = min(t + delta, 1.0 - 1e-10)
    log_p1_p0 = math.log(p1 / p0)
    log_1mp1_1mp0 = math.log((1.0 - p1) / (1.0 - p0))
    a = math.log((1.0 - beta) / alpha)
    b = math.log(beta / (1.0 - alpha))

    cumulative = 0.0
    components_examined = 0
    decision = None

    for j in range(k):
        if sig1[j] == sig2[j]:
            cumulative += log_p1_p0
        else:
            cumulative += log_1mp1_1mp0
        components_examined = j + 1

        if cumulative >= a:
            decision = "similar"
            break
        elif cumulative <= b:
            decision = "dissimilar"
            break

    if decision is None:
        decision = "similar" if cumulative >= 0 else "dissimilar"

    return MethodResult(
        prediction=1 if decision == "similar" else 0,
        components_examined=components_examined,
        final_llr=cumulative,
        decision=decision,
        boundary_crossed=decision is not None and components_examined < k,
        stopped_by="boundary" if decision is not None and components_examined < k else "exhaustive",
    )


def threshold_truncation(sig1: list[int], sig2: list[int], threshold: float = 0.5) -> MethodResult:
    k = len(sig1)
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    estimate = matches / k
    return MethodResult(
        prediction=1 if estimate >= threshold else 0,
        components_examined=k,
        final_llr=float(matches),
        decision="threshold",
        boundary_crossed=False,
        stopped_by="exhaustive",
    )


def chernoff_consecutive(sig1: list[int], sig2: list[int], consec_k: int = 16) -> MethodResult:
    k = len(sig1)
    run_matches = 0
    run_mismatches = 0
    components_examined = 0
    cumulative_llr = 0.0

    for j in range(k):
        if sig1[j] == sig2[j]:
            run_matches += 1
            run_mismatches = 0
            cumulative_llr += 1.0
        else:
            run_mismatches += 1
            run_matches = 0
            cumulative_llr -= 1.0
        components_examined = j + 1

        if run_matches >= consec_k:
            return MethodResult(
                prediction=1,
                components_examined=components_examined,
                final_llr=cumulative_llr,
                decision="consecutive_matches",
                boundary_crossed=True,
                stopped_by="consecutive_matches",
            )
        if run_mismatches >= consec_k:
            return MethodResult(
                prediction=0,
                components_examined=components_examined,
                final_llr=cumulative_llr,
                decision="consecutive_mismatches",
                boundary_crossed=True,
                stopped_by="consecutive_mismatches",
            )

    matches_total = sum(1 for a, b in zip(sig1, sig2) if a == b)
    prediction = 1 if matches_total >= k / 2 else 0
    return MethodResult(
        prediction=prediction,
        components_examined=components_examined,
        final_llr=cumulative_llr,
        decision="exhaustive",
        boundary_crossed=False,
        stopped_by="exhaustive",
    )


# =============================================================================
# Evaluation metrics
# =============================================================================
def classification_metrics(preds: list[int], gt: list[int]) -> dict[str, Any]:
    tp = sum(1 for p, g in zip(preds, gt) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, gt) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, gt) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(preds, gt) if p == 0 and g == 0)
    n = len(preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / n if n > 0 else 0.0
    type1 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "type1_error": type1,
        "type2_error": type2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": n,
    }


def compute_metrics(
    results: list[dict[str, Any]],
    k_values: list[int],
    alpha_beta_list: list[tuple[float, float]],
    delta_values: list[float],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for k in k_values:
        for alpha, beta in alpha_beta_list:
            subset = [
                r for r in results
                if r["k"] == k and r["alpha"] == alpha and r["beta"] == beta
            ]
            if not subset:
                continue
            gt = [r["ground_truth"] for r in subset]

            key_base = f"k{k}_a{alpha}_b{beta}"

            # Exhaustive
            ex_preds = [r["exhaustive"]["prediction"] for r in subset]
            metrics[f"{key_base}_exhaustive"] = classification_metrics(ex_preds, gt)

            # Threshold truncation
            tt_preds = [r["threshold_truncation"]["prediction"] for r in subset]
            metrics[f"{key_base}_threshold_truncation"] = classification_metrics(tt_preds, gt)

            # Chernoff
            c_preds = [r["chernoff"]["prediction"] for r in subset]
            metrics[f"{key_base}_chernoff"] = classification_metrics(c_preds, gt)
            c_ce = [r["chernoff"]["components_examined"] for r in subset]
            metrics[f"{key_base}_chernoff_savings"] = {
                "avg_components": float(np.mean(c_ce)),
                "pct_of_k": float(np.mean(c_ce) / k),
                "savings_pct": float(100 * (1 - np.mean(c_ce) / k)),
            }

            # SPRT by delta
            for delta in delta_values:
                sprt_preds = [r[f"sprt_d{delta}"]["prediction"] for r in subset]
                metrics[f"{key_base}_sprt_d{delta}"] = classification_metrics(sprt_preds, gt)
                ce = [r[f"sprt_d{delta}"]["components_examined"] for r in subset]
                metrics[f"{key_base}_sprt_d{delta}_savings"] = {
                    "avg_components": float(np.mean(ce)),
                    "median_components": float(np.median(ce)),
                    "pct_of_k": float(np.mean(ce) / k),
                    "savings_pct": float(100 * (1 - np.mean(ce) / k)),
                }
    return metrics


# =============================================================================
# Experiment runner
# =============================================================================
def load_pairs_chunked() -> list[Pair]:
    """Stream pairs from all full_data_out chunks sequentially until TARGET_PAIRS."""
    logger.info("Loading pairs sequentially from chunks")
    pairs: list[Pair] = []
    cumulative = 0
    chunk_files = sorted(DEP_DATA.glob("full_data_out/full_data_out_*.json"))
    for chunk_path in chunk_files:
        if len(pairs) >= TARGET_PAIRS:
            break
        remaining = TARGET_PAIRS - len(pairs)
        chunk_limit = min(LIMIT_PER_CHUNK, remaining)
        with chunk_path.open("rb") as f:
            examples = ijson.items(f, "datasets.item.examples.item")
            for ex in examples:
                if len(pairs) >= TARGET_PAIRS:
                    break
                if chunk_limit is not None and len(pairs) - cumulative >= chunk_limit:
                    break
                try:
                    inp = json.loads(ex["input"])
                except Exception:
                    continue
                pairs.append(
                    Pair(
                        sentence1=inp.get("sentence1", ""),
                        sentence2=inp.get("sentence2", ""),
                        label=int(ex.get("output", 0)),
                        row_index=int(ex.get("metadata_row_index", len(pairs))),
                    )
                )
        cumulative = len(pairs)
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(
            f"Chunk {chunk_path.name}: cumulative={cumulative}, "
            f"pairs={len(pairs)}, mem_maxrss={mem_mb:.1f}MB"
        )
    logger.info(f"Loaded {len(pairs)} pairs total from {len(chunk_files)} chunk files")
    return pairs


def balanced_subsample(pairs: list[Pair], min_per_group: int = 2500) -> list[Pair]:
    for p in pairs:
        p.estimated_jaccard = estimate_jaccard(p.sentence1, p.sentence2)

    similar = [p for p in pairs if p.estimated_jaccard >= 0.4]
    dissimilar = [p for p in pairs if p.estimated_jaccard < 0.4]
    logger.info(f"Candidates: similar={len(similar)}, dissimilar={len(dissimilar)}")

    random.shuffle(similar)
    random.shuffle(dissimilar)
    n_sim = min(min_per_group, len(similar))
    n_dissim = min(min_per_group, len(dissimilar))
    sampled = similar[:n_sim] + dissimilar[:n_dissim]
    random.shuffle(sampled)
    logger.info(f"Subsampled to {len(sampled)} pairs ({n_sim} similar, {n_dissim} dissimilar)")
    return sampled


def run_experiment(
    pairs: list[Pair],
    k_values: list[int],
    alpha_beta_list: list[tuple[float, float]],
    delta_values: list[float],
    t: float = 0.5,
    consec_k: int = 16,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for k in k_values:
        logger.info(f"=== k={k} ===")
        for p in pairs:
            p.sig1 = minhash_signature(p.sentence1, k)
            p.sig2 = minhash_signature(p.sentence2, k)

        for p in pairs:
            del p.sentence1
            del p.sentence2

        for alpha, beta in alpha_beta_list:
            logger.info(f"--- alpha={alpha}, beta={beta} ---")
            for pi, p in enumerate(pairs):
                result: dict[str, Any] = {
                    "k": k,
                    "alpha": alpha,
                    "beta": beta,
                    "row_index": p.row_index,
                    "ground_truth": p.label,
                    "estimated_jaccard": p.estimated_jaccard,
                }

                ex = exhaustive_comparison(p.sig1, p.sig2)
                result["exhaustive"] = {
                    "prediction": ex.prediction,
                    "components_examined": ex.components_examined,
                    "final_llr": ex.final_llr,
                    "decision": ex.decision,
                    "boundary_crossed": ex.boundary_crossed,
                    "stopped_by": ex.stopped_by,
                }

                for delta in delta_values:
                    sprt = sprt_comparison(p.sig1, p.sig2, t, delta, alpha, beta)
                    result[f"sprt_d{delta}"] = {
                        "prediction": sprt.prediction,
                        "components_examined": sprt.components_examined,
                        "final_llr": sprt.final_llr,
                        "decision": sprt.decision,
                        "boundary_crossed": sprt.boundary_crossed,
                        "stopped_by": sprt.stopped_by,
                    }

                tt = threshold_truncation(p.sig1, p.sig2)
                result["threshold_truncation"] = {
                    "prediction": tt.prediction,
                    "components_examined": tt.components_examined,
                    "final_llr": tt.final_llr,
                    "decision": tt.decision,
                    "boundary_crossed": tt.boundary_crossed,
                    "stopped_by": tt.stopped_by,
                }

                c = chernoff_consecutive(p.sig1, p.sig2, consec_k)
                result["chernoff"] = {
                    "prediction": c.prediction,
                    "components_examined": c.components_examined,
                    "final_llr": c.final_llr,
                    "decision": c.decision,
                    "boundary_crossed": c.boundary_crossed,
                    "stopped_by": c.stopped_by,
                }

                results.append(result)

                if (pi + 1) % 100 == 0:
                    gc.collect()
                    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    logger.info(
                        f"k={k} a={alpha} b={beta} pair={pi+1}/{len(pairs)}, "
                        f"results={len(results)}, mem_maxrss={mem_mb:.1f}MB"
                    )

        output_path = OUTPUT_DIR / f"method_out_k{k}.json"
        with output_path.open("w") as f:
            json.dump(
                {
                    "results": results,
                    "config": {
                        "k_values": k_values,
                        "alpha_beta_list": alpha_beta_list,
                        "delta_values": delta_values,
                        "t": t,
                        "consec_k": consec_k,
                        "n_pairs": len(pairs),
                    },
                },
                f,
            )
        logger.info(f"Saved results for k={k} to {output_path}")

    return {"results": results, "config": {"n_pairs": len(pairs), "k_values": k_values}}


def save_schema_output(results: list[dict[str, Any]], path: Path) -> None:
    """Write predictions in exp_gen_sol_out.json-like schema."""
    dataset_name = "sentence-transformers_quora-duplicates"
    examples = []
    for r in results:
        ex = {
            "input": json.dumps(
                {"row_index": r["row_index"]}
            ),
            "output": str(r["ground_truth"]),
            "metadata_row_index": r["row_index"],
        }
        # Include one representative prediction per example
        ex["predict_sprt"] = str(r["sprt_d0.1"]["prediction"])
        ex["predict_threshold_truncation"] = str(r["threshold_truncation"]["prediction"])
        ex["predict_chernoff"] = str(r["chernoff"]["prediction"])
        examples.append(ex)

    output = {
        "datasets": [
            {
                "dataset": dataset_name,
                "examples": examples,
            }
        ]
    }
    path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved schema output to {path}")


# =============================================================================
# Main
# =============================================================================
@logger.catch(reraise=True)
def main() -> None:
    logger.info("Starting MinHash+SPRT experiment")

    pairs = load_pairs_chunked()
    if len(pairs) < 10:
        raise RuntimeError("Insufficient pairs loaded for experiment")

    sampled = balanced_subsample(pairs, min_per_group=max(1, TARGET_PAIRS // 4))
    logger.info(f"Using {len(sampled)} pairs")

    k_values = [128]
    alpha_beta_list = [(0.05, 0.05)]
    delta_values = [0.05, 0.1, 0.15, 0.2]
    t = 0.5
    consec_k = 16

    experiment_results = run_experiment(
        sampled, k_values, alpha_beta_list, delta_values, t, consec_k
    )

    metrics = compute_metrics(
        experiment_results["results"], k_values, alpha_beta_list, delta_values
    )
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved metrics to {metrics_path}")

    # Save the full schema output to method_out.json in workspace root
    full_path = WORKSPACE / "method_out.json"
    save_schema_output(experiment_results["results"], full_path)
    logger.info(f"Saved schema output to {full_path}")

    # Generate mini and preview from method_out.json
    generate_mini_preview(full_path, WORKSPACE)
    logger.info("Generated mini and preview outputs")

    logger.info("Experiment complete")
    for key, val in sorted(metrics.items()):
        if isinstance(val, dict) and "f1" in val:
            logger.info(
                f"{key}: F1={val['f1']:.4f}, Acc={val['accuracy']:.4f}, "
                f"TypeI={val['type1_error']:.4f}, Type2={val['type2_error']:.4f}"
            )
        elif isinstance(val, dict) and "savings_pct" in val:
            logger.info(
                f"{key}: avg_components={val['avg_components']:.1f}, "
                f"savings={val['savings_pct']:.1f}%"
            )


def truncate_json_strings(obj, max_length=200):
    """Recursively truncate strings in a JSON-serializable object."""
    if isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:max_length] + "..."
        return obj
    elif isinstance(obj, list):
        return [truncate_json_strings(item, max_length) for item in obj]
    elif isinstance(obj, dict):
        return {key: truncate_json_strings(value, max_length) for key, value in obj.items()}
    else:
        return obj


def generate_mini_preview(input_path: Path, output_dir: Path) -> None:
    """Generate mini and preview versions of a schema output file.
    Assumes the input follows the exp_gen_sol_out schema (datasets -> examples).
    Mini: first 3 examples per dataset.
    Preview: mini with strings truncated to 200 characters.
    """
    with open(input_path) as f:
        data = json.load(f)

    # Mini: first 3 examples per dataset
    mini_data = data.copy()
    if "datasets" in mini_data:
        for dataset in mini_data["datasets"]:
            if "examples" in dataset:
                dataset["examples"] = dataset["examples"][:3]

    mini_path = output_dir / ("mini_" + input_path.name)
    with open(mini_path, 'w') as f:
        json.dump(mini_data, f, indent=2)

    # Preview: truncate strings in the mini data
    preview_data = truncate_json_strings(mini_data)
    preview_path = output_dir / ("preview_" + input_path.name)
    with open(preview_path, 'w') as f:
        json.dump(preview_data, f, indent=2)


if __name__ == "__main__":
    main()
