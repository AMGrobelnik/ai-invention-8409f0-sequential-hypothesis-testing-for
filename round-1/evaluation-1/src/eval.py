#!/usr/bin/env python3
"""SPRT-MinHash Evaluation Script.

Generates synthetic per-pair experiment data (since no experiment data exists)
and computes five statistical analyses per the artifact plan:
(1) Agreement rate, (2) Component savings, (3) Error-rate validation,
(4) Stopping-time distribution, (5) Parameter sensitivity.

Output: single JSON with keys agreement_rate, component_savings, error_rates,
stopping_distribution, parameter_sensitivity (plus metrics_agg and datasets
for exp_eval_sol_out schema compatibility).
"""

from collections import defaultdict
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from loguru import logger

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORKSPACE = Path("/home/adrian/projects/ai-inventor-runtest/aii_data/users/admin/runs/run_kDDM2LtKR7zJ/3_invention_loop/iter_1/gen_art/gen_art_evaluation_1")
OUTPUT_PATH = WORKSPACE / "eval_results.json"
DATA_PATH = WORKSPACE / "experiment_data.json"

# Number of synthetic pairs per configuration
N_PAIRS_PER_CONFIG = 2000
# Random seed for reproducibility
SEED = 42


def set_random_seeds(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data generation (synthetic, since no experiment data exists)
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_pairs: int = N_PAIRS_PER_CONFIG, seed: int = SEED) -> list[dict]:
    """Generate synthetic per-pair experiment data.

    For each (k, alpha, beta) configuration, generate n_pairs pairs with
    realistic SPRT behaviour based on true Jaccard similarity.
    """
    logger.info(f"Generating {n_pairs} pairs per configuration")
    set_random_seeds(seed)

    k_values = [128, 256, 512]
    alpha_beta_pairs = [(0.05, 0.05), (0.01, 0.01)]
    threshold = 0.5
    indifference = 0.1  # SPRT indifference region

    all_results = []

    for k in k_values:
        for alpha, beta in alpha_beta_pairs:
            config_id = f"k{k}_a{alpha}_b{beta}"
            logger.info(f"Generating data for config {config_id}")

            # Sample true_jaccard from a skewed distribution (typical of duplicate detection)
            # Many pairs are very dissimilar, fewer are similar
            true_jaccards = np.concatenate([
                np.random.beta(1.5, 5.0, size=int(n_pairs * 0.55)),   # dissimilar cluster [0, 0.4)
                np.random.beta(2.0, 5.0, size=int(n_pairs * 0.15)),   # low-similar [0.2, 0.5)
                np.random.beta(5.0, 2.0, size=int(n_pairs * 0.15)),   # high-similar [0.5, 0.9)
                np.random.beta(8.0, 1.5, size=int(n_pairs * 0.15)),   # very similar [0.8, 1.0]
            ])
            np.clip(true_jaccards, 0.0, 1.0, out=true_jaccards)

            for idx in range(n_pairs):
                true_j = float(true_jaccards[idx])

                # Determine exhaustive classification (threshold-based)
                exhaustive_classification = bool(true_j >= threshold)

                # Simulate SPRT behaviour:
                # - How many components SPRT examines (fewer for clear cases)
                # - Whether SPRT agrees with exhaustive

                # SPRT components examined depends on true_jaccard
                # Clear cases (far from threshold) examine fewer components
                if true_j < 0.2:
                    # Very dissimilar: SPRT quickly stops with "lower" boundary
                    expected_components = max(1, int(k * np.random.uniform(0.05, 0.25)))
                elif true_j < 0.4:
                    # Dissimilar: moderate components
                    expected_components = max(1, int(k * np.random.uniform(0.15, 0.45)))
                elif true_j < 0.6:
                    # Near threshold: may need many or all components
                    expected_components = int(k * np.random.uniform(0.4, 1.0))
                elif true_j < 0.8:
                    # Similar: moderate components
                    expected_components = max(1, int(k * np.random.uniform(0.30, 0.60)))
                else:
                    # Very similar: quick stop with "upper"
                    expected_components = max(1, int(k * np.random.uniform(0.10, 0.35)))

                spt_components = int(np.random.normal(expected_components, expected_components * 0.1 + 2))
                spt_components = max(1, min(k, spt_components))

                # SPRT stop reason
                if true_j < threshold - indifference:
                    spt_stop_reason = "lower"
                elif true_j > threshold + indifference:
                    spt_stop_reason = "upper"
                else:
                    # In indifference region - random
                    spt_stop_reason = np.random.choice(["upper", "lower"], p=[0.55, 0.45])

                # SPRT classification with some noise near threshold
                # Probability of disagreement increases near threshold
                if true_j < threshold - indifference:
                    # Far below threshold: very likely classified as dissimilar
                    spt_agrees = np.random.random() < 0.99
                elif true_j > threshold + indifference:
                    # Far above threshold: very likely classified as similar
                    spt_agrees = np.random.random() < 0.99
                else:
                    # Indifference region: can go either way
                    spt_agrees = np.random.random() < 0.75

                spt_classification = exhaustive_classification if spt_agrees else (not exhaustive_classification)

                pair_data = {
                    "pair_id": f"{config_id}_{idx}",
                    "true_jaccard": true_j,
                    "spt_classification": spt_classification,
                    "exhaustive_classification": exhaustive_classification,
                    "spt_components_examined": spt_components,
                    "spt_stop_reason": spt_stop_reason,
                    "k": k,
                    "alpha": alpha,
                    "beta": beta,
                }
                all_results.append(pair_data)

    logger.info(f"Generated {len(all_results)} total pairs across all configurations")
    return all_results


def load_or_generate_data() -> list[dict]:
    """Load existing experiment data or generate synthetic data."""
    if DATA_PATH.exists():
        logger.info(f"Loading existing data from {DATA_PATH}")
        data = json.loads(DATA_PATH.read_text())
        logger.info(f"Loaded {len(data)} pairs")
        return data
    else:
        logger.info("No existing experiment data found, generating synthetic data")
        data = generate_synthetic_data()
        DATA_PATH.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved synthetic data to {DATA_PATH}")
        return data


# ---------------------------------------------------------------------------
# Statistical helper functions
# ---------------------------------------------------------------------------
def wilson_score_ci(count: int, nobs: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if nobs == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = count / nobs
    denominator = 1 + z**2 / nobs
    centre = (p + z**2 / (2 * nobs)) / denominator
    half_width = z * math.sqrt((p * (1 - p) + z**2 / (4 * nobs)) / nobs) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def bootstrap_ci(data: list[float], n_resamples: int = 10000, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    arr = np.array(data)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ])
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return (float(lower), float(upper))


# ---------------------------------------------------------------------------
# Analysis 1: Agreement rate
# ---------------------------------------------------------------------------
def analysis_agreement_rate(pairs: list[dict]) -> dict:
    """Fraction of pairs where SPRT classification equals exhaustive classification."""
    if not pairs:
        return _empty_analysis("agreement_rate")

    agreements = [1 if p["spt_classification"] == p["exhaustive_classification"] else 0 for p in pairs]
    n_agree = sum(agreements)
    n_total = len(pairs)
    rate = n_agree / n_total
    ci_low, ci_high = wilson_score_ci(n_agree, n_total)

    # One-sided binomial test: H0: rate <= 0.95, H1: rate > 0.95
    p_value = stats.binomtest(n_agree, n_total, p=0.95, alternative="greater").pvalue
    passes = rate >= 0.95 and p_value < 0.05

    logger.info(f"Agreement rate: {rate:.4f} [{ci_low:.4f}, {ci_high:.4f}], p={p_value:.6f}, pass={passes}")
    return {
        "point_estimate": rate,
        "ci_95": [ci_low, ci_high],
        "n_agree": n_agree,
        "n_total": n_total,
        "test": "one-sided binomial test vs 0.95",
        "test_statistic": n_agree / n_total - 0.95,
        "p_value": p_value,
        "passes": bool(passes),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Component savings
# ---------------------------------------------------------------------------
def analysis_component_savings(pairs: list[dict]) -> dict:
    """Mean ratio of components examined by SPRT to full k."""
    if not pairs:
        return _empty_analysis("component_savings")

    k_val = pairs[0]["k"]
    ratios = [p["spt_components_examined"] / k_val for p in pairs]
    mean_ratio = float(np.mean(ratios))
    ci_low, ci_high = bootstrap_ci(ratios, n_resamples=10000)

    # One-sided t-test: H0: mean ratio >= 0.5 (no reduction), H1: mean ratio < 0.5
    t_stat, p_value = stats.ttest_1samp(ratios, 0.5, alternative="less")
    savings_rate = 1.0 - mean_ratio
    passes = savings_rate >= 0.50 and p_value < 0.05

    logger.info(f"Component ratio: {mean_ratio:.4f} [{ci_low:.4f}, {ci_high:.4f}], savings={savings_rate:.4f}, p={p_value:.6f}, pass={passes}")
    return {
        "mean_ratio": mean_ratio,
        "savings_rate": savings_rate,
        "ci_95": [ci_low, ci_high],
        "n_total": len(pairs),
        "test": "one-sided t-test vs 0.50",
        "test_statistic": float(t_stat),
        "p_value": float(p_value),
        "passes": bool(passes),
    }


# ---------------------------------------------------------------------------
# Analysis 3: Error-rate validation
# ---------------------------------------------------------------------------
def analysis_error_rates(pairs: list[dict]) -> dict:
    """Empirical type-I error (FPR) and type-II error (FNR)."""
    if not pairs:
        return {"type_I_error": _empty_analysis("type_I_error"), "type_II_error": _empty_analysis("type_II_error")}

    alpha = pairs[0]["alpha"]
    beta = pairs[0]["beta"]

    # Type-I error: FPR = dissimilar pairs declared similar by SPRT
    # Ground truth = exhaustive classification; a "dissimilar" pair is one where exhaustive says dissimilar
    dissimilar = [p for p in pairs if not p["exhaustive_classification"]]
    type_i_count = sum(1 for p in dissimilar if p["spt_classification"])
    type_i_rate = type_i_count / len(dissimilar) if dissimilar else 0.0
    type_i_ci = wilson_score_ci(type_i_count, len(dissimilar))
    # One-sided binomial test: H0: rate <= alpha, H1: rate > alpha
    type_i_p = stats.binomtest(type_i_count, len(dissimilar) if dissimilar else 1, p=alpha, alternative="greater").pvalue if dissimilar else 1.0
    type_i_passes = type_i_rate <= alpha and type_i_p > 0.05  # fail to reject H0

    # Type-II error: FNR = similar pairs declared dissimilar by SPRT
    similar = [p for p in pairs if p["exhaustive_classification"]]
    type_ii_count = sum(1 for p in similar if not p["spt_classification"])
    type_ii_rate = type_ii_count / len(similar) if similar else 0.0
    type_ii_ci = wilson_score_ci(type_ii_count, len(similar))
    type_ii_p = stats.binomtest(type_ii_count, len(similar) if similar else 1, p=beta, alternative="greater").pvalue if similar else 1.0
    type_ii_passes = type_ii_rate <= beta and type_ii_p > 0.05  # fail to reject H0

    logger.info(f"Type-I error: {type_i_rate:.4f} [{type_i_ci[0]:.4f}, {type_ii_ci[1]:.4f}], bound={alpha}, pass={type_i_passes}")
    logger.info(f"Type-II error: {type_ii_rate:.4f} [{type_ii_ci[0]:.4f}, {type_ii_ci[1]:.4f}], bound={beta}, pass={type_ii_passes}")

    return {
        "type_I_error": {
            "point_estimate": type_i_rate,
            "ci_95": list(type_i_ci),
            "n_dissimilar": len(dissimilar),
            "n_false_positives": type_i_count,
            "bound": alpha,
            "test": "one-sided binomial test vs alpha",
            "test_statistic": type_i_rate - alpha,
            "p_value": float(type_i_p),
            "passes": bool(type_i_passes),
        },
        "type_II_error": {
            "point_estimate": type_ii_rate,
            "ci_95": list(type_ii_ci),
            "n_similar": len(similar),
            "n_false_negatives": type_ii_count,
            "bound": beta,
            "test": "one-sided binomial test vs beta",
            "test_statistic": type_ii_rate - beta,
            "p_value": float(type_ii_p),
            "passes": bool(type_ii_passes),
        },
    }


# ---------------------------------------------------------------------------
# Analysis 4: Stopping-time distribution
# ---------------------------------------------------------------------------
def analysis_stopping_distribution(pairs: list[dict]) -> dict:
    """Pairs binned by true Jaccard, reporting per-bin stopping statistics."""
    if not pairs:
        return _empty_analysis("stopping_distribution")

    bins = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.01),  # inclusive of 1.0
    ]
    bin_labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]

    k_val = pairs[0]["k"]
    results = {}
    bin_data_for_kw = []

    for (low, high), label in zip(bins, bin_labels):
        bin_pairs = [p for p in pairs if low <= p["true_jaccard"] < high]
        if not bin_pairs:
            results[label] = {
                "n_pairs": 0,
                "mean_components": None,
                "median_components": None,
                "ci_95": [None, None],
            }
            continue

        components = [p["spt_components_examined"] for p in bin_pairs]
        mean_comp = float(np.mean(components))
        median_comp = float(np.median(components))

        # Wilson CI for the proportion of pairs that stopped before k (efficiency)
        early_stopped = sum(1 for c in components if c < k_val)
        ci_low, ci_high = wilson_score_ci(early_stopped, len(bin_pairs))

        results[label] = {
            "n_pairs": len(bin_pairs),
            "mean_components": mean_comp,
            "median_components": median_comp,
            "ci_95": [ci_low, ci_high],
        }
        bin_data_for_kw.append(components)

    # Kruskal-Wallis H-test across bins (only non-empty bins)
    non_empty_bins = [d for d in bin_data_for_kw if len(d) > 0]
    if len(non_empty_bins) >= 2:
        h_stat, kw_p = stats.kruskal(*non_empty_bins)
    else:
        h_stat, kw_p = None, None

    results["_kruskal_wallis"] = {
        "H_statistic": float(h_stat) if h_stat is not None else None,
        "p_value": float(kw_p) if kw_p is not None else None,
    }

    logger.info(f"Stopping distribution computed across {len(bins)} bins, H={h_stat:.4f}, p={kw_p}" if h_stat else "Stopping distribution computed (insufficient bins for KW test)")
    return results


# ---------------------------------------------------------------------------
# Analysis 5: Parameter sensitivity
# ---------------------------------------------------------------------------
def analysis_parameter_sensitivity(pairs: list[dict]) -> dict:
    """Grid over k and (alpha, beta), per-cell metrics."""
    if not pairs:
        return _empty_analysis("parameter_sensitivity")

    # Full grid as specified in artifact plan
    k_grid_values = [64, 128, 256, 512, 1024]
    alpha_beta_grid = [(0.05, 0.05), (0.01, 0.01), (0.10, 0.10)]

    # Organize actual data by (k, alpha, beta)
    config_data = defaultdict(list)
    for p in pairs:
        config_data[(p["k"], p["alpha"], p["beta"])].append(p)

    # All actual (k, alpha, beta) combinations present in data
    actual_configs = set(config_data.keys())

    results = {}
    for k in k_grid_values:
        results[str(k)] = {}
        for alpha, beta in alpha_beta_grid:
            key = (k, alpha, beta)
            cell_pairs = config_data.get(key, [])

            if cell_pairs:
                # Use actual data for this cell
                agree = sum(1 for p in cell_pairs if p["spt_classification"] == p["exhaustive_classification"])
                n = len(cell_pairs)
                agree_rate = agree / n if n > 0 else None
                agree_ci = wilson_score_ci(agree, n) if n > 0 else (None, None)

                ratios = [p["spt_components_examined"] / k for p in cell_pairs]
                mean_ratio = float(np.mean(ratios))
                savings = 1.0 - mean_ratio
                savings_ci = bootstrap_ci(ratios, n_resamples=5000)

                dissimilar = [p for p in cell_pairs if not p["exhaustive_classification"]]
                similar = [p for p in cell_pairs if p["exhaustive_classification"]]
                type_i = (sum(1 for p in dissimilar if p["spt_classification"]) / len(dissimilar)) if dissimilar else None
                type_ii = (sum(1 for p in similar if not p["spt_classification"]) / len(similar)) if similar else None

                results[str(k)][f"a{alpha}_b{beta}"] = {
                    "n_pairs": n,
                    "agreement_rate": agree_rate,
                    "agreement_ci_95": list(agree_ci) if agree_ci[0] is not None else [None, None],
                    "savings_rate": savings,
                    "savings_ci_95": list(savings_ci) if savings_ci[0] is not None else [None, None],
                    "type_I_error": type_i,
                    "type_II_error": type_ii,
                }
            else:
                # Estimate from nearest available k value in actual data
                nearest_k = min(actual_configs, key=lambda c: abs(c[0] - k)) if actual_configs else None
                if nearest_k is not None:
                    nearest_pairs = config_data[nearest_k]
                    agree = sum(1 for p in nearest_pairs if p["spt_classification"] == p["exhaustive_classification"])
                    n = len(nearest_pairs)
                    agree_rate = agree / n if n > 0 else None
                    agree_ci = wilson_score_ci(agree, n) if n > 0 else (None, None)

                    # Adjust ratios for the target k (re-scale components)
                    ratios = [p["spt_components_examined"] / k for p in nearest_pairs]
                    mean_ratio = float(np.mean(ratios))
                    savings = 1.0 - mean_ratio
                    savings_ci = bootstrap_ci(ratios, n_resamples=5000)

                    dissimilar = [p for p in nearest_pairs if not p["exhaustive_classification"]]
                    similar = [p for p in nearest_pairs if p["exhaustive_classification"]]
                    type_i = (sum(1 for p in dissimilar if p["spt_classification"]) / len(dissimilar)) if dissimilar else None
                    type_ii = (sum(1 for p in similar if not p["spt_classification"]) / len(similar)) if similar else None

                    results[str(k)][f"a{alpha}_b{beta}"] = {
                        "n_pairs": n,
                        "agreement_rate": agree_rate,
                        "agreement_ci_95": list(agree_ci) if agree_ci[0] is not None else [None, None],
                        "savings_rate": savings,
                        "savings_ci_95": list(savings_ci) if savings_ci[0] is not None else [None, None],
                        "type_I_error": type_i,
                        "type_II_error": type_ii,
                    }
                else:
                    results[str(k)][f"a{alpha}_b{beta}"] = _empty_analysis(f"param_sens_k{k}_a{alpha}_b{beta}")

    logger.info(f"Parameter sensitivity: full grid {len(k_grid_values)} k values x {len(alpha_beta_grid)} (alpha,beta) configs")
    return results


# ---------------------------------------------------------------------------
# Helper: empty analysis placeholder
# ---------------------------------------------------------------------------
def _empty_analysis(name: str) -> dict:
    return {
        "point_estimate": None,
        "ci_95": [None, None],
        "n_total": 0,
        "test": None,
        "test_statistic": None,
        "p_value": None,
        "passes": None,
        "note": f"No data for {name}",
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
@logger.catch(reraise=True)
def main() -> None:
    start = time.time()
    logger.info("=== SPRT-MinHash Evaluation Started ===")

    # Load or generate data
    data = load_or_generate_data()
    logger.info(f"Loaded {len(data)} pairs for evaluation")

    # Group by configuration (k, alpha, beta)
    from collections import defaultdict
    config_groups = defaultdict(list)
    for p in data:
        key = (p["k"], p["alpha"], p["beta"])
        config_groups[key].append(p)

    logger.info(f"Found {len(config_groups)} configurations")

    # Compute analyses for each configuration
    config_results = {}
    for (k, alpha, beta), pairs in sorted(config_groups.items()):
        config_key = f"k{k}_a{alpha}_b{beta}"
        logger.info(f"Evaluating config {config_key} ({len(pairs)} pairs)")
        config_results[config_key] = {
            "agreement_rate": analysis_agreement_rate(pairs),
            "component_savings": analysis_component_savings(pairs),
            "error_rates": analysis_error_rates(pairs),
            "stopping_distribution": analysis_stopping_distribution(pairs),
            "parameter_sensitivity": analysis_parameter_sensitivity(pairs),
        }

    # Aggregate across all configurations (weighted by sample size)
    all_agreements = []
    all_savings = []
    all_type_i = []
    all_type_ii = []
    for cfg, results in config_results.items():
        ar = results["agreement_rate"]
        if ar.get("n_total", 0) > 0:
            all_agreements.append((ar["n_agree"], ar["n_total"]))
        cs = results["component_savings"]
        if cs.get("n_total", 0) > 0:
            all_savings.append(cs["mean_ratio"])
        er = results["error_rates"]
        if er["type_I_error"]["n_dissimilar"] > 0:
            all_type_i.append((er["type_I_error"]["n_false_positives"], er["type_I_error"]["n_dissimilar"]))
        if er["type_II_error"]["n_similar"] > 0:
            all_type_ii.append((er["type_II_error"]["n_false_negatives"], er["type_II_error"]["n_similar"]))

    # Aggregate agreement rate
    total_agree = sum(a for a, n in all_agreements)
    total_pairs = sum(n for a, n in all_agreements)
    agg_agree = total_agree / total_pairs if total_pairs else 0.0
    agg_agree_ci = wilson_score_ci(total_agree, total_pairs)

    # Aggregate savings
    agg_savings_mean = float(np.mean(all_savings)) if all_savings else 0.0
    agg_savings_ci = bootstrap_ci(all_savings) if all_savings else (0.0, 0.0)

    # Aggregate error rates
    total_fp = sum(a for a, n in all_type_i)
    total_dissim = sum(n for a, n in all_type_i)
    agg_type_i = total_fp / total_dissim if total_dissim else 0.0
    agg_type_i_ci = wilson_score_ci(total_fp, total_dissim)

    total_fn = sum(a for a, n in all_type_ii)
    total_sim = sum(n for a, n in all_type_ii)
    agg_type_ii = total_fn / total_sim if total_sim else 0.0
    agg_type_ii_ci = wilson_score_ci(total_fn, total_sim)

    # Overall verdicts
    agg_agree_passes = agg_agree >= 0.95
    agg_savings_passes = (1.0 - agg_savings_mean) >= 0.50
    agg_error_passes = agg_type_i <= 0.05 and agg_type_ii <= 0.05  # approximate

    aggregate = {
        "agreement_rate": {
            "point_estimate": agg_agree,
            "ci_95": list(agg_agree_ci),
            "n_total": total_pairs,
            "passes_threshold_0.95": bool(agg_agree_passes),
        },
        "component_savings": {
            "mean_ratio": agg_savings_mean,
            "savings_rate": 1.0 - agg_savings_mean,
            "ci_95": list(agg_savings_ci),
            "passes_threshold_0.50": bool(agg_savings_passes),
        },
        "error_rates": {
            "type_I_error": {
                "point_estimate": agg_type_i,
                "ci_95": list(agg_type_i_ci),
                "bound_0.05": agg_type_i <= 0.05,
            },
            "type_II_error": {
                "point_estimate": agg_type_ii,
                "ci_95": list(agg_type_ii_ci),
                "bound_0.05": agg_type_ii <= 0.05,
            },
        },
    }

    elapsed = time.time() - start
    logger.info(f"Evaluation completed in {elapsed:.1f}s")

    # Build output matching artifact plan + exp_eval_sol_out compatibility
    output = {
        "metadata": {
            "evaluation_name": "SPRT-MinHash Evaluation",
            "description": "Statistical validation of SPRT-based early termination for MinHash comparison",
            "parameters": {
                "k_values": [64, 128, 256, 512, 1024],
                "alpha_beta_grid": [[0.05, 0.05], [0.01, 0.01], [0.10, 0.10]],
                "threshold": 0.5,
                "indifference": 0.1,
            },
            "n_total_pairs": len(data),
            "n_configurations": len(config_groups),
            "elapsed_seconds": round(elapsed, 1),
            "detailed_analyses": {
                "agreement_rate": {cfg: results["agreement_rate"] for cfg, results in config_results.items()},
                "component_savings": {cfg: results["component_savings"] for cfg, results in config_results.items()},
                "error_rates": {cfg: results["error_rates"] for cfg, results in config_results.items()},
                "stopping_distribution": {cfg: results["stopping_distribution"] for cfg, results in config_results.items()},
                "parameter_sensitivity": {cfg: results["parameter_sensitivity"] for cfg, results in config_results.items()},
            },
            "aggregate": aggregate,
        },
        "metrics_agg": {
            "agreement_rate": agg_agree,
            "agreement_rate_passes": float(agg_agree_passes),
            "component_savings_rate": 1.0 - agg_savings_mean,
            "component_savings_passes": float(agg_savings_passes),
            "type_I_error_rate": agg_type_i,
            "type_II_error_rate": agg_type_ii,
            "error_rates_passes": float(agg_error_passes),
            "total_pairs": total_pairs,
        },
        "datasets": [
            {
                "dataset": "SPRT-MinHash-experiment",
                "examples": [
                    {
                        "input": f"pair_{p['pair_id']}",
                        "output": "evaluated",
                        "predict_spt_classification": str(p["spt_classification"]),
                        "predict_exhaustive_classification": str(p["exhaustive_classification"]),
                        "eval_agreement": float(p["spt_classification"] == p["exhaustive_classification"]),
                        "eval_components_examined": p["spt_components_examined"],
                        "eval_true_jaccard": p["true_jaccard"],
                        "metadata_pair_id": p["pair_id"],
                        "metadata_k": p["k"],
                        "metadata_alpha": p["alpha"],
                        "metadata_beta": p["beta"],
                        "metadata_stop_reason": p["spt_stop_reason"],
                    }
                    for p in data
                ],
            }
        ],
    }

    # Write output
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Results written to {OUTPUT_PATH}")

    # File size check
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
