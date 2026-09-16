#!/usr/bin/env python3
"""Evaluation script for SPRT MinHash experiment results.

Computes comprehensive metrics per the evaluation plan:
- Ground-truth accuracy (precision, recall, F1 with Wilson 95% CI) for SPRT and
  exhaustive baselines, compared via McNemar's test.
- Agreement with exhaustive comparison with Wilson 95% CI and one-sided binomial
  test against 0.95.
- Component savings (mean, bootstrap 95% CI, one-sided t-test against 0.50).
- Error rate validation (empirical type-I/II with one-sided binomial tests).
- Expected sample size (Wald-Lehmann approximation vs empirical).
- Delta sensitivity by re-applying SPRT with varying indifference widths.
- Parameter sensitivity grid from measured configurations.
- Stopping-time distribution by true Jaccard range with Kruskal-Wallis test.
- Composite-hypothesis framing qualification.
"""

from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from loguru import logger

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
NUM_CPUS = max(1, os.cpu_count() or 1)
THRESHOLD = 0.5
INDIFERENCE = 0.1
ALPHA = 0.05
BETA = 0.05

# Paths - experiment data lives in iter_1 gen_art_experiment_1
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = (
    SCRIPT_DIR.parent.parent.parent
    / "iter_1"
    / "gen_art"
    / "gen_art_experiment_1"
)
FULL_OUTPUT = EXPERIMENT_DIR / "full_method_out.json"
METHOD_PY = EXPERIMENT_DIR / "method.py"

# Delta values for sensitivity analysis
DELTA_VALUES = [0.05, 0.1, 0.15, 0.2]

# Jaccard bins for stopping-time analysis
JACCARD_BINS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# Bootstrap parameters
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95

# Wilson CI z-value for 95% CI
Z_95 = 1.96


# ============================================================
# MINHASH & SPRT (duplicated from method.py for self-contained eval)
# ============================================================
class MinHash:
    """MinHash signature generator using character n-grams."""

    def __init__(self, k: int, seed: int = 42, ngram: int = 5):
        self.k = k
        self.ngram = ngram
        self.hash_seeds = [seed + i for i in range(k)]

    def shingle(self, text: str) -> set:
        text = text.lower().strip()
        if len(text) < self.ngram:
            return {hash(text)}
        shingles = set()
        for i in range(len(text) - self.ngram + 1):
            shingles.add(hash(text[i:i + self.ngram]))
        return shingles

    def compute_signature(self, text: str) -> List[int]:
        shingles = self.shingle(text)
        signature = []
        for seed in self.hash_seeds:
            min_hash = float("inf")
            for s in shingles:
                h = hash((s, seed)) & 0xFFFFFFFF
                if h < min_hash:
                    min_hash = h
            signature.append(min_hash)
        return signature


class SPRTComparator:
    """Sequential Probability Ratio Test for MinHash comparison."""

    def __init__(self, threshold: float, alpha: float = 0.05, beta: float = 0.05, indifference: float = 0.1):
        self.p0 = threshold - indifference
        self.p1 = threshold + indifference
        self.alpha = alpha
        self.beta = beta
        self.A = math.log((1 - beta) / alpha)
        self.B = math.log(beta / (1 - alpha))
        self.llr_match = math.log(self.p1 / self.p0) if self.p0 > 0 else float("inf")
        self.llr_mismatch = math.log((1 - self.p1) / (1 - self.p0)) if self.p0 < 1 else float("-inf")

    def compare(self, sig1: List[int], sig2: List[int]) -> Tuple[bool, int, float]:
        cumulative_llr = 0.0
        k = len(sig1)
        for i, (a, b) in enumerate(zip(sig1, sig2)):
            if a == b:
                cumulative_llr += self.llr_match
            else:
                cumulative_llr += self.llr_mismatch
            if cumulative_llr >= self.A:
                return True, i + 1, cumulative_llr
            if cumulative_llr <= self.B:
                return False, i + 1, cumulative_llr
        return cumulative_llr > 0, k, cumulative_llr


def generate_synthetic_corpus(num_docs: int, seed: int = 42) -> List[str]:
    """Generate synthetic text documents (copied from method.py)."""
    random.seed(seed)
    base_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "problem", "solution", "code", "error", "function", "class", "data", "result",
        "file", "system", "user", "application", "server", "database", "network", "security",
    ]
    documents = []
    for doc_id in range(num_docs):
        length = random.randint(50, 500)
        words = [random.choice(base_words) for _ in range(length)]
        documents.append(" ".join(words))
    return documents


def create_duplicate_variant(base_text: str, similarity: float, seed: int) -> str:
    random.seed(seed)
    words = base_text.split()
    num_keep = int(len(words) * similarity)
    num_replace = len(words) - num_keep
    keep_indices = set(random.sample(range(len(words)), num_keep))
    new_words = []
    for i, word in enumerate(words):
        if i in keep_indices:
            new_words.append(word)
        else:
            new_words.append(random.choice([
                "solution", "problem", "code", "error", "function", "class", "data",
                "result", "file", "system", "user", "application", "server", "database",
            ]))
    return " ".join(new_words)


def build_evaluation_pairs(
    documents: List[str], num_duplicate: int, num_nondeplicate: int, seed: int = 42
) -> List[Tuple[int, int, bool, float]]:
    """Build evaluation pairs (copied from method.py)."""
    random.seed(seed)
    pairs = []
    for _ in range(num_duplicate):
        idx1 = random.randint(0, len(documents) - 1)
        base_text = documents[idx1]
        similarity = random.uniform(0.3, 0.8)
        variant = create_duplicate_variant(base_text, similarity, random.randint(0, 10000))
        documents.append(variant)
        idx2 = len(documents) - 1
        pairs.append((idx1, idx2, True, similarity))
    for _ in range(num_nondeplicate):
        idx1 = random.randint(0, len(documents) - 1)
        idx2 = random.randint(0, len(documents) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(documents) - 1)
        sim = 0.0  # placeholder; we'll compute from signatures
        pairs.append((idx1, idx2, False, sim))
    random.shuffle(pairs)
    return pairs


# ============================================================
# JSON SERIALIZATION HELPERS
# ============================================================
class NumpyAwareEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return super().default(obj)


def json_dumps(obj, indent=2):
    return json.dumps(obj, indent=indent, cls=NumpyAwareEncoder)


# ============================================================
# STATISTICAL HELPERS
# ============================================================
def wilson_ci(prop: float, n: int, z: float = Z_95) -> Tuple[float, float]:
    z2 = z * z
    center = prop + z2 / (2 * n)
    half = z * math.sqrt((prop * (1 - prop) + z2 / (4 * n)) / n)
    denom = 1 + z2 / n
    return (max(0.0, (center - half) / denom), min(1.0, (center + half) / denom))


def one_sided_binomial_test(k: int, n: int, p0: float, alternative: str = "less") -> float:
    """One-sided binomial test: P(X >= k | n, p0) or P(X <= k | n, p0)."""
    if n == 0:
        return 1.0
    if alternative == "less":
        return stats.binom.cdf(k, n, p0)
    elif alternative == "greater":
        return 1.0 - stats.binom.cdf(k - 1, n, p0)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")


def McNemar_test(y_true: List[bool], y_pred1: List[bool], y_pred2: List[bool]) -> Tuple[float, float]:
    """McNemar's test: discordant pairs between two classifiers.
    Returns (chi2_statistic, p_value).
    """
    n01 = sum(1 for a, b in zip(y_pred1, y_pred2) if a is False and b is True)
    n10 = sum(1 for a, b in zip(y_pred1, y_pred2) if a is True and b is False)
    if n01 + n10 == 0:
        return (0.0, 1.0)
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = stats.chi2.sf(chi2, df=1)
    return (chi2, p_value)


def bootstrap_ci(values: List[float], n_resamples: int = BOOTSTRAP_N, ci: float = BOOTSTRAP_CI) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values)
    rng = np.random.default_rng(SEED)
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)])
    lower = np.percentile(boot_means, (1 - ci) * 100 / 2)
    upper = np.percentile(boot_means, 100 - (1 - ci) * 100 / 2)
    return (float(lower), float(upper))


# ============================================================
# DATA LOADING
# ============================================================
def load_experiment_data(path: Path) -> List[Dict[str, Any]]:
    """Load experiment output and return list of example dicts."""
    logger.info(f"Loading experiment data from {path}")
    raw = json.loads(path.read_text())
    examples = []
    for ds in raw.get("datasets", []):
        for ex in ds["examples"]:
            out = json.loads(ex["output"]) if isinstance(ex["output"], str) else ex["output"]
            # Parse pair indices from input field: "Pair (idx1, idx2): ..."
            input_str = ex.get("input", "")
            pair_match = None
            try:
                import re
                pair_match = re.search(r"Pair\s*\((\d+),\s*(\d+)\)", input_str)
            except Exception:
                pass
            example = {
                **out,
                "predict_exhaustive": ex.get("predict_exhaustive", ""),
                "predict_sprt": ex.get("predict_sprt", ""),
                "input": input_str,
                "pair_indices": (int(pair_match.group(1)), int(pair_match.group(2))) if pair_match else None,
            }
            examples.append(example)
    logger.info(f"Loaded {len(examples)} examples")
    return examples


# ============================================================
# METRIC COMPUTATIONS
# ============================================================
def compute_ground_truth_metrics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute precision/recall/F1 for SPRT and exhaustive with Wilson CIs and McNemar's test."""
    logger.info("Computing ground-truth accuracy metrics")

    def _metrics_for(predictor: str) -> Dict[str, Any]:
        pred_field = f"predict_{predictor}"
        preds = [e[pred_field].lower() == "true" for e in examples]
        truths = [e["true_label"] for e in examples]

        tp = sum(1 for p, t in zip(preds, truths) if p and t)
        tn = sum(1 for p, t in zip(preds, truths) if not p and not t)
        fp = sum(1 for p, t in zip(preds, truths) if p and not t)
        fn = sum(1 for p, t in zip(preds, truths) if not p and t)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(precision, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
        ci_lo_r, ci_hi_r = wilson_ci(recall, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
        ci_lo_f1, ci_hi_f1 = wilson_ci(f1, len(preds)) if len(preds) > 0 else (0.0, 0.0)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "precision_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "recall_ci_95": [round(ci_lo_r, 4), round(ci_hi_r, 4)],
            "f1_ci_95": [round(ci_lo_f1, 4), round(ci_hi_f1, 4)],
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    sprt_metrics = _metrics_for("sprt")
    exh_metrics = _metrics_for("exhaustive")

    # McNemar's test between SPRT and exhaustive
    sprt_preds = [e["predict_sprt"].lower() == "true" for e in examples]
    exh_preds = [e["predict_exhaustive"].lower() == "true" for e in examples]
    chi2, pval = McNemar_test(truths := [e["true_label"] for e in examples], sprt_preds, exh_preds)

    return {
        "sprt": sprt_metrics,
        "exhaustive": exh_metrics,
        "mcnemar": {
            "chi2": round(chi2, 4),
            "p_value": round(pval, 6),
        },
    }


def compute_agreement_metrics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agreement with exhaustive comparison: proportion, Wilson CI, one-sided binomial test vs 0.95."""
    logger.info("Computing agreement metrics")
    match_count = sum(
        1 for e in examples
        if e["predict_sprt"].lower() == e["predict_exhaustive"].lower()
    )
    n = len(examples)
    prop = match_count / n if n > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(prop, n)
    # One-sided binomial test: H1: proportion >= 0.95
    p_value = one_sided_binomial_test(match_count, n, 0.95, alternative="greater")
    return {
        "agreement_proportion": round(prop, 4),
        "agreement_count": match_count,
        "n": n,
        "agreement_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "binomial_test_p_value_vs_0.95": round(p_value, 6),
    }


def compute_component_savings(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Component savings: mean, bootstrap CI, one-sided t-test against 0.50, per k."""
    logger.info("Computing component savings")
    by_k = defaultdict(list)
    for e in examples:
        k = e["k"]
        savings = 1.0 - e["components_used"] / k if k > 0 else 0.0
        by_k[k].append(savings)

    results = {}
    for k in sorted(by_k.keys()):
        vals = by_k[k]
        mean_sav = np.mean(vals)
        ci_lo, ci_hi = bootstrap_ci(vals)
        # One-sided t-test: H0: mean <= 0.50 vs H1: mean > 0.50
        t_stat, p_value = stats.ttest_1samp(vals, 0.50, alternative="greater")
        results[str(k)] = {
            "mean_savings": round(float(mean_sav), 4),
            "savings_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "t_test_statistic": round(float(t_stat), 4),
            "t_test_p_value": round(float(p_value), 6),
            "n": len(vals),
        }
    return results


def compute_error_rate_validation(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Empirical type-I/II error rates tested against alpha/beta bounds.

    Type-I: SPRT declares similar but true J < threshold (FPR).
    Type-II: SPRT declares dissimilar but true J >= threshold (FNR).
    """
    logger.info("Computing error rate validation")

    # Type-I: declared similar but true J < threshold
    type_i_pairs = [e for e in examples if e["sprt_pred"] and e["true_jaccard"] < THRESHOLD]
    type_i_total = len(type_i_pairs)
    type_i_rate = type_i_total / len(examples) if len(examples) > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(type_i_rate, len(examples)) if len(examples) > 0 else (0.0, 0.0)
    # One-sided binomial test: is type-I rate < alpha?
    type_i_rej = sum(1 for e in type_i_pairs if e["true_label"])  # false positives among these
    type_i_pvalue = one_sided_binomial_test(type_i_rej, type_i_total, ALPHA, alternative="less") if type_i_total > 0 else 1.0

    # Type-II: declared dissimilar but true J >= threshold
    type_ii_pairs = [e for e in examples if not e["sprt_pred"] and e["true_jaccard"] >= THRESHOLD]
    type_ii_total = len(type_ii_pairs)
    type_ii_rate = type_ii_total / len(examples) if len(examples) > 0 else 0.0
    ci_lo_ii, ci_hi_ii = wilson_ci(type_ii_rate, len(examples)) if len(examples) > 0 else (0.0, 0.0)
    # One-sided binomial test: is type-II rate < beta?
    type_ii_fn = sum(1 for e in type_ii_pairs if not e["true_label"])  # false negatives among these
    type_ii_pvalue = one_sided_binomial_test(type_ii_fn, type_ii_total, BETA, alternative="less") if type_ii_total > 0 else 1.0

    return {
        "type_I_FPR": {
            "rate": round(type_i_rate, 4),
            "count": type_i_total,
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "alpha_bound": ALPHA,
            "one_sided_p_value": round(type_i_pvalue, 6),
            "within_alpha_bound": type_i_pvalue > 0.05,  # fail to reject H0: rate <= alpha
        },
        "type_II_FNR": {
            "rate": round(type_ii_rate, 4),
            "count": type_ii_total,
            "ci_95": [round(ci_lo_ii, 4), round(ci_hi_ii, 4)],
            "beta_bound": BETA,
            "one_sided_p_value": round(type_ii_pvalue, 6),
            "within_beta_bound": type_ii_pvalue > 0.05,
        },
    }


def compute_expected_sample_size(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wald-Lehmann expected sample size vs empirical.

    For each configuration, compute:
    1. Empirical ESS = mean(components_used)
    2. Wald-Lehmann ESS for Bernoulli model
    3. Approximation error and qualitative assessment
    """
    logger.info("Computing expected sample size analysis")

    by_config = defaultdict(list)
    for e in examples:
        key = (e["k"], e["alpha"], e["beta"])
        by_config[key].append(e["components_used"])

    results = {}
    for (k, alpha, beta), comps in sorted(by_config.items()):
        empirical_ess = float(np.mean(comps))
        p0 = THRESHOLD - INDIFERENCE
        p1 = THRESHOLD + INDIFERENCE

        # Per-step LLR expectations
        llr_match = math.log(p1 / p0)
        llr_mismatch = math.log((1 - p1) / (1 - p0))
        mu_0 = p0 * llr_match + (1 - p0) * llr_mismatch  # under H0
        mu_1 = p1 * llr_match + (1 - p1) * llr_mismatch  # under H1

        A = math.log((1 - beta) / alpha)
        B = math.log(beta / (1 - alpha))

        # Wald-Lehmann approximation: weighted average of stopping at both boundaries
        # Under H0: prob stop at A ~ alpha, prob stop at B ~ 1-alpha
        wald_ess_0 = (alpha * A + (1 - alpha) * B) / mu_0
        # Under H1: prob stop at B ~ beta, prob stop at A ~ 1-beta
        wald_ess_1 = (beta * B + (1 - beta) * A) / mu_1

        # Use H0 ESS as the representative (since most pairs are non-duplicate)
        wald_ess = float((wald_ess_0 + wald_ess_1) / 2)  # average of both

        approx_error = abs(wald_ess - empirical_ess) / max(empirical_ess, 1e-10)

        results[f"k={k}_a={alpha}_b={beta}"] = {
            "empirical_ess": round(empirical_ess, 2),
            "wald_lehmann_ess": round(wald_ess, 2),
            "approximation_error": round(approx_error, 4),
            "wald_boundary_note": (
                "Wald approximation assumes fixed boundaries and no overshoot; "
                "boundary crossing may cause empirical ESS to deviate."
            ),
        }
    return results


def _reconstruct_per_component_llr(
    idx1: int, idx2: int, sig1: List[int], sig2: List[int],
    k: int, p0: float, p1: float, max_k: int,
) -> List[float]:
    """Reconstruct per-component LLR sequence for a pair."""
    llr_match = math.log(p1 / p0)
    llr_mismatch = math.log((1 - p1) / (1 - p0))
    seq = []
    for i in range(min(max_k, len(sig1), len(sig2))):
        if sig1[i] == sig2[i]:
            seq.append(llr_match)
        else:
            seq.append(llr_mismatch)
    return seq


def _run_delta_sensitivity_for_config(
    examples: List[Dict[str, Any]],
    sigs_by_doc: Dict[int, List[int]],
    k: int, alpha: float, beta: float,
    corpus_docs: List[str],
    pairs_info: List[Tuple[int, int, bool, float]],
) -> Dict[str, Any]:
    """Run delta sensitivity for one (k, alpha, beta) configuration."""
    # Build lookup from (idx1, idx2) -> example data
    ex_lookup = {}
    for e in examples:
        key = (e.get("pair_indices", (None, None)))
        if key[0] is not None:
            ex_lookup[key] = e

    results_by_delta = {}
    for delta in DELTA_VALUES:
        p0 = THRESHOLD - delta
        p1 = THRESHOLD + delta
        if p0 <= 0 or p1 >= 1:
            results_by_delta[str(delta)] = {"skipped": True, "reason": "p0<=0 or p1>=1"}
            continue

        sprt_preds = []
        components_list = []
        savings_list = []
        true_labels = []
        true_jaccards = []

        for idx1, idx2, true_label, true_j in pairs_info:
            if idx1 not in sigs_by_doc or idx2 not in sigs_by_doc:
                continue
            sig1 = sigs_by_doc[idx1]
            sig2 = sigs_by_doc[idx2]
            max_k = min(len(sig1), k)

            # Per-component LLR sequence
            llr_match = math.log(p1 / p0)
            llr_mismatch = math.log((1 - p1) / (1 - p0))

            # Run SPRT with new delta
            sprt = SPRTComparator(threshold=THRESHOLD, alpha=alpha, beta=beta, indifference=delta)
            sprt_pred, components, _ = sprt.compare(sig1[:max_k], sig2[:max_k])

            sprt_preds.append(sprt_pred)
            components_list.append(components)
            savings_list.append(1.0 - components / k if k > 0 else 0.0)
            true_labels.append(true_label)
            true_jaccards.append(true_j)

        n = len(sprt_preds)
        if n == 0:
            results_by_delta[str(delta)] = {"skipped": True, "reason": "no pairs"}
            continue

        # Agreement with exhaustive
        exh_preds = []
        for idx1, idx2, _, _ in pairs_info:
            if idx1 in sigs_by_doc and idx2 in sigs_by_doc:
                matches = sum(1 for a, b in zip(sigs_by_doc[idx1][:k], sigs_by_doc[idx2][:k]) if a == b)
                exh_preds.append((matches / k) >= THRESHOLD)
            else:
                exh_preds.append(None)
        valid = [(s, e) for s, e in zip(sprt_preds, exh_preds) if e is not None]
        agreement = sum(1 for s, e in valid if s == e) / len(valid) if valid else 0.0
        ci_lo, ci_hi = wilson_ci(agreement, len(valid)) if valid else (0.0, 0.0)

        # Accuracy metrics
        tp = sum(1 for p, t in zip(sprt_preds, true_labels) if p and t)
        tn = sum(1 for p, t in zip(sprt_preds, true_labels) if not p and not t)
        fp = sum(1 for p, t in zip(sprt_preds, true_labels) if p and not t)
        fn = sum(1 for p, t in zip(sprt_preds, true_labels) if not p and t)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Savings
        mean_sav = np.mean(savings_list)
        ci_lo_s, ci_hi_s = bootstrap_ci(savings_list)

        # Error rates (type-I, type-II)
        type_i_rate = sum(1 for s, j in zip(sprt_preds, true_jaccards) if s and j < THRESHOLD) / n
        type_ii_rate = sum(1 for s, j in zip(sprt_preds, true_jaccards) if not s and j >= THRESHOLD) / n

        # ESS
        empirical_ess = float(np.mean(components_list))

        results_by_delta[str(delta)] = {
            "agreement_with_exhaustive": round(agreement, 4),
            "agreement_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_savings": round(float(mean_sav), 4),
            "savings_ci_95": [round(ci_lo_s, 4), round(ci_hi_s, 4)],
            "type_I_rate": round(type_i_rate, 4),
            "type_II_rate": round(type_ii_rate, 4),
            "empirical_ess": round(empirical_ess, 2),
            "n": n,
        }
    return results_by_delta


def compute_delta_sensitivity(
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Re-apply SPRT with different delta values using reconstructed per-component sequences.

    Since per-component LLR sequences are not stored, we reconstruct them:
    1. From components_used and final_llr, infer # matches per pair (deterministic).
    2. Construct a representative sequence with matches uniformly distributed.
    3. Run SPRT with delta in {0.05, 0.1, 0.15, 0.2} on each reconstructed sequence.
    This is an approximation since the true component order is unknown, but the
    match/mismatch counts are exact.
    """
    logger.info("Computing delta sensitivity analysis")
    delta_results = {}
    try:
        llr_match_orig = math.log((THRESHOLD + INDIFERENCE) / (THRESHOLD - INDIFERENCE))
        llr_mismatch_orig = -llr_match_orig

        for e in examples:
            c = e["components_used"]
            final_llr = e["final_llr"]
            e["_matches"] = int(round((final_llr - c * llr_mismatch_orig) / (2 * llr_match_orig)))
            e["_matches"] = max(0, min(c, e["_matches"]))

        for key_ex, exs in _group_by_config(examples).items():
            delta_results[key_ex] = {}
            for delta in DELTA_VALUES:
                p0 = THRESHOLD - delta
                p1 = THRESHOLD + delta
                if p0 <= 0 or p1 >= 1:
                    delta_results[key_ex][str(delta)] = {"skipped": True, "reason": "p0<=0 or p1>=1"}
                    continue

                llr_match = math.log(p1 / p0)
                llr_mismatch = math.log((1 - p1) / (1 - p0))

                sprt_preds = []
                components_list = []
                savings_list = []
                true_labels = []
                true_jaccards = []

                for e in exs:
                    c = e["components_used"]
                    m = e["_matches"]
                    k = e["k"]

                    llr_seq = [llr_match if i < m else llr_mismatch for i in range(c)]

                    cumulative = 0.0
                    A = math.log((1 - e["beta"]) / e["alpha"])
                    B = math.log(e["beta"] / (1 - e["alpha"]))
                    sprt_pred = None
                    components = c
                    for i, d in enumerate(llr_seq):
                        cumulative += d
                        if cumulative >= A:
                            sprt_pred = True
                            components = i + 1
                            break
                        if cumulative <= B:
                            sprt_pred = False
                            components = i + 1
                            break
                    if sprt_pred is None:
                        sprt_pred = cumulative > 0

                    sprt_preds.append(sprt_pred)
                    components_list.append(components)
                    savings_list.append(1.0 - components / k if k > 0 else 0.0)
                    true_labels.append(e["true_label"])
                    true_jaccards.append(e["true_jaccard"])

                n = len(sprt_preds)
                if n == 0:
                    delta_results[key_ex][str(delta)] = {"skipped": True, "reason": "no pairs"}
                    continue

                exh_preds = [e["predict_exhaustive"].lower() == "true" for e in exs]
                valid = [(s, e_) for s, e_ in zip(sprt_preds, exh_preds)]
                agreement = sum(1 for s, e_ in valid if s == e_) / len(valid)
                ci_lo, ci_hi = wilson_ci(agreement, len(valid))

                tp = sum(1 for p, t in zip(sprt_preds, true_labels) if p and t)
                tn = sum(1 for p, t in zip(sprt_preds, true_labels) if not p and not t)
                fp = sum(1 for p, t in zip(sprt_preds, true_labels) if p and not t)
                fn = sum(1 for p, t in zip(sprt_preds, true_labels) if not p and t)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                mean_sav = float(np.mean(savings_list))
                ci_lo_s, ci_hi_s = bootstrap_ci(savings_list)

                type_i_rate = sum(1 for s, j in zip(sprt_preds, true_jaccards) if s and j < THRESHOLD) / n
                type_ii_rate = sum(1 for s, j in zip(sprt_preds, true_jaccards) if not s and j >= THRESHOLD) / n

                empirical_ess = float(np.mean(components_list))

                delta_results[key_ex][str(delta)] = {
                    "agreement_with_exhaustive": round(agreement, 4),
                    "agreement_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "mean_savings": round(mean_sav, 4),
                    "savings_ci_95": [round(ci_lo_s, 4), round(ci_hi_s, 4)],
                    "type_I_rate": round(type_i_rate, 4),
                    "type_II_rate": round(type_ii_rate, 4),
                    "empirical_ess": round(empirical_ess, 2),
                    "n": n,
                }
    except Exception as ex:
        logger.error(f"Delta sensitivity error: {ex}")
        import traceback
        logger.error(traceback.format_exc())
        delta_results = {"error": str(ex)}

    return delta_results


def _group_by_config(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in examples:
        key = f"k={e['k']}_a={e['alpha']}_b={e['beta']}"
        groups.setdefault(key, []).append(e)
    return groups


def compute_parameter_grid(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parameter sensitivity grid from actual measured configurations only."""
    logger.info("Computing parameter sensitivity grid")
    by_config = defaultdict(list)
    for e in examples:
        key = f"k={e['k']}_a={e['alpha']}_b={e['beta']}"
        by_config[key].append(e)

    grid = {}
    for key, exs in sorted(by_config.items()):
        n = len(exs)
        sprt_preds = [e["predict_sprt"].lower() == "true" for e in exs]
        exh_preds = [e["predict_exhaustive"].lower() == "true" for e in exs]
        truths = [e["true_label"] for e in exs]

        tp = sum(1 for p, t in zip(sprt_preds, truths) if p and t)
        fp = sum(1 for p, t in zip(sprt_preds, truths) if p and not t)
        fn = sum(1 for p, t in zip(sprt_preds, truths) if not p and t)
        tn = sum(1 for p, t in zip(sprt_preds, truths) if not p and not t)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n

        comps = [e["components_used"] for e in exs]
        k = exs[0]["k"]
        savings = 1.0 - np.mean(comps) / k

        agreement = sum(1 for a, b in zip(sprt_preds, exh_preds) if a == b) / n

        grid[key] = {
            "n_pairs": n,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "agreement_with_exhaustive": round(agreement, 4),
            "mean_components_used": round(float(np.mean(comps)), 2),
            "mean_savings": round(float(savings), 4),
            "k": k,
            "alpha": exs[0]["alpha"],
            "beta": exs[0]["beta"],
        }
    return grid


def compute_stopping_time_analysis(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stopping-time distribution by true Jaccard range with Kruskal-Wallis test."""
    logger.info("Computing stopping-time analysis by Jaccard range")

    # Bin by true Jaccard
    by_bin = defaultdict(list)
    for e in examples:
        j = e["true_jaccard"]
        bin_idx = np.digitize(j, JACCARD_BINS) - 1
        lo = JACCARD_BINS[max(0, bin_idx)]
        hi = JACCARD_BINS[min(len(JACCARD_BINS) - 1, bin_idx + 1)]
        bin_label = f"[{lo:.2f}, {hi:.2f})"
        by_bin[bin_label].append(e["components_used"] / e["k"] if e["k"] > 0 else 0.0)

    # Per-bin stats
    bin_stats = {}
    all_groups = []
    for bin_label in sorted(by_bin.keys()):
        vals = by_bin[bin_label]
        mean_st = float(np.mean(vals))
        median_st = float(np.median(vals))
        ci_lo, ci_hi = wilson_ci(mean_st, len(vals)) if len(vals) > 0 else (0.0, 0.0)
        bin_stats[bin_label] = {
            "n": len(vals),
            "mean_stopping_ratio": round(mean_st, 4),
            "median_stopping_ratio": round(median_st, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        }
        all_groups.append(np.array(vals))

    # Kruskal-Wallis test
    if len(all_groups) >= 2:
        h_stat, kw_pval = stats.kruskal(*all_groups)
    else:
        h_stat, kw_pval = 0.0, 1.0

    return {
        "bins": bin_stats,
        "kruskal_wallis": {
            "H_statistic": round(float(h_stat), 4),
            "p_value": round(float(kw_pval), 6),
            "n_groups": len(all_groups),
        },
    }


def compute_composite_hypothesis(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Qualify composite-hypothesis framing.

    Compare SPRT expected sample size against threshold-truncation expected sample size (k).
    Note where SPRT is beneficial vs conservative.
    """
    logger.info("Computing composite-hypothesis qualification")

    by_config = defaultdict(list)
    for e in examples:
        key = f"k={e['k']}_a={e['alpha']}_b={e['beta']}"
        by_config[key].append(e)

    results = {}
    for key, exs in sorted(by_config.items()):
        k = exs[0]["k"]
        truncation_ess = float(k)
        sprt_ess = float(np.mean([e["components_used"] for e in exs]))
        savings_over_truncation = 1.0 - sprt_ess / truncation_ess

        # Where SPRT is conservative: declares dissimilar but exhaustive would say similar
        sprt_preds = [e["predict_sprt"].lower() == "true" for e in exs]
        exh_preds = [e["predict_exhaustive"].lower() == "true" for e in exs]
        true_jaccards = [e["true_jaccard"] for e in exs]

        conservative_count = sum(
            1 for sp, ex, j in zip(sprt_preds, exh_preds, true_jaccards)
            if not sp and (ex or j >= THRESHOLD)
        )

        results[key] = {
            "threshold_truncation_ess": round(truncation_ess, 2),
            "sprt_empirical_ess": round(sprt_ess, 2),
            "savings_vs_truncation": round(savings_over_truncation, 4),
            "composite_hypothesis_note": (
                f"SPRT ESS is {sprt_ess:.1f} vs truncation ESS of {k}. "
                f"SPRT saves {savings_over_truncation:.1%} components on average. "
            ),
            "conservative_cases": conservative_count,
            "conservative_fraction": round(conservative_count / len(exs), 4),
            "beneficial": sprt_ess < truncation_ess,
            "framing_qualification": (
                "SPRT's simple-hypothesis framing (H0: J<=T-d, H1: J>=T+d) is beneficial "
                "when savings > 0 and error rates are within bounds; conservative near "
                "the indifference region boundary where true J falls within [T-d, T+d]."
            ),
        }
    return results


# ============================================================
# MAIN
# ============================================================
@logger.catch(reraise=True)
def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    log_path = Path("logs") / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), rotation="30 MB", level="DEBUG")

    logger.info("=" * 60)
    logger.info("SPRT MinHash Evaluation")
    logger.info("=" * 60)

    # Load data
    examples = load_experiment_data(FULL_OUTPUT)

    if not examples:
        logger.error("No examples loaded!")
        return

    results = {}

    # 1. Ground-truth accuracy metrics
    results["ground_truth_metrics"] = compute_ground_truth_metrics(examples)

    # 2. Agreement metrics
    results["agreement_metrics"] = compute_agreement_metrics(examples)

    # 3. Component savings
    results["component_savings"] = compute_component_savings(examples)

    # 4. Error rate validation
    results["error_rate_validation"] = compute_error_rate_validation(examples)

    # 5. Expected sample size
    results["expected_sample_size"] = compute_expected_sample_size(examples)

    # 6. Delta sensitivity (requires recomputing signatures)
    try:
        results["delta_sensitivity"] = compute_delta_sensitivity(examples)
    except Exception as e:
        logger.error(f"Delta sensitivity failed: {e}")
        results["delta_sensitivity"] = {"error": str(e)}

    # 7. Parameter sensitivity grid
    results["parameter_grid"] = compute_parameter_grid(examples)

    # 8. Stopping-time analysis
    results["stopping_time_analysis"] = compute_stopping_time_analysis(examples)

    # 9. Composite hypothesis qualification
    results["composite_hypothesis"] = compute_composite_hypothesis(examples)

    # Build output following exp_eval_sol_out schema
    # Aggregate metrics - include all key results
    metrics_agg = {
        "total_examples": len(examples),
        "configs_evaluated": len(set((e["k"], e["alpha"], e["beta"]) for e in examples)),
        # Ground-truth accuracy metrics
        "ground_truth_sprt_precision": results["ground_truth_metrics"]["sprt"]["precision"],
        "ground_truth_sprt_recall": results["ground_truth_metrics"]["sprt"]["recall"],
        "ground_truth_sprt_f1": results["ground_truth_metrics"]["sprt"]["f1"],
        "ground_truth_sprt_precision_ci_95": results["ground_truth_metrics"]["sprt"]["precision_ci_95"],
        "ground_truth_sprt_recall_ci_95": results["ground_truth_metrics"]["sprt"]["recall_ci_95"],
        "ground_truth_sprt_f1_ci_95": results["ground_truth_metrics"]["sprt"]["f1_ci_95"],
        "ground_truth_exhaustive_precision": results["ground_truth_metrics"]["exhaustive"]["precision"],
        "ground_truth_exhaustive_recall": results["ground_truth_metrics"]["exhaustive"]["recall"],
        "ground_truth_exhaustive_f1": results["ground_truth_metrics"]["exhaustive"]["f1"],
        "ground_truth_exhaustive_precision_ci_95": results["ground_truth_metrics"]["exhaustive"]["precision_ci_95"],
        "ground_truth_exhaustive_recall_ci_95": results["ground_truth_metrics"]["exhaustive"]["recall_ci_95"],
        "ground_truth_exhaustive_f1_ci_95": results["ground_truth_metrics"]["exhaustive"]["f1_ci_95"],
        "mcnemar_chi2": results["ground_truth_metrics"]["mcnemar"]["chi2"],
        "mcnemar_p_value": results["ground_truth_metrics"]["mcnemar"]["p_value"],
        # Agreement metrics
        "agreement_proportion": results["agreement_metrics"]["agreement_proportion"],
        "agreement_count": results["agreement_metrics"]["agreement_count"],
        "agreement_ci_95": results["agreement_metrics"]["agreement_ci_95"],
        "agreement_p_value_vs_0.95": results["agreement_metrics"]["binomial_test_p_value_vs_0.95"],
        # Error rate validation
        "type_I_FPR_rate": results["error_rate_validation"]["type_I_FPR"]["rate"],
        "type_I_FPR_count": results["error_rate_validation"]["type_I_FPR"]["count"],
        "type_I_FPR_ci_95": results["error_rate_validation"]["type_I_FPR"]["ci_95"],
        "type_I_FPR_p_value": results["error_rate_validation"]["type_I_FPR"]["one_sided_p_value"],
        "type_I_within_alpha": results["error_rate_validation"]["type_I_FPR"]["within_alpha_bound"],
        "type_II_FNR_rate": results["error_rate_validation"]["type_II_FNR"]["rate"],
        "type_II_FNR_count": results["error_rate_validation"]["type_II_FNR"]["count"],
        "type_II_FNR_ci_95": results["error_rate_validation"]["type_II_FNR"]["ci_95"],
        "type_II_FNR_p_value": results["error_rate_validation"]["type_II_FNR"]["one_sided_p_value"],
        "type_II_within_beta": results["error_rate_validation"]["type_II_FNR"]["within_beta_bound"],
    }

    output = {
        "metadata": {
            "evaluation_name": "SPRT MinHash Evaluation",
            "description": "Comprehensive evaluation of SPRT MinHash experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "synthetic_stackoverflow",
                "examples": [
                    {
                        "input": e["input"],
                        "output": json.dumps({
                            "true_label": e["true_label"],
                            "true_jaccard": e["true_jaccard"],
                            "sprt_pred": e["sprt_pred"],
                            "exhaustive_pred": e["exhaustive_pred"],
                            "components_used": e["components_used"],
                            "k": e["k"],
                            "alpha": e["alpha"],
                            "beta": e["beta"],
                            "final_llr": e["final_llr"],
                        }),
                        "eval_sprt_precision": results["ground_truth_metrics"]["sprt"]["precision"],
                        "eval_sprt_recall": results["ground_truth_metrics"]["sprt"]["recall"],
                        "eval_sprt_f1": results["ground_truth_metrics"]["sprt"]["f1"],
                    }
                    for e in examples
                ],
            }
        ],
        "detailed_results": results,
    }

    # Save output
    output_path = SCRIPT_DIR / "full_eval_out.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dumps(output))
    logger.info(f"Evaluation results saved to {output_path}")
    logger.info("=" * 60)
    logger.info("Evaluation completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
