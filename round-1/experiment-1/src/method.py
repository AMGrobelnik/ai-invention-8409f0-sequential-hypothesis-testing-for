#!/usr/bin/env python3
"""SPRT MinHash Comparison Experiment.

Implements MinHash signature generation, exhaustive baseline comparison,
and SPRT early-stopping comparator for near-duplicate text detection.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any

from loguru import logger

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
NUM_CPUS = max(1, os.cpu_count() or 1)

# Experiment configurations
K_VALUES = [128, 256, 512]
THRESHOLD = 0.5
ALPHA_BETA_CONFIGS = [(0.05, 0.05), (0.01, 0.01)]
INDIFERENCE = 0.1

# Synthetic data settings
NUM_DOCS = 500
NUM_DUPLICATE_PAIRS = 200
NUM_NONDUPLICATE_PAIRS = 200


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class ComparisonResult:
    pair_id: Tuple[int, int]
    true_label: bool
    true_jaccard: float
    exhaustive_pred: bool
    exhaustive_jaccard: float
    sprt_pred: bool
    components_used: int
    k: int
    alpha: float
    beta: float
    final_llr: float


@dataclass
class ExperimentConfig:
    k: int
    threshold: float
    alpha: float
    beta: float
    indifference: float


# ============================================================
# MINHASH SIGNATURE GENERATION
# ============================================================
class MinHash:
    """MinHash signature generator using character n-grams."""

    def __init__(self, k: int, seed: int = 42, ngram: int = 5):
        self.k = k
        self.ngram = ngram
        self.hash_seeds = [seed + i for i in range(k)]

    def shingle(self, text: str) -> Set[int]:
        """Extract character n-grams from text, hash to integers."""
        text = text.lower().strip()
        if len(text) < self.ngram:
            return {hash(text)}
        shingles = set()
        for i in range(len(text) - self.ngram + 1):
            shingle = text[i:i + self.ngram]
            shingles.add(hash(shingle))
        return shingles

    def compute_signature(self, text: str) -> List[int]:
        """Compute MinHash signature of length k."""
        shingles = self.shingle(text)
        signature = []
        for seed in self.hash_seeds:
            min_hash = float('inf')
            for s in shingles:
                h = hash((s, seed)) & 0xFFFFFFFF  # 32-bit
                if h < min_hash:
                    min_hash = h
            signature.append(min_hash)
        return signature

    def compute_signatures_batch(self, texts: List[str]) -> List[List[int]]:
        """Parallel batch computation using ProcessPoolExecutor."""
        logger.info(f"Computing MinHash signatures for {len(texts)} texts with k={self.k}")
        with ProcessPoolExecutor(
            max_workers=NUM_CPUS,
            mp_context=mp.get_context('spawn')
        ) as pool:
            return list(pool.map(self.compute_signature, texts))


# ============================================================
# JACCARD SIMILARITY COMPUTATION
# ============================================================
def compute_true_jaccard(text1: str, text2: str) -> float:
    """Compute true Jaccard similarity between two texts using character n-grams."""
    ngram = 5
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    if len(text1) < ngram and len(text2) < ngram:
        return 1.0 if text1 == text2 else 0.0

    shingles1 = set()
    for i in range(max(0, len(text1) - ngram + 1)):
        shingles1.add(text1[i:i + ngram])

    shingles2 = set()
    for i in range(max(0, len(text2) - ngram + 1)):
        shingles2.add(text2[i:i + ngram])

    if not shingles1 and not shingles2:
        return 1.0
    intersection = len(shingles1 & shingles2)
    union = len(shingles1 | shingles2)
    return intersection / union if union > 0 else 0.0


# ============================================================
# EXHAUSTIVE COMPARISON BASELINE
# ============================================================
def exhaustive_compare(
    sig1: List[int],
    sig2: List[int],
    threshold: float
) -> Tuple[bool, float]:
    """Compare all k components, return (is_similar, estimated_jaccard)."""
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    estimated_jaccard = matches / len(sig1) if sig1 else 0.0
    return estimated_jaccard >= threshold, estimated_jaccard


# ============================================================
# SPRT COMPARISON WITH EARLY STOPPING
# ============================================================
class SPRTComparator:
    """Sequential Probability Ratio Test for MinHash comparison.

    H0: J <= threshold - indifference (dissimilar)
    H1: J >= threshold + indifference (similar)
    Indifference region: [threshold - indifference, threshold + indifference]
    """

    def __init__(self, threshold: float, alpha: float = 0.05, beta: float = 0.05, indifference: float = 0.1):
        self.p0 = threshold - indifference  # H0 boundary
        self.p1 = threshold + indifference  # H1 boundary
        self.alpha = alpha
        self.beta = beta

        # SPRT boundaries (Wald's approximations)
        self.A = math.log((1 - beta) / alpha)   # upper boundary -> accept H1
        self.B = math.log(beta / (1 - alpha))   # lower boundary -> accept H0

        # Pre-compute log-likelihood ratios per observation
        self.llr_match = math.log(self.p1 / self.p0) if self.p0 > 0 else float('inf')
        self.llr_mismatch = math.log((1 - self.p1) / (1 - self.p0)) if self.p0 < 1 else float('-inf')

    def compare(self, sig1: List[int], sig2: List[int]) -> Tuple[bool, int, float]:
        """Sequential comparison with early stopping.
        Returns: (is_similar, components_examined, final_llr)
        """
        cumulative_llr = 0.0
        k = len(sig1)

        for i, (a, b) in enumerate(zip(sig1, sig2)):
            if a == b:
                cumulative_llr += self.llr_match
            else:
                cumulative_llr += self.llr_mismatch

            if cumulative_llr >= self.A:
                return True, i + 1, cumulative_llr      # H1: similar
            if cumulative_llr <= self.B:
                return False, i + 1, cumulative_llr     # H0: dissimilar

        # Exhausted all k components without crossing boundary
        # Default decision based on final LLR sign
        return cumulative_llr > 0, k, cumulative_llr


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
def generate_synthetic_corpus(num_docs: int, seed: int = 42) -> List[str]:
    """Generate synthetic text documents with controlled similarity."""
    random.seed(seed)
    documents = []

    # Base vocabulary
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
        "file", "system", "user", "application", "server", "database", "network", "security"
    ]

    for doc_id in range(num_docs):
        # Generate random length between 50 and 500 words
        length = random.randint(50, 500)
        words = []
        for _ in range(length):
            words.append(random.choice(base_words))
        documents.append(" ".join(words))

    return documents


def create_duplicate_variant(base_text: str, similarity: float, seed: int) -> str:
    """Create a variant of base_text with controlled Jaccard similarity."""
    random.seed(seed)
    words = base_text.split()

    # Number of words to keep unchanged
    num_keep = int(len(words) * similarity)
    num_replace = len(words) - num_keep

    # Select words to keep
    keep_indices = set(random.sample(range(len(words)), num_keep))
    new_words = []
    for i, word in enumerate(words):
        if i in keep_indices:
            new_words.append(word)
        else:
            # Replace with random word
            new_words.append(random.choice([
                "solution", "problem", "code", "error", "function", "class", "data",
                "result", "file", "system", "user", "application", "server", "database"
            ]))

    return " ".join(new_words)


def build_evaluation_pairs(
    documents: List[str],
    num_duplicate: int,
    num_nondeplicate: int,
    seed: int = 42
) -> List[Tuple[int, int, bool, float]]:
    """Build evaluation pairs with ground truth labels."""
    random.seed(seed)
    pairs = []

    # Duplicate pairs: create variants with known similarity
    for _ in range(num_duplicate):
        idx1 = random.randint(0, len(documents) - 1)
        base_text = documents[idx1]
        # Target similarity around threshold + indifference
        similarity = random.uniform(0.3, 0.8)
        variant = create_duplicate_variant(base_text, similarity, random.randint(0, 10000))
        documents.append(variant)
        idx2 = len(documents) - 1
        pairs.append((idx1, idx2, True, similarity))

    # Non-duplicate pairs: random documents
    for _ in range(num_nondeplicate):
        idx1 = random.randint(0, len(documents) - 1)
        idx2 = random.randint(0, len(documents) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(documents) - 1)
        # Estimate similarity
        sim = compute_true_jaccard(documents[idx1], documents[idx2])
        pairs.append((idx1, idx2, False, sim))

    random.shuffle(pairs)
    return pairs


# ============================================================
# EXPERIMENT DRIVER
# ============================================================
@logger.catch(reraise=True)
def run_single_comparison(
    args: Tuple[int, int, bool, float, List[List[int]], ExperimentConfig]
) -> ComparisonResult:
    """Run a single comparison with both exhaustive and SPRT methods."""
    idx1, idx2, true_label, true_jaccard, signatures, config = args

    sig1 = signatures[idx1]
    sig2 = signatures[idx2]

    # Exhaustive baseline
    exh_pred, exh_jaccard = exhaustive_compare(sig1, sig2, config.threshold)

    # SPRT
    sprt = SPRTComparator(
        threshold=config.threshold,
        alpha=config.alpha,
        beta=config.beta,
        indifference=config.indifference
    )
    sprt_pred, components_used, final_llr = sprt.compare(sig1, sig2)

    return ComparisonResult(
        pair_id=(idx1, idx2),
        true_label=true_label,
        true_jaccard=true_jaccard,
        exhaustive_pred=exh_pred,
        exhaustive_jaccard=exh_jaccard,
        sprt_pred=sprt_pred,
        components_used=components_used,
        k=config.k,
        alpha=config.alpha,
        beta=config.beta,
        final_llr=final_llr
    )


def run_experiment(output_path: Path) -> Dict[str, Any]:
    """Main experiment loop."""
    start_time = time.time()
    logger.info("Starting SPRT MinHash experiment")

    # Generate synthetic corpus
    logger.info(f"Generating synthetic corpus with {NUM_DOCS} documents")
    documents = generate_synthetic_corpus(NUM_DOCS, seed=SEED)
    logger.info(f"Generated {len(documents)} documents")

    # Build evaluation pairs
    logger.info(f"Building evaluation pairs: {NUM_DUPLICATE_PAIRS} duplicates, {NUM_NONDUPLICATE_PAIRS} non-duplicates")
    pairs = build_evaluation_pairs(documents, NUM_DUPLICATE_PAIRS, NUM_NONDUPLICATE_PAIRS, seed=SEED)
    logger.info(f"Built {len(pairs)} evaluation pairs")

    all_results: List[ComparisonResult] = []

    # Run experiments for each configuration
    for k in K_VALUES:
        logger.info(f"Processing k={k}")

        # Generate signatures for all documents
        minhash = MinHash(k=k, seed=SEED, ngram=5)
        signatures = minhash.compute_signatures_batch(documents)
        logger.info(f"Generated {len(signatures)} signatures of length {k}")

        for alpha, beta in ALPHA_BETA_CONFIGS:
            logger.info(f"Running config: k={k}, alpha={alpha}, beta={beta}")
            config = ExperimentConfig(
                k=k,
                threshold=THRESHOLD,
                alpha=alpha,
                beta=beta,
                indifference=INDIFERENCE
            )

            # Prepare arguments for parallel processing
            args_list = [
                (idx1, idx2, true_label, true_jaccard, signatures, config)
                for idx1, idx2, true_label, true_jaccard in pairs
            ]

            # Process in parallel
            batch_results = []
            with ProcessPoolExecutor(
                max_workers=NUM_CPUS,
                mp_context=mp.get_context('spawn')
            ) as pool:
                futures = [pool.submit(run_single_comparison, args) for args in args_list]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Comparison failed: {e}")
                        continue

            all_results.extend(batch_results)
            logger.info(
                f"Completed k={k}, alpha={alpha}, beta={beta}: "
                f"{len(batch_results)} results"
            )

    # Compute summary statistics
    logger.info("Computing summary statistics")
    summary = analyze_results(all_results)

    # Prepare output
    output = {
        "metadata": {
            "method_name": "SPRT MinHash",
            "description": "MinHash with Sequential Probability Ratio Test early-stopping for near-duplicate detection",
            "threshold": THRESHOLD,
            "indifference": INDIFERENCE,
            "num_documents": len(documents),
            "num_pairs": len(pairs),
            "runtime_seconds": time.time() - start_time
        },
        "datasets": [
            {
                "dataset": "synthetic_stackoverflow",
                "examples": [
                    {
                        "input": f"Pair ({r.pair_id[0]}, {r.pair_id[1]}): k={r.k}, alpha={r.alpha}, beta={r.beta}",
                        "output": json.dumps({
                            "true_label": r.true_label,
                            "true_jaccard": round(r.true_jaccard, 4),
                            "exhaustive_pred": r.exhaustive_pred,
                            "exhaustive_jaccard": round(r.exhaustive_jaccard, 4),
                            "sprt_pred": r.sprt_pred,
                            "components_used": r.components_used,
                            "k": r.k,
                            "alpha": r.alpha,
                            "beta": r.beta,
                            "final_llr": round(r.final_llr, 4),
                            "savings": round(1.0 - r.components_used / r.k, 4) if r.k > 0 else 0.0
                        }),
                        "predict_exhaustive": str(r.exhaustive_pred).lower(),
                        "predict_sprt": str(r.sprt_pred).lower()
                    }
                    for r in all_results
                ]
            }
        ]
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Results saved to {output_path}")

    return output


def analyze_results(results: List[ComparisonResult]) -> Dict[str, Any]:
    """Compute metrics and generate summary."""
    if not results:
        return {"error": "No results to analyze"}

    # Group by configuration
    configs: Dict[Tuple[int, float, float], List[ComparisonResult]] = {}
    for r in results:
        key = (r.k, r.alpha, r.beta)
        configs.setdefault(key, []).append(r)

    summary = {
        "total_comparisons": len(results),
        "configs": {}
    }

    for (k, alpha, beta), config_results in configs.items():
        # Agreement rate
        agreement = sum(1 for r in config_results if r.exhaustive_pred == r.sprt_pred)
        agreement_rate = agreement / len(config_results) if config_results else 0.0

        # Component savings
        avg_components = sum(r.components_used for r in config_results) / len(config_results)
        savings = 1.0 - avg_components / k if k > 0 else 0.0

        # Error rates using ground truth
        tp = sum(1 for r in config_results if r.true_label and r.sprt_pred)
        tn = sum(1 for r in config_results if not r.true_label and not r.sprt_pred)
        fp = sum(1 for r in config_results if not r.true_label and r.sprt_pred)
        fn = sum(1 for r in config_results if r.true_label and not r.sprt_pred)

        accuracy = (tp + tn) / len(config_results) if config_results else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Stopping time distribution
        stopping_times = [r.components_used for r in config_results]
        stopping_time_pcts = [st / k for st in stopping_times] if k > 0 else []
        avg_stopping_pct = sum(stopping_time_pcts) / len(stopping_time_pcts) if stopping_time_pcts else 0.0

        summary["configs"][f"k={k}_alpha={alpha}_beta={beta}"] = {
            "num_comparisons": len(config_results),
            "agreement_with_exhaustive": round(agreement_rate, 4),
            "avg_components_used": round(avg_components, 2),
            "component_savings": round(savings, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "avg_stopping_time_pct": round(avg_stopping_pct, 4),
            "true_positive_rate": round(recall, 4),
            "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4)
        }

    return summary


# ============================================================
# UNIT TESTS
# ============================================================
def test_sprt_correctness():
    """Unit tests for SPRT implementation."""
    logger.info("Running SPRT correctness tests")

    # Test 1: Basic SPRT with known sequence
    sprt = SPRTComparator(threshold=0.5, alpha=0.05, beta=0.05, indifference=0.1)

    # Test 2: All matches should quickly exceed upper boundary
    sig1 = [1, 2, 3, 4, 5] * 26  # 130 elements
    sig2 = [1, 2, 3, 4, 5] * 26
    pred, components, llr = sprt.compare(sig1, sig2)
    assert pred is True, "All matches should be similar"
    assert components < len(sig1), "Should stop early on all matches"
    logger.info(f"Test 2 passed: all matches -> similar, stopped at {components}/{len(sig1)}")

    # Test 3: All mismatches should quickly exceed lower boundary
    sig1 = [1, 2, 3, 4, 5] * 26
    sig2 = [10, 20, 30, 40, 50] * 26
    pred, components, llr = sprt.compare(sig1, sig2)
    assert pred is False, "All mismatches should be dissimilar"
    assert components < len(sig1), "Should stop early on all mismatches"
    logger.info(f"Test 3 passed: all mismatches -> dissimilar, stopped at {components}/{len(sig1)}")

    # Test 4: Boundary cases - verify LLR computation
    expected_llr_match = math.log(0.6 / 0.4)  # p1=0.6, p0=0.4
    expected_llr_mismatch = math.log(0.4 / 0.6)
    assert abs(sprt.llr_match - expected_llr_match) < 1e-6, "LLR match mismatch"
    assert abs(sprt.llr_mismatch - expected_llr_mismatch) < 1e-6, "LLR mismatch mismatch"
    logger.info("Test 4 passed: LLR computation correct")

    # Test 5: Verify boundaries
    expected_A = math.log(0.95 / 0.05)
    expected_B = math.log(0.05 / 0.95)
    assert abs(sprt.A - expected_A) < 1e-6, "Upper boundary mismatch"
    assert abs(sprt.B - expected_B) < 1e-6, "Lower boundary mismatch"
    logger.info("Test 5 passed: SPRT boundaries correct")

    logger.info("All SPRT correctness tests passed")


def test_minhash():
    """Unit tests for MinHash implementation."""
    logger.info("Running MinHash tests")

    minhash = MinHash(k=128, seed=42, ngram=5)

    # Test 1: Same text should produce same signature
    text = "This is a test document for MinHash similarity detection"
    sig1 = minhash.compute_signature(text)
    sig2 = minhash.compute_signature(text)
    assert sig1 == sig2, "Same text should produce identical signatures"
    logger.info("Test 1 passed: identical texts produce identical signatures")

    # Test 2: Different texts should produce different signatures
    text2 = "This is a completely different document about something else entirely"
    sig3 = minhash.compute_signature(text2)
    assert sig1 != sig3, "Different texts should produce different signatures"
    logger.info("Test 2 passed: different texts produce different signatures")

    # Test 3: Signature length
    assert len(sig1) == 128, f"Signature length should be 128, got {len(sig1)}"
    logger.info("Test 3 passed: signature length correct")

    # Test 4: Empty/short text
    short_sig = minhash.compute_signature("hi")
    assert len(short_sig) == 128, "Short text should still produce full signature"
    logger.info("Test 4 passed: short text handling correct")

    logger.info("All MinHash tests passed")


# ============================================================
# MAIN
# ============================================================
def main():
    """Main entry point."""
    # Set up logging
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time:HH:mm:ss}|{level:<7}|{message}"
    )
    log_path = Path("logs") / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        rotation="30 MB",
        level="DEBUG"
    )

    logger.info("=" * 60)
    logger.info("SPRT MinHash Comparison Experiment")
    logger.info("=" * 60)

    # Run tests
    test_minhash()
    test_sprt_correctness()

    # Run experiment
    output_path = Path("method_out.json")
    results = run_experiment(output_path)

    logger.info("=" * 60)
    logger.info("Experiment completed successfully")
    logger.info(f"Total runtime: {results['metadata']['runtime_seconds']:.2f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
