"""
Experiment 1: how well do common drift detectors catch real, controlled
drift in a text embedding distribution?

Setup:
  - reference distribution: 500 AG News documents from category "World"
  - four test distributions, each n=500:
      * control:  500 more "World" docs (no drift)
      * mild:     80% World + 20% Sports
      * moderate: 50% World + 50% Sports
      * severe:   100% Sports

Embedding model: all-MiniLM-L6-v2 (22M params, runs on CPU).

Detectors compared:
  1. per-dim KS with Bonferroni correction (cheap, common)
  2. MMD with RBF kernel + permutation test (kernel two-sample test)
  3. classifier two-sample test - train logreg to distinguish ref vs test;
     accuracy > 0.5 + epsilon implies drift (Lopez-Paz & Oquab 2017)

For each (detector, test distribution) pair we report:
  - p-value or detection score
  - whether drift was flagged at alpha=0.01
  - wall-clock time

Outputs:
  results.json
  detection_table.png
  timing.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from scipy.stats import ks_2samp
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

OUT_DIR = Path(__file__).parent
RANDOM_SEED = 42

REF_CATEGORY = 1   # AG News: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
DRIFT_CATEGORY = 2  # We use "Business" as the drift target so it's a different topic

# AG News labels: {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def sample_docs(ds, label: int, n: int, rng: np.random.Generator) -> list[str]:
    pool = [r["text"] for r in ds if r["label"] == label]
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def build_distributions(rng: np.random.Generator) -> dict[str, list[str]]:
    print("loading AG News (test split, ~7600 examples)...")
    ds = load_dataset("ag_news", split="test")
    n = 500

    reference = sample_docs(ds, REF_CATEGORY, n, rng)
    control = sample_docs(ds, REF_CATEGORY, n, rng)
    mild_ref = sample_docs(ds, REF_CATEGORY, int(n * 0.8), rng)
    mild_drift = sample_docs(ds, DRIFT_CATEGORY, int(n * 0.2), rng)
    moderate_ref = sample_docs(ds, REF_CATEGORY, int(n * 0.5), rng)
    moderate_drift = sample_docs(ds, DRIFT_CATEGORY, int(n * 0.5), rng)
    severe = sample_docs(ds, DRIFT_CATEGORY, n, rng)

    return {
        "reference": reference,
        "control (100% World)": control,
        "mild (80W/20B)": mild_ref + mild_drift,
        "moderate (50W/50B)": moderate_ref + moderate_drift,
        "severe (100% Business)": severe,
    }


def embed_all(distributions: dict[str, list[str]]) -> dict[str, np.ndarray]:
    print("loading sentence-transformer...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    out = {}
    for name, docs in distributions.items():
        print(f"  encoding {name} (n={len(docs)})...")
        out[name] = model.encode(docs, show_progress_bar=False, batch_size=64)
    return out


# -----------------------------------------------------------------------------
# Detectors
# -----------------------------------------------------------------------------

def detector_per_dim_ks(ref: np.ndarray, test: np.ndarray, alpha: float = 0.01):
    """Per-dimension KS with Bonferroni correction across dimensions."""
    t0 = time.perf_counter()
    n_dims = ref.shape[1]
    pvals = np.array([ks_2samp(ref[:, d], test[:, d]).pvalue for d in range(n_dims)])
    bonferroni_threshold = alpha / n_dims
    n_significant = int((pvals < bonferroni_threshold).sum())
    drift_flagged = bool(n_significant > 0)
    return {
        "drift_flagged": drift_flagged,
        "min_pvalue": float(pvals.min()),
        "n_significant_dims": n_significant,
        "n_dims": n_dims,
        "bonferroni_threshold": bonferroni_threshold,
        "elapsed_sec": time.perf_counter() - t0,
    }


def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    sq_dists = (
        np.sum(X * X, axis=1)[:, None]
        + np.sum(Y * Y, axis=1)[None, :]
        - 2 * X @ Y.T
    )
    return np.exp(-gamma * np.maximum(sq_dists, 0))


def detector_mmd(
    ref: np.ndarray,
    test: np.ndarray,
    n_permutations: int = 200,
    alpha: float = 0.01,
    rng: np.random.Generator | None = None,
):
    """MMD with RBF kernel, permutation test for the p-value."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    t0 = time.perf_counter()

    combined = np.vstack([ref, test])
    median_dist = np.median(np.linalg.norm(combined[:200] - combined[200:400], axis=1) + 1e-9)
    gamma = 1.0 / (2 * median_dist ** 2)

    Kxx = _rbf_kernel_matrix(ref, ref, gamma)
    Kyy = _rbf_kernel_matrix(test, test, gamma)
    Kxy = _rbf_kernel_matrix(ref, test, gamma)
    n, m = ref.shape[0], test.shape[0]

    def mmd_squared(Kxx_, Kyy_, Kxy_, n_, m_):
        np.fill_diagonal(Kxx_, 0)
        np.fill_diagonal(Kyy_, 0)
        return (
            Kxx_.sum() / (n_ * (n_ - 1))
            + Kyy_.sum() / (m_ * (m_ - 1))
            - 2 * Kxy_.mean()
        )

    observed = mmd_squared(Kxx.copy(), Kyy.copy(), Kxy.copy(), n, m)

    K_full = _rbf_kernel_matrix(combined, combined, gamma)
    null = []
    for _ in range(n_permutations):
        perm = rng.permutation(n + m)
        idx_x = perm[:n]
        idx_y = perm[n:]
        Kxx_p = K_full[np.ix_(idx_x, idx_x)]
        Kyy_p = K_full[np.ix_(idx_y, idx_y)]
        Kxy_p = K_full[np.ix_(idx_x, idx_y)]
        null.append(mmd_squared(Kxx_p, Kyy_p, Kxy_p, n, m))
    null = np.array(null)
    pvalue = float((null >= observed).mean())

    return {
        "drift_flagged": bool(pvalue < alpha),
        "mmd_squared": float(observed),
        "pvalue": pvalue,
        "n_permutations": n_permutations,
        "elapsed_sec": time.perf_counter() - t0,
    }


def detector_classifier_two_sample(
    ref: np.ndarray,
    test: np.ndarray,
    alpha: float = 0.01,
    n_splits: int = 5,
):
    """Lopez-Paz & Oquab 2017: train classifier to distinguish ref vs test.
    Accuracy > 0.5 (with binomial p-value < alpha) implies distributions differ.
    """
    t0 = time.perf_counter()
    X = np.vstack([ref, test])
    y = np.concatenate([np.zeros(len(ref)), np.ones(len(test))])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    mean_acc = float(np.mean(accs))

    n_total = len(y)
    from scipy.stats import binomtest
    n_correct = int(round(mean_acc * n_total))
    pval = binomtest(n_correct, n_total, p=0.5, alternative="greater").pvalue

    return {
        "drift_flagged": bool(pval < alpha),
        "mean_accuracy": mean_acc,
        "pvalue": float(pval),
        "n_splits": n_splits,
        "elapsed_sec": time.perf_counter() - t0,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    distributions = build_distributions(rng)
    embeddings = embed_all(distributions)

    ref = embeddings["reference"]
    test_names = ["control (100% World)", "mild (80W/20B)",
                  "moderate (50W/50B)", "severe (100% Business)"]

    results: dict = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": int(ref.shape[1]),
        "reference_n": int(ref.shape[0]),
        "test_distributions": {},
    }

    for name in test_names:
        print(f"\n--- {name} (n={len(embeddings[name])}) ---")
        test = embeddings[name]

        ks = detector_per_dim_ks(ref, test)
        mmd = detector_mmd(ref, test, rng=np.random.default_rng(RANDOM_SEED))
        clf = detector_classifier_two_sample(ref, test)

        print(f"  per-dim KS    flagged={ks['drift_flagged']}  "
              f"sig_dims={ks['n_significant_dims']}/{ks['n_dims']}  "
              f"({ks['elapsed_sec']:.2f}s)")
        print(f"  MMD           flagged={mmd['drift_flagged']}  "
              f"mmd^2={mmd['mmd_squared']:.5f}  p={mmd['pvalue']:.4f}  "
              f"({mmd['elapsed_sec']:.2f}s)")
        print(f"  classifier    flagged={clf['drift_flagged']}  "
              f"acc={clf['mean_accuracy']:.3f}  p={clf['pvalue']:.4f}  "
              f"({clf['elapsed_sec']:.2f}s)")

        results["test_distributions"][name] = {
            "per_dim_ks": ks,
            "mmd": mmd,
            "classifier_two_sample": clf,
        }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    plot_detection_table(results, OUT_DIR / "detection_table.png", test_names)
    plot_timing(results, OUT_DIR / "timing.png", test_names)

    print(f"\nwrote: {OUT_DIR/'results.json'}")
    print(f"wrote: {OUT_DIR/'detection_table.png'}")
    print(f"wrote: {OUT_DIR/'timing.png'}")


def plot_detection_table(results, out_path, test_names):
    methods = ["per_dim_ks", "mmd", "classifier_two_sample"]
    method_labels = ["per-dim KS\n(Bonferroni)", "MMD\n(permutation)",
                     "classifier\ntwo-sample"]
    grid = np.zeros((len(methods), len(test_names)))
    for j, t in enumerate(test_names):
        for i, m in enumerate(methods):
            grid[i, j] = 1.0 if results["test_distributions"][t][m]["drift_flagged"] else 0.0

    fig, ax = plt.subplots(figsize=(9, 3.5))
    cmap = plt.get_cmap("RdYlGn_r")
    for i in range(len(methods)):
        for j in range(len(test_names)):
            color = "#cc4b37" if grid[i, j] == 1 else "#dbe7d7"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            label = "FLAG" if grid[i, j] == 1 else "no drift"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
                    fontsize=10, color="white" if grid[i, j] == 1 else "#333")
    ax.set_xlim(0, len(test_names))
    ax.set_ylim(0, len(methods))
    ax.set_xticks([j + 0.5 for j in range(len(test_names))])
    ax.set_xticklabels([t.replace(" (", "\n(") for t in test_names], fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(methods))])
    ax.set_yticklabels(method_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Drift detection on AG News (alpha=0.01)\n"
                 "control should NOT flag; severe should ALWAYS flag")
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_timing(results, out_path, test_names):
    methods = ["per_dim_ks", "mmd", "classifier_two_sample"]
    method_labels = ["per-dim KS", "MMD (200 perms)", "classifier two-sample"]
    avg_times = []
    for m in methods:
        times = [results["test_distributions"][t][m]["elapsed_sec"] for t in test_names]
        avg_times.append(np.mean(times))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(method_labels, avg_times, color=["#3b6fa3", "#cc4b37", "#7a5fa3"])
    ax.set_xlabel("seconds (avg across 4 test distributions, n=500 each)")
    ax.set_title("Detector wall-clock cost per drift check")
    for bar, val in zip(bars, avg_times):
        ax.text(val * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s", va="center")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
