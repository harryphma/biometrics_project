"""HW3 evaluation adapter with validation threshold selection and fixed-threshold metrics."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from vendor.hw3_original import Evaluator


def evaluator(y, s, title="", num_thresholds=500):
    """Validate trial labels and scores, then construct the bundled HW3 evaluator."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Need genuine (1) and impostor (0) trials")
    if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError("Similarity scores must be finite in [0,1]")
    return Evaluator(num_thresholds, s[y == 1], s[y == 0], title)


def eer(y, s, num_thresholds=500):
    """Return the HW3 finite-grid EER approximation and its corresponding threshold."""
    e = evaluator(y, s, num_thresholds=num_thresholds)
    far, frr, _ = e.compute_rates()
    return tuple(map(float, e.get_EER(far, frr)))


def far_threshold(y, s, target):
    """Select a validation threshold whose empirical FAR does not exceed target.

    Acceptance requires score >= threshold. The threshold is the next floating-point
    value above the first disallowed impostor score, so tied scores are rejected
    together. A target below the available resolution may reject every attempt.
    The experiment driver supplies validation scores for this calculation.
    """
    if not 0 <= target < 1:
        raise ValueError("target FAR must be in [0,1)")
    impostors = np.sort(np.asarray(s)[np.asarray(y) == 0])[::-1]
    if not len(impostors):
        raise ValueError("No validation impostors")
    allowed = int(np.floor(target * len(impostors)))
    return float(np.nextafter(impostors[allowed], np.inf))


def rates_at(y, s, t):
    """Compute confusion counts and error rates at a supplied acceptance threshold."""
    y = np.asarray(y)
    accept = np.asarray(s) >= t
    tp = int(np.sum(accept & (y == 1)))
    fn = int(np.sum(~accept & (y == 1)))
    fp = int(np.sum(accept & (y == 0)))
    tn = int(np.sum(~accept & (y == 0)))
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "FAR": fp / (fp + tn),
        "FRR": fn / (tp + fn),
        "TPR": tp / (tp + fn),
        "TNR": tn / (tn + fp),
        "precision": tp / max(1, tp + fp),
        "F1": 2 * tp / max(1, 2 * tp + fp + fn),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def measure(y, s, thresholds, num_thresholds=500):
    """Combine HW3 curve metrics with error rates at supplied operating thresholds.

    EER and d-prime come from the HW3 evaluator. Fixed-threshold metrics are
    computed separately by rates_at; the experiment driver supplies thresholds
    calibrated on validation data.
    """
    e = evaluator(y, s, num_thresholds=num_thresholds)
    far, frr, tpr = e.compute_rates()
    v, t = e.get_EER(far, frr)
    result = {
        "EER": float(v),
        "test_curve_EER_threshold": float(t),
        "dprime": float(e.get_dprime()),
        "n_genuine": int(np.sum(np.asarray(y) == 1)),
        "n_impostor": int(np.sum(np.asarray(y) == 0)),
    }
    for name, threshold in thresholds.items():
        result[name + "_threshold"] = float(threshold)
        result.update({name + "_" + k: v for k, v in rates_at(y, s, threshold).items()})
    return result, pd.DataFrame(
        {"threshold": e.thresholds, "FAR": far, "FRR": frr, "TPR": tpr}
    )


def plot_hw3(y, s, title, directory, num_thresholds=500):
    """Save HW3 score-distribution, DET, and ROC plots for the supplied trial scores."""
    p = Path(directory).resolve()
    p.mkdir(parents=True, exist_ok=True)
    e = evaluator(y, s, title, num_thresholds)
    far, frr, tpr = e.compute_rates()
    old = Path.cwd()
    try:
        os.chdir(p)
        e.plot_score_distribution()
        e.plot_det_curve(far, frr)
        e.plot_roc_curve(far, tpr)
    finally:
        os.chdir(old)
