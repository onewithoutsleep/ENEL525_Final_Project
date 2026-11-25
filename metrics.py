"""Lower values are better for all three error measures."""

import numpy as np

from config import VARIANCE_FLOOR


def evaluate_predictions(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted values must have matching, nonempty shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("A model produced nonfinite values; check the data and settings.")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("Volatility cannot be negative.")

    # MAE and MSE use volatility in decimal units.
    error = actual - predicted

    # QLIKE uses VARIANCE, so square the volatility before calculating it.
    actual_variance = np.maximum(actual ** 2, VARIANCE_FLOOR)
    predicted_variance = np.maximum(predicted ** 2, VARIANCE_FLOOR)
    ratio = actual_variance / predicted_variance
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error ** 2)),
        "QLIKE": float(np.mean(ratio - np.log(ratio) - 1)),
    }


def improvement_percent(baseline_qlike, model_qlike):
    # A percentage reduction is undefined when the baseline has zero loss.
    if baseline_qlike <= 1e-15:
        return None
    return 100 * (baseline_qlike - model_qlike) / baseline_qlike
