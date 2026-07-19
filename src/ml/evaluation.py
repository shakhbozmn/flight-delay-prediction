"""Metric helpers and figure export for held-out evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - non-interactive backend for headless runs

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
)


def metric_bundle(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """Return the standard portfolio metric dictionary.

    PR-AUC uses the average precision formulation, which is robust under
    moderate class imbalance.
    """

    if len(y_true) != len(y_pred) or len(y_true) != len(y_score):
        raise ValueError("y_true, y_pred, and y_score must have equal length")

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(np.trapz(precision_curve, recall_curve))
    pr_auc = max(pr_auc, 0.0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": pr_auc,
    }


def select_best_validation_result(
    results: Mapping[str, Mapping[str, float]],
) -> str:
    """Pick the best candidate name using F1, then PR-AUC, ROC-AUC, name."""

    if not results:
        raise ValueError("results mapping is empty")

    def _sort_key(name: str) -> tuple[float, float, float, str]:
        bundle = results[name]
        return (
            bundle["f1"],
            bundle["pr_auc"],
            bundle["roc_auc"],
            name,
        )

    return max(results.keys(), key=_sort_key)


def save_model_comparison_figure(
    results: Mapping[str, Mapping[str, float]], output_path: Path
) -> Path:
    """Render a five-panel bar chart of validation metrics and save it."""

    if not results:
        raise ValueError("results mapping is empty")

    metrics = ("accuracy", "f1", "precision", "recall", "pr_auc")
    names = list(results.keys())
    values = [[results[name][metric] for name in names] for metric in metrics]

    figure, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=False)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for index, metric in enumerate(metrics):
        axes[index].bar(names, values[index], color=palette[: len(names)])
        axes[index].set_title(metric.upper())
        axes[index].set_ylim(0.0, 1.0)
        axes[index].tick_params(axis="x", rotation=20)
    figure.suptitle("Validation metrics by candidate model")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_confusion_matrix_figure(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
) -> Path:
    """Render and save the 2024 held-out confusion matrix."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")

    matrix = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["No high delay", "High delay"])
    axis.set_yticklabels(["No high delay", "High delay"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("2024 held-out confusion matrix")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path