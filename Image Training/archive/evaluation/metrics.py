"""Evaluation utilities for classification models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    criterion: Optional[nn.Module] = None,
    channels_last: bool = False,
) -> Dict[str, object]:
    """Run model evaluation and return aggregate + per-class metrics.

    ``inference_mode`` is used instead of ``no_grad`` for lower overhead. Evaluation
    runs in full precision so that reported metrics for a given checkpoint stay
    reproducible regardless of training precision.
    """

    model.eval()
    use_channels_last = channels_last and device.type == "cuda"
    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        if criterion is not None:
            total_loss += criterion(outputs, labels).item() * labels.size(0)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("Cannot evaluate on empty dataset.")

    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / total_samples
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = []
    for idx, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_index": idx,
                "class_name": class_name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    result: Dict[str, object] = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "num_samples": total_samples,
    }

    if criterion is not None:
        result["loss"] = float(total_loss / total_samples)

    return result


def save_evaluation_report(
    metrics: Dict[str, object],
    output_dir: str,
    split_name: str,
    class_names: List[str],
) -> Dict[str, str]:
    """Save evaluation outputs as JSON + confusion matrix CSV."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / f"{split_name}_metrics.json"
    cm_path = output_path / f"{split_name}_confusion_matrix.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    confusion_matrix_rows = metrics.get("confusion_matrix", [])
    with cm_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, confusion_matrix_rows):
            writer.writerow([class_name, *row])

    return {
        "metrics_json": str(metrics_path),
        "confusion_matrix_csv": str(cm_path),
    }
