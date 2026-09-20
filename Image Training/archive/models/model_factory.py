"""Model factory to swap architectures with config-driven selection."""

from __future__ import annotations

from typing import Literal

from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from .baseline_cnn import BaselineCNN

ModelName = Literal["baseline_cnn", "resnet18", "resnet50"]


def _freeze_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def build_model(
    model_name: ModelName,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create a classification model with requested architecture."""

    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        _freeze_parameters(model)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model
