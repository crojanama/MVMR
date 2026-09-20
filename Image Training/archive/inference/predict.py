"""Inference helpers for local and S3 image paths."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch import nn

from utils.s3_utils import build_s3_client, split_s3_path


def _load_image(image_source: str, default_bucket: Optional[str], region_name: Optional[str]) -> Image.Image:
    if image_source.startswith("s3://"):
        bucket, key = split_s3_path(image_source)
        client = build_s3_client(region_name=region_name)
        response = client.get_object(Bucket=bucket, Key=key)
        image_bytes = response["Body"].read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    local_path = Path(image_source)
    if local_path.exists():
        return Image.open(local_path).convert("RGB")

    if default_bucket:
        bucket, key = split_s3_path(image_source, default_bucket=default_bucket)
        client = build_s3_client(region_name=region_name)
        response = client.get_object(Bucket=bucket, Key=key)
        image_bytes = response["Body"].read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    raise FileNotFoundError(f"Image path not found locally and is not an S3 URI: {image_source}")


@torch.inference_mode()
def predict_single_image(
    model: nn.Module,
    image_source: str,
    class_names: List[str],
    transform,
    device: torch.device,
    default_bucket: Optional[str] = None,
    region_name: Optional[str] = None,
    channels_last: bool = False,
) -> Dict[str, object]:
    """Predict class and confidence for a single image."""

    model.eval()
    image = _load_image(image_source=image_source, default_bucket=default_bucket, region_name=region_name)
    tensor = transform(image).unsqueeze(0).to(device)
    if channels_last and device.type == "cuda":
        tensor = tensor.to(memory_format=torch.channels_last)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)

    class_index = int(predicted_idx.item())
    return {
        "image_source": image_source,
        "predicted_index": class_index,
        "predicted_class": class_names[class_index],
        "confidence": float(confidence.item()),
    }
