"""Image transform pipelines."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    mean: Iterable[float] = (0.485, 0.456, 0.406),
    std: Iterable[float] = (0.229, 0.224, 0.225),
    rotation_degrees: int = 12,
    horizontal_flip_prob: float = 0.5,
    random_resized_crop: bool = True,
    trivial_augment: bool = True,
    color_jitter: float = 0.0,
    random_erasing_prob: float = 0.0,
    rrc_scale: Sequence[float] = (0.65, 1.0),
) -> transforms.Compose:
    """Training-time preprocessing + augmentation pipeline.

    Stronger augmentation improves generalization (accuracy) at some CPU cost in
    the DataLoader workers. Every component is toggleable so the speed/accuracy
    balance can be tuned:

    - ``random_resized_crop``: scale/aspect crop instead of a fixed squash-resize.
    - ``trivial_augment``: TrivialAugmentWide, a strong parameter-free policy.
    - ``color_jitter`` / ``random_erasing_prob``: optional extra regularization.

    When ``trivial_augment`` is disabled the pipeline falls back to the original
    flip + fixed-rotation behavior.
    """

    ops = []
    if random_resized_crop:
        ops.append(transforms.RandomResizedCrop(image_size, scale=tuple(rrc_scale)))
    else:
        ops.append(transforms.Resize((image_size, image_size)))

    if horizontal_flip_prob and horizontal_flip_prob > 0:
        ops.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))

    if trivial_augment:
        ops.append(transforms.TrivialAugmentWide())
    elif rotation_degrees and rotation_degrees > 0:
        ops.append(transforms.RandomRotation(degrees=rotation_degrees))

    if color_jitter and color_jitter > 0:
        ops.append(
            transforms.ColorJitter(
                brightness=color_jitter, contrast=color_jitter, saturation=color_jitter
            )
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=tuple(mean), std=tuple(std)))

    if random_erasing_prob and random_erasing_prob > 0:
        ops.append(transforms.RandomErasing(p=random_erasing_prob))

    return transforms.Compose(ops)


def get_eval_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    resize_size: Optional[int] = None,
) -> transforms.Compose:
    """Validation/test-time deterministic preprocessing pipeline.

    Uses the standard ImageNet protocol (resize shorter side, then center crop)
    rather than squashing to a square, which preserves aspect ratio and better
    matches how the pretrained backbones were trained.
    """

    resize = resize_size if resize_size else int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
