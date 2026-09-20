"""Command-line entrypoint for S3-backed image classification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from botocore.exceptions import ClientError, NoCredentialsError
from torch import nn
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import S3ImageClassificationDataset, get_eval_transforms, get_train_transforms
from evaluation import evaluate_classifier, save_evaluation_report
from inference import predict_single_image
from models import build_model
from training import Trainer
from utils import load_checkpoint, load_config, set_seed, setup_logger


def _effective_num_workers(configured_workers: int) -> int:
    """Choose a safe worker count for current host resources.

    Caps workers based on CPU affinity when available (best signal inside
    containers), otherwise falls back to os.cpu_count(). Defaults to an upper
    bound of 3 workers (4 vCPUs, leaving one for the main/GPU-feed thread)
    unless overridden via the ``S3_DATA_MAX_WORKERS`` environment variable.
    """
    if configured_workers <= 0:
        return 0

    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except OSError:
            available_cpus = os.cpu_count() or 1
    else:
        available_cpus = os.cpu_count() or 1

    configured_cap = int(os.environ.get("S3_DATA_MAX_WORKERS", "3"))
    recommended_cap = max(1, min(available_cpus, configured_cap))
    return min(configured_workers, recommended_cap)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _build_transforms(config: Dict) -> Tuple:
    preprocess_cfg = config["preprocessing"]
    train_transform = get_train_transforms(
        image_size=preprocess_cfg["image_size"],
        mean=tuple(preprocess_cfg["mean"]),
        std=tuple(preprocess_cfg["std"]),
        rotation_degrees=preprocess_cfg["rotation_degrees"],
        horizontal_flip_prob=preprocess_cfg["horizontal_flip_prob"],
        random_resized_crop=preprocess_cfg.get("random_resized_crop", True),
        trivial_augment=preprocess_cfg.get("trivial_augment", True),
        color_jitter=preprocess_cfg.get("color_jitter", 0.0),
        random_erasing_prob=preprocess_cfg.get("random_erasing_prob", 0.0),
        rrc_scale=preprocess_cfg.get("rrc_scale", (0.65, 1.0)),
    )
    eval_transform = get_eval_transforms(
        image_size=preprocess_cfg["image_size"],
        mean=tuple(preprocess_cfg["mean"]),
        std=tuple(preprocess_cfg["std"]),
        resize_size=preprocess_cfg.get("resize_size"),
    )
    return train_transform, eval_transform


def _build_datasets(
    config: Dict,
    train_transform,
    eval_transform,
    sanity_max_samples: Optional[int] = None,
) -> Tuple[S3ImageClassificationDataset, S3ImageClassificationDataset, S3ImageClassificationDataset]:
    data_cfg = config["data"]
    common_kwargs = {
        "bucket_name": data_cfg["bucket_name"],
        "region_name": data_cfg.get("region_name"),
        "cache_images": data_cfg.get("cache_images", False),
        "cache_dir": data_cfg.get("cache_dir", ".cache/s3_images"),
        "s3_read_retries": data_cfg.get("s3_read_retries", 8),
        "s3_retry_backoff_seconds": data_cfg.get("s3_retry_backoff_seconds", 0.3),
    }

    train_dataset = S3ImageClassificationDataset(
        split_prefix=data_cfg["train_prefix"],
        transform=train_transform,
        max_samples=sanity_max_samples,
        **common_kwargs,
    )
    val_dataset = S3ImageClassificationDataset(
        split_prefix=data_cfg["val_prefix"],
        transform=eval_transform,
        max_samples=sanity_max_samples,
        class_to_idx=train_dataset.class_to_idx,
        **common_kwargs,
    )
    test_dataset = S3ImageClassificationDataset(
        split_prefix=data_cfg["test_prefix"],
        transform=eval_transform,
        max_samples=sanity_max_samples,
        class_to_idx=train_dataset.class_to_idx,
        **common_kwargs,
    )

    return train_dataset, val_dataset, test_dataset


def _build_dataloaders(
    config: Dict,
    train_dataset: S3ImageClassificationDataset,
    val_dataset: S3ImageClassificationDataset,
    test_dataset: S3ImageClassificationDataset,
    force_num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_cfg = config["training"]
    use_cuda = torch.cuda.is_available() and train_cfg["device"] in ("cuda", "auto")
    effective_workers = (
        force_num_workers if force_num_workers is not None else _effective_num_workers(train_cfg["num_workers"])
    )
    common_loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": effective_workers,
        "pin_memory": use_cuda,
    }
    # persistent_workers / prefetch_factor are only valid with worker processes.
    # Keeping workers alive across epochs avoids re-spawning them (and rebuilding
    # each worker's S3 client) every epoch.
    if effective_workers > 0:
        common_loader_kwargs["persistent_workers"] = train_cfg.get("persistent_workers", True)
        common_loader_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 4)

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader, test_loader


def _build_optimizer(config: Dict, model: nn.Module) -> Optimizer:
    train_cfg = config["training"]
    optimizer_name = train_cfg["optimizer"]["name"].lower()
    optimizer_lr = train_cfg["optimizer"]["lr"]
    weight_decay = train_cfg["optimizer"].get("weight_decay", 0.0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adam":
        return Adam(trainable_params, lr=optimizer_lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = train_cfg["optimizer"].get("momentum", 0.9)
        return SGD(trainable_params, lr=optimizer_lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(config: Dict, optimizer: Optimizer) -> Optional[LRScheduler]:
    scheduler_cfg = config["training"].get("scheduler", {})
    if not scheduler_cfg or not scheduler_cfg.get("enabled", False):
        return None

    scheduler_name = scheduler_cfg.get("name", "").lower()
    if scheduler_name == "step":
        return StepLR(optimizer, step_size=scheduler_cfg["step_size"], gamma=scheduler_cfg["gamma"])
    if scheduler_name == "cosine":
        t_max = scheduler_cfg.get("t_max") or config["training"]["epochs"]
        return CosineAnnealingLR(optimizer, T_max=t_max)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def command_verify_data(config: Dict) -> None:
    logger = setup_logger()
    train_transform, eval_transform = _build_transforms(config)
    train_ds, val_ds, test_ds = _build_datasets(
        config=config, train_transform=train_transform, eval_transform=eval_transform
    )

    logger.info("Class extraction verified for train/val/test.")
    logger.info("Total classes: %d", len(train_ds.class_names))
    for idx, class_name in enumerate(train_ds.class_names):
        logger.info("class_idx=%d class_name=%s", idx, class_name)

    logger.info("Train distribution: %s", json.dumps(train_ds.class_distribution(), indent=2))
    logger.info("Val distribution: %s", json.dumps(val_ds.class_distribution(), indent=2))
    logger.info("Test distribution: %s", json.dumps(test_ds.class_distribution(), indent=2))


def command_train(config: Dict, sanity_max_samples: Optional[int]) -> None:
    logger = setup_logger(log_file=config["output"]["log_file"])
    train_cfg = config["training"]
    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", False))
    set_seed(train_cfg["seed"], cudnn_benchmark=cudnn_benchmark)

    device = _resolve_device(config["training"]["device"])
    logger.info("Using device: %s", device)
    effective_workers = _effective_num_workers(config["training"]["num_workers"])
    if effective_workers != config["training"]["num_workers"]:
        logger.warning(
            "Reducing DataLoader workers from %d to %d based on host capacity.",
            config["training"]["num_workers"],
            effective_workers,
        )

    train_transform, eval_transform = _build_transforms(config)
    train_ds, val_ds, test_ds = _build_datasets(
        config=config,
        train_transform=train_transform,
        eval_transform=eval_transform,
        sanity_max_samples=sanity_max_samples,
    )
    force_num_workers: Optional[int] = None
    if sanity_max_samples is not None:
        force_num_workers = 0
        logger.info(
            "Sanity mode detected (--sanity-max-samples=%d): forcing DataLoader num_workers=0 "
            "to avoid worker startup/network contention.",
            sanity_max_samples,
        )
    train_loader, val_loader, test_loader = _build_dataloaders(
        config, train_ds, val_ds, test_ds, force_num_workers=force_num_workers
    )

    model_cfg = config["model"]
    model = build_model(
        model_name=model_cfg["name"],
        num_classes=len(train_ds.class_names),
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )
    model = model.to(device)

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    class_weights = None
    if train_cfg.get("class_weighted_loss", False):
        distribution = train_ds.class_distribution()
        counts = [distribution.get(name, 0) for name in train_ds.class_names]
        total = sum(counts)
        num_classes = len(counts)
        weights = [
            (total / (num_classes * count)) if count > 0 else 0.0 for count in counts
        ]
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        logger.info("Using class-weighted loss: %s", weights)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)

    writer: Optional[SummaryWriter] = None
    if config["training"].get("tensorboard", {}).get("enabled", False):
        tb_dir = config["training"]["tensorboard"]["log_dir"]
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    log_interval_batches = config["training"].get(
        "log_interval_batches",
        config["training"].get("log_every_n_batches", config["training"].get("log_every_n_steps", 10)),
    )
    channels_last = bool(train_cfg.get("channels_last", False))
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=config["output"]["checkpoint_dir"],
        writer=writer,
        log_interval_batches=log_interval_batches,
        logger=logger,
        use_amp=bool(train_cfg.get("mixed_precision", False)),
        channels_last=channels_last,
        compile_model=bool(train_cfg.get("compile", False)),
    )

    train_metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        class_names=train_ds.class_names,
        model_name=model_cfg["name"],
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 0)),
    )
    logger.info("Training complete: %s", train_metrics)

    best_checkpoint = load_checkpoint(str(trainer.best_checkpoint_path), map_location=str(device))
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # Report standard (unweighted, unsmoothed) cross-entropy on the test split so
    # the reported test loss stays comparable across configs.
    test_metrics = evaluate_classifier(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=train_ds.class_names,
        criterion=nn.CrossEntropyLoss(),
        channels_last=channels_last,
    )
    report_paths = save_evaluation_report(
        metrics=test_metrics,
        output_dir=config["output"]["metrics_dir"],
        split_name="test",
        class_names=train_ds.class_names,
    )
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))
    logger.info("Saved test report files: %s", report_paths)

    if writer is not None:
        writer.close()


def _load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device, channels_last: bool = False
) -> Tuple[nn.Module, List[str]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]
    model = build_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, class_names


def command_evaluate(config: Dict, checkpoint_path: str, split: str) -> None:
    logger = setup_logger(log_file=config["output"]["log_file"])
    train_cfg = config["training"]
    device = _resolve_device(train_cfg["device"])
    logger.info("Using device: %s", device)
    if device.type == "cuda" and train_cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    effective_workers = _effective_num_workers(train_cfg["num_workers"])
    if effective_workers != train_cfg["num_workers"]:
        logger.warning(
            "Reducing DataLoader workers from %d to %d based on host capacity.",
            train_cfg["num_workers"],
            effective_workers,
        )

    channels_last = bool(train_cfg.get("channels_last", False))
    _, eval_transform = _build_transforms(config)
    model, class_names = _load_model_from_checkpoint(
        checkpoint_path=checkpoint_path, device=device, channels_last=channels_last
    )
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    data_cfg = config["data"]
    split_to_prefix = {
        "train": data_cfg["train_prefix"],
        "val": data_cfg["val_prefix"],
        "test": data_cfg["test_prefix"],
    }
    dataset = S3ImageClassificationDataset(
        bucket_name=data_cfg["bucket_name"],
        split_prefix=split_to_prefix[split],
        transform=eval_transform,
        region_name=data_cfg.get("region_name"),
        cache_images=data_cfg.get("cache_images", False),
        cache_dir=data_cfg.get("cache_dir", ".cache/s3_images"),
        # Honor the configured retry policy on the eval path too (previously this
        # silently fell back to the dataset defaults of 5 / 0.25s).
        s3_read_retries=data_cfg.get("s3_read_retries", 8),
        s3_retry_backoff_seconds=data_cfg.get("s3_retry_backoff_seconds", 0.3),
        class_to_idx=class_to_idx,
    )
    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "shuffle": False,
        "num_workers": effective_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = train_cfg.get("persistent_workers", True)
        loader_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 4)
    dataloader = DataLoader(dataset, **loader_kwargs)

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_classifier(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        criterion=criterion,
        channels_last=channels_last,
    )
    report_paths = save_evaluation_report(
        metrics=metrics,
        output_dir=config["output"]["metrics_dir"],
        split_name=split,
        class_names=class_names,
    )
    logger.info("Evaluation metrics (%s): %s", split, json.dumps(metrics, indent=2))
    logger.info("Saved report files: %s", report_paths)


def command_infer(config: Dict, checkpoint_path: str, image_path: str) -> None:
    logger = setup_logger(log_file=config["output"]["log_file"])
    train_cfg = config["training"]
    device = _resolve_device(train_cfg["device"])
    channels_last = bool(train_cfg.get("channels_last", False))

    model, class_names = _load_model_from_checkpoint(
        checkpoint_path=checkpoint_path, device=device, channels_last=channels_last
    )
    _, eval_transform = _build_transforms(config)
    prediction = predict_single_image(
        model=model,
        image_source=image_path,
        class_names=class_names,
        transform=eval_transform,
        device=device,
        default_bucket=config["data"]["bucket_name"],
        region_name=config["data"].get("region_name"),
        channels_last=channels_last,
    )
    logger.info("Prediction: %s", json.dumps(prediction, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S3 image classification pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-data", help="Index S3 splits and verify class extraction.")

    train_parser = subparsers.add_parser("train", help="Train model and evaluate on test split.")
    train_parser.add_argument(
        "--sanity-max-samples",
        type=int,
        default=None,
        help="Optional cap per split for quick sanity checks.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on selected split.")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best_model.pt",
        help="Path to model checkpoint.",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )

    infer_parser = subparsers.add_parser("infer", help="Run single-image inference from local/S3 path.")
    infer_parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best_model.pt")
    infer_parser.add_argument("--image", type=str, required=True, help="Local path or s3:// URI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    try:
        if args.command == "verify-data":
            command_verify_data(config)
        elif args.command == "train":
            command_train(config=config, sanity_max_samples=args.sanity_max_samples)
        elif args.command == "evaluate":
            command_evaluate(config=config, checkpoint_path=args.checkpoint, split=args.split)
        elif args.command == "infer":
            command_infer(config=config, checkpoint_path=args.checkpoint, image_path=args.image)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except NoCredentialsError as error:
        raise SystemExit(
            "AWS credentials were not found. Configure IAM role, AWS profile, or environment "
            "variables before running S3-backed commands."
        ) from error
    except ClientError as error:
        raise SystemExit(f"AWS S3 request failed: {error}") from error


if __name__ == "__main__":
    main()
