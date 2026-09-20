# S3-Integrated Image Classification Pipeline (PyTorch)

Production-ready, modular image classification system that reads dataset splits directly from Amazon S3 and supports:

- S3-native train/val/test data loading
- Class extraction from folder names (with numeric-prefix cleanup)
- Configurable preprocessing + augmentation
- Baseline CNN and transfer learning (ResNet18/ResNet50)
- Training + validation loops with best-checkpoint saving
- Test evaluation with confusion matrix and per-class metrics
- Single-image inference from local file or S3 URI

---

## Project Structure

```text
data_loader/
  s3_dataset.py
  transforms.py
models/
  baseline_cnn.py
  model_factory.py
training/
  trainer.py
evaluation/
  metrics.py
inference/
  predict.py
utils/
  checkpoint.py
  config.py
  labels.py
  logging_utils.py
  s3_utils.py
  seed.py
config/
  default_config.yaml
main.py
requirements.txt
README.md
```

---

## Dataset Assumptions

S3 layout:

```text
train/<class_folder>/<image>.jpg
val/<class_folder>/<image>.jpg
test/<class_folder>/<image>.jpg
```

Examples of class folders:

- `1_Copper`
- `2_Wire`
- `Aluminum_Cans`

Class names are generated from folder names, stripping numeric prefixes:

- `1_Copper` -> `Copper`
- `2_Wire` -> `Wire`

> If multiple folders collapse to the same cleaned label (for example `1_Copper`, `2_Copper`), they are merged into one class so all matching samples are retained.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure AWS credentials using one of:

- environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)
- AWS profile/role (recommended in production)

---

## Configuration

Edit `config/default_config.yaml`:

- `data.bucket_name`: S3 bucket
- `data.train_prefix`, `data.val_prefix`, `data.test_prefix`: split prefixes
- `model.*`: architecture and transfer learning options
- `training.*`: batch size, epochs, optimizer, device, tensorboard
- `output.*`: checkpoints/logs/metrics paths

Optional cache:

- `data.cache_images: true` stores downloaded S3 objects under `.cache/s3_images/` for faster repeated epochs.
- On a g4dn.xlarge, point `data.cache_dir` at the **local NVMe instance store** (default config now uses `/opt/dlami/nvme/s3_images`) rather than the EBS root volume. The cache is fully reproducible from S3, so it is safe on ephemeral storage, and NVMe is both faster and far roomier than the ~3.9GB free on the root EBS volume. Keep checkpoints/metrics/logs on EBS.

### Using the NVMe instance store

On the AWS Deep Learning AMI the instance store is **already formatted and mounted** at `/opt/dlami/nvme` (verify with `lsblk` / `df -h /opt/dlami/nvme`). Do **not** run `mkfs`/`mount` on `nvme1n1` — that would destroy the existing volume. You only need a writable cache subdirectory:

```bash
# Confirm it's mounted (~116G) and check ownership.
df -h /opt/dlami/nvme
ls -ld /opt/dlami/nvme

# Create the cache dir and give your user write access (one-time per boot).
sudo mkdir -p /opt/dlami/nvme/s3_images
sudo chown -R "$USER":"$USER" /opt/dlami/nvme/s3_images
```

The instance store is **ephemeral** — wiped on stop/terminate (not reboot). The DLAMI re-creates `/opt/dlami/nvme` on boot, but the `s3_images` subdir is gone, so re-run the `mkdir`/`chown` before training after a stop (or add it to user-data). **If `cache_dir` isn't on the mounted volume, the cache silently falls back onto the near-full EBS root** — so create the dir first, or set `data.cache_dir` back to `.cache/s3_images`.

Optional — migrate the existing 2.8GB EBS cache instead of re-downloading:

```bash
rsync -a --remove-source-files ~/IRTrain/.cache/s3_images/ /opt/dlami/nvme/s3_images/
rm -rf ~/IRTrain/.cache/s3_images
```

> "Overflow" note: this relocates the *whole* cache to NVMe rather than spilling EBS into NVMe. Since the cache is reproducible from S3, a single location on the large NVMe volume is simpler and faster than a two-tier overflow scheme, and it removes the disk-fill risk entirely.

---

## Performance tuning (T4 / g4dn.xlarge)

The following `training.*` flags control GPU/throughput optimizations. They ship enabled (tuned for the Tesla T4) but every one can be set to `false` to restore the original fp32, deterministic behavior.

- `mixed_precision` (AMP): runs forward/backward in fp16 via `torch.autocast` + `GradScaler`, using the T4's Tensor Cores. Largest single training speedup and lowers VRAM use. Slightly changes training numerics, so a run may land on marginally different weights/accuracy than an fp32 run. Evaluation/inference stay fp32 so reported metrics for a given checkpoint remain reproducible.
- `channels_last`: stores activations in NHWC memory format, which is faster for convolutions on GPU and numerically equivalent.
- `cudnn_benchmark`: lets cuDNN autotune the fastest kernels for the fixed 224×224 input. **This disables strict run-to-run determinism** (it sets `cudnn.deterministic = False`). Set `false` if you need bit-reproducible training.
- `persistent_workers` / `prefetch_factor`: keep DataLoader workers (and their S3 clients) alive across epochs and prefetch more batches ahead of the GPU. Only take effect when `num_workers > 0`.
- `compile`: experimental `torch.compile`. Off by default — gains on T4/Turing are uncertain and the first step pays a compilation cost; benchmark before relying on it.

These only affect CUDA runs (AMP/channels_last auto-disable on CPU). DataLoader worker count is still governed by `training.num_workers` and the `S3_DATA_MAX_WORKERS` cap.

---

## Accuracy recipe

The defaults now target the best accuracy obtainable within roughly a 2-hour T4 run, leaning on the speed flags above to afford a stronger setup. All are configurable.

- `model.name: resnet50` — larger backbone than ResNet18; higher accuracy ceiling, more compute (AMP keeps it within budget).
- Stronger train augmentation (`preprocessing.*`): `RandomResizedCrop` + horizontal flip + `TrivialAugmentWide` + `RandomErasing`. Improves generalization at some DataLoader CPU cost — dial back with `trivial_augment: false` / `random_erasing_prob: 0.0` if training becomes CPU-bound.
- Standard eval preprocessing: resize shorter side to `resize_size` (256), then center-crop `image_size` (224) — preserves aspect ratio and matches the pretrained backbones, instead of squashing to a square.
- `training.label_smoothing: 0.1` — small accuracy + calibration gain.
- `training.scheduler: cosine` (enabled) — cosine LR decay over `epochs`, better fine-tuning than a flat LR.
- `training.epochs: 20` with `training.early_stopping_patience: 5` — trains longer but stops once validation accuracy plateaus, so you don't pay for dead epochs or overfit. Best checkpoint (by val accuracy) is always what's saved/evaluated.
- `training.class_weighted_loss` — off by default; set `true` if `verify-data` shows imbalanced classes (weights are derived from the train split). Reported test loss stays standard (unweighted) cross-entropy for comparability.

Tuning notes: if a run exceeds your time budget, lower `epochs` (early stopping may already cut it short) or switch `model.name` back to `resnet18`. For more headroom toward accuracy, raise `epochs` and/or `preprocessing.image_size` (e.g. 256, with `resize_size` ~292).

---

## CLI Usage

### 1) Verify S3 class extraction (recommended first)

```bash
python main.py --config config/default_config.yaml verify-data
```

This validates:

- split indexing from S3
- cleaned class mapping
- class distributions per split

### 2) Sanity training on a small subset

```bash
python main.py --config config/default_config.yaml train --sanity-max-samples 128
```

When `--sanity-max-samples` is set, the pipeline automatically uses single-process
data loading (`num_workers=0`) for better stability and faster startup diagnostics.

### 3) Full training + automatic test evaluation

```bash
python main.py --config config/default_config.yaml train
```

If your system warns about excessive DataLoader workers, lower `training.num_workers` (for example `2` or `1`).

Outputs:

- best model checkpoint: `artifacts/checkpoints/best_model.pt`
- test report: `artifacts/metrics/test_metrics.json`
- confusion matrix CSV: `artifacts/metrics/test_confusion_matrix.csv`
- tensorboard logs (if enabled): `artifacts/tensorboard/`

### 4) Evaluate any checkpoint on selected split

```bash
python main.py --config config/default_config.yaml evaluate \
  --checkpoint artifacts/checkpoints/best_model.pt \
  --split test
```

### 5) Inference on one image

Local image:

```bash
python main.py --config config/default_config.yaml infer \
  --checkpoint artifacts/checkpoints/best_model.pt \
  --image /path/to/local/image.jpg
```

S3 image URI:

```bash
python main.py --config config/default_config.yaml infer \
  --checkpoint artifacts/checkpoints/best_model.pt \
  --image s3://your-bucket/test/2_Wire/MV2007_04_2_Wire.jpg
```

---

## Key Engineering Decisions

1. **S3 loading strategy**
   - Dataset indexes keys once via `list_objects_v2` paginator.
   - Images are fetched on-demand in `__getitem__` for memory efficiency.
   - Optional local caching balances cloud throughput and repeated-epoch speed.
   - Streaming reads include retry/backoff for transient network or TLS failures.

2. **Label processing**
   - Labels are folder-derived and cleaned with regex stripping numeric prefixes.
   - Multiple raw folders that map to the same cleaned label are merged into a single class index.

3. **Modeling**
   - Baseline CNN supports quick iteration.
   - ResNet18/50 transfer learning is available for stronger production quality.
   - Backbone freezing is configurable for low-data/fast-finetuning scenarios.

---

## Notes for Production

- Use `num_workers` tuning based on your compute + S3 bandwidth.
- Prefer IAM roles over static AWS credentials.
- For large jobs, enable caching and ensure storage lifecycle management.
- Add experiment tracking (MLflow/W&B) if required by your platform.
