"""PyTorch Dataset implementation for image classification data in S3."""

from __future__ import annotations

import io
import socket
import ssl
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from botocore.exceptions import BotoCoreError
from botocore.client import BaseClient
from PIL import Image
from torch.utils.data import Dataset
from urllib3.exceptions import HTTPError, SSLError

from utils.labels import clean_class_label
from utils.s3_utils import build_s3_client

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class S3ImageClassificationDataset(Dataset):
    """Image dataset backed directly by Amazon S3 object storage.

    Directory layout must follow:
        split_prefix/<class_folder>/<image_file>

    Example:
        train/2_Wire/MV2007_04_2_Wire.jpg
    """

    def __init__(
        self,
        bucket_name: str,
        split_prefix: str,
        transform: Optional[Callable] = None,
        region_name: Optional[str] = None,
        s3_client: Optional[BaseClient] = None,
        allowed_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
        cache_images: bool = False,
        cache_dir: str = ".cache/s3_images",
        s3_read_retries: int = 5,
        s3_retry_backoff_seconds: float = 0.25,
        max_samples: Optional[int] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.split_prefix = split_prefix.strip("/").strip()
        self.transform = transform
        self.region_name = region_name
        self._s3_client = s3_client
        self.allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)
        self.cache_images = cache_images
        self.cache_dir = Path(cache_dir)
        self.s3_read_retries = max(1, s3_read_retries)
        self.s3_retry_backoff_seconds = max(0.0, s3_retry_backoff_seconds)
        self._provided_class_to_idx = class_to_idx.copy() if class_to_idx else None

        if self.cache_images:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples: List[Dict[str, object]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.raw_to_clean: Dict[str, str] = {}

        self._build_index(max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        sample = self.samples[index]
        image = self._load_image(sample["s3_key"])  # type: ignore[index]
        if self.transform:
            image = self.transform(image)
        return image, sample["class_idx"]  # type: ignore[return-value]

    def __getstate__(self):
        state = self.__dict__.copy()
        # boto3 clients are not safely pickleable across dataloader workers.
        state["_s3_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def class_names(self) -> List[str]:
        """Ordered class names by class index."""

        return [self.idx_to_class[idx] for idx in sorted(self.idx_to_class)]

    def class_distribution(self) -> Dict[str, int]:
        """Class sample counts for this split."""

        counter = Counter({class_name: 0 for class_name in self.class_names})
        for sample in self.samples:
            class_name = self.idx_to_class[sample["class_idx"]]  # type: ignore[index]
            counter[class_name] += 1
        return dict(counter)

    def _ensure_client(self) -> BaseClient:
        if self._s3_client is None:
            self._s3_client = build_s3_client(region_name=self.region_name)
        return self._s3_client

    def _is_image_key(self, key: str) -> bool:
        return key.lower().endswith(self.allowed_extensions)

    def _class_prefixes(self) -> List[str]:
        client = self._ensure_client()
        paginator = client.get_paginator("list_objects_v2")
        prefixes: List[str] = []
        root_prefix = f"{self.split_prefix}/"

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=root_prefix, Delimiter="/"):
            for common_prefix in page.get("CommonPrefixes", []):
                prefix = common_prefix.get("Prefix")
                if prefix and prefix != root_prefix:
                    prefixes.append(prefix)

        return sorted(prefixes)

    def _image_keys_for_class(self, class_prefix: str) -> List[str]:
        client = self._ensure_client()
        paginator = client.get_paginator("list_objects_v2")
        keys: List[str] = []

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=class_prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key and not key.endswith("/") and self._is_image_key(key):
                    keys.append(key)
        return sorted(keys)

    def _build_index(self, max_samples: Optional[int] = None) -> None:
        class_prefixes = self._class_prefixes()
        if not class_prefixes:
            raise ValueError(
                f"No class prefixes found under s3://{self.bucket_name}/{self.split_prefix}/."
            )

        clean_to_raw: Dict[str, List[str]] = defaultdict(list)
        raw_to_keys: Dict[str, List[str]] = {}

        for class_prefix in class_prefixes:
            raw_name = class_prefix.rstrip("/").split("/")[-1]
            clean_name = clean_class_label(raw_name)
            if not clean_name:
                continue

            image_keys = self._image_keys_for_class(class_prefix)
            if not image_keys:
                continue

            self.raw_to_clean[raw_name] = clean_name
            clean_to_raw[clean_name].append(raw_name)
            raw_to_keys[raw_name] = image_keys

        if self._provided_class_to_idx is not None:
            unknown_labels = [label for label in clean_to_raw.keys() if label not in self._provided_class_to_idx]
            if unknown_labels:
                raise ValueError(
                    "Split contains labels not present in provided class_to_idx mapping: "
                    + ", ".join(sorted(unknown_labels))
                )
            self.class_to_idx = dict(self._provided_class_to_idx)
            self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
            class_names = [name for name, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])]
        else:
            class_names = sorted(clean_to_raw.keys())
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        sample_count = 0
        for clean_name in class_names:
            if clean_name not in clean_to_raw:
                continue
            class_idx = self.class_to_idx[clean_name]
            for raw_name in sorted(clean_to_raw[clean_name]):
                for key in raw_to_keys[raw_name]:
                    self.samples.append(
                        {
                            "s3_key": key,
                            "class_idx": class_idx,
                            "class_name": clean_name,
                            "raw_class_name": raw_name,
                        }
                    )
                    sample_count += 1
                    if max_samples is not None and sample_count >= max_samples:
                        return

    def _cache_path_for_key(self, key: str) -> Path:
        return self.cache_dir / self.bucket_name / key

    def _download_image_bytes(self, key: str) -> bytes:
        """Download object bytes with retry for transient transport failures."""

        last_error: Optional[Exception] = None
        for attempt in range(1, self.s3_read_retries + 1):
            try:
                response = self._ensure_client().get_object(Bucket=self.bucket_name, Key=key)
                body_stream = response["Body"]
                try:
                    return body_stream.read()
                finally:
                    body_stream.close()
            except (HTTPError, SSLError, BotoCoreError, ssl.SSLError, socket.error, OSError) as error:
                last_error = error
                if attempt == self.s3_read_retries:
                    break
                # Reset connection pool/client on retry to avoid stale TLS sockets.
                self._s3_client = None
                # Exponential backoff for flaky TLS/network connections.
                time.sleep(self.s3_retry_backoff_seconds * (2 ** (attempt - 1)))

        raise RuntimeError(
            "Failed to download "
            f"s3://{self.bucket_name}/{key} after {self.s3_read_retries} attempts. "
            "Consider reducing DataLoader workers and/or enabling cache_images."
        ) from last_error

    def _load_image(self, key: str) -> Image.Image:
        if self.cache_images:
            local_path = self._cache_path_for_key(key)
            if local_path.exists():
                with local_path.open("rb") as file:
                    return Image.open(file).convert("RGB")

            local_path.parent.mkdir(parents=True, exist_ok=True)
            body = self._download_image_bytes(key)
            local_path.write_bytes(body)
            return Image.open(io.BytesIO(body)).convert("RGB")

        image_bytes = self._download_image_bytes(key)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
