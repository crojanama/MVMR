"""Amazon S3 helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import boto3
from botocore.client import BaseClient
from botocore.config import Config


@dataclass(frozen=True)
class S3Uri:
    """Parsed representation of an S3 URI."""

    bucket: str
    key: str


def build_s3_client(
    region_name: Optional[str] = None,
    connect_timeout: float = 10.0,
    read_timeout: float = 30.0,
    max_pool_connections: int = 16,
) -> BaseClient:
    """Create a boto3 S3 client with optional explicit region.

    Explicit connect/read timeouts ensure a wedged TCP/TLS socket raises a
    (retryable) timeout instead of blocking a DataLoader worker indefinitely —
    which would stall the in-order DataLoader and hang training. Timed-out reads
    surface as ``BotoCoreError`` subclasses, so the dataset's retry + client-rebuild
    logic handles them. Retry behavior itself is unchanged.
    """

    session = boto3.session.Session()
    return session.client(
        "s3",
        region_name=region_name,
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_pool_connections=max_pool_connections,
        ),
    )


def parse_s3_uri(uri: str) -> S3Uri:
    """Parse ``s3://bucket/key`` into bucket and key parts."""

    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URI: {uri}")

    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Not a valid S3 URI: {uri}")
    return S3Uri(bucket=parts[0], key=parts[1])


def split_s3_path(path_or_uri: str, default_bucket: Optional[str] = None) -> Tuple[str, str]:
    """Return ``(bucket, key)`` from either URI or key-style path."""

    if path_or_uri.startswith("s3://"):
        parsed = parse_s3_uri(path_or_uri)
        return parsed.bucket, parsed.key

    if not default_bucket:
        raise ValueError("A default bucket is required when path is not an S3 URI.")
    return default_bucket, path_or_uri.lstrip("/")
