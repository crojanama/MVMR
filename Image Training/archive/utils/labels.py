"""Label processing utilities for folder-derived class names."""

from __future__ import annotations

import re

_NUMERIC_PREFIX_PATTERN = re.compile(r"^\d+[_\-\s]*")


def clean_class_label(raw_folder_name: str) -> str:
    """Normalize class labels by stripping numeric prefixes.

    Examples:
        - "1_Copper" -> "Copper"
        - "2-Wire" -> "Wire"
        - "12  Aluminum_Cans" -> "Aluminum_Cans"
    """

    label = raw_folder_name.strip().strip("/")
    label = _NUMERIC_PREFIX_PATTERN.sub("", label)
    return label.strip()
