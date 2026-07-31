# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Strict, reversible representation of historical recipe evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LEGACY_VIEW_ENCODING_SCHEMA = "tuningfork.legacy-current-view.v1"
LEGACY_NONFINITE_TAG = "\u0000tuningfork_legacy_current_view_nonfinite_float"


def encode_legacy_value(value: Any) -> tuple[Any, bool]:
    """Convert historical values to strict JSON without erasing their meaning."""
    if isinstance(value, float):
        if value != value:
            return {LEGACY_NONFINITE_TAG: "nan"}, True
        if value == float("inf"):
            return {LEGACY_NONFINITE_TAG: "+inf"}, True
        if value == float("-inf"):
            return {LEGACY_NONFINITE_TAG: "-inf"}, True
        return value, False
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            encoded_item, item_changed = encode_legacy_value(item)
            encoded[key] = encoded_item
            changed = changed or item_changed
        return encoded, changed
    if isinstance(value, (list, tuple)):
        encoded_items = []
        changed = False
        for item in value:
            encoded_item, item_changed = encode_legacy_value(item)
            encoded_items.append(encoded_item)
            changed = changed or item_changed
        return encoded_items, changed
    return value, False


def legacy_encoding_metadata() -> dict[str, Any]:
    """Describe the tagged legacy-value encoding carried by an attempt."""
    return {
        "schema": LEGACY_VIEW_ENCODING_SCHEMA,
        "kind": "strict-json-tagged-nonfinite-float",
        "tag_key": LEGACY_NONFINITE_TAG,
        "values": ["nan", "+inf", "-inf"],
    }


__all__ = [
    "LEGACY_NONFINITE_TAG",
    "LEGACY_VIEW_ENCODING_SCHEMA",
    "encode_legacy_value",
    "legacy_encoding_metadata",
]
