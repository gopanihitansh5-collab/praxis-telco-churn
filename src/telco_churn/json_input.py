"""Strict JSON loading for CLI and API: no duplicate keys or NaN/Infinity."""

import json
from typing import Any


def parse_customer(text: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON number: {value}")

    payload = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(payload, dict):
        raise ValueError("Input must be one JSON object")
    return payload
