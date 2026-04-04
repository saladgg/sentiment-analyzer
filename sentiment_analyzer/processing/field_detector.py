"""
Automatically detects text-bearing fields in records.

This enables the system to work with heterogeneous schemas
without requiring manual field configuration.
"""


def detect_text_fields(records: list[dict]) -> list[str]:
    """
    Identify candidate text fields based on heuristics.

    Current heuristic:
    - Field value is a string
    - String length > threshold

    This is intentionally simple but extensible.
    """
    if not records:
        return []

    sample = records[0]
    text_fields = []

    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 10:
            text_fields.append(key)

    return text_fields
