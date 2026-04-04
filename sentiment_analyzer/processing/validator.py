"""
Validation logic for incoming records.
"""


def validate_record(record: dict) -> bool:
    """
    Validate a normalized record.

    Current rules:
    - Record must have at least one non-empty value
    """
    return any(bool(v) for v in record.values())
