"""
Input normalization utilities.

Ensures all records conform to a predictable structure
before sentiment analysis.
"""


def normalize_record(record: dict) -> dict:
    """
    Normalize a single record.

    - Converts all values to strings where applicable
    - Strips whitespace
    """
    normalized = {}

    for k, v in record.items():
        if v is None:
            normalized[k] = ""
        else:
            normalized[k] = str(v).strip()

    return normalized
