"""
Validation logic for incoming records.
"""


def validate_record(record: dict, text_fields: list[str] | None = None) -> bool:
    """
    Validate a normalized record.

    With ``text_fields``, the record must have text in at least one of
    them (a row with only an id has nothing to analyze). Without, it must
    have at least one non-empty value.
    """
    if text_fields is not None:
        return any(record.get(f) for f in text_fields)
    return any(bool(v) for v in record.values())
