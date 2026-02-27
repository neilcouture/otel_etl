"""HTTP method bucketing - transforms HTTP methods into read/write categories."""

from typing import Literal

MethodBucket = Literal["read", "write", "other", "unknown"]


READ_METHODS = frozenset(["GET", "HEAD", "OPTIONS", "TRACE"])
WRITE_METHODS = frozenset(["POST", "PUT", "PATCH", "DELETE"])
OTHER_METHODS = frozenset(["CONNECT"])


def bucket_http_method(method: str) -> MethodBucket:
    """Bucket HTTP method into read/write/other category.

    Args:
        method: HTTP method string

    Returns:
        Method bucket category
    """
    if not method:
        return "unknown"

    upper = method.upper().strip()

    if upper in READ_METHODS:
        return "read"

    if upper in WRITE_METHODS:
        return "write"

    if upper in OTHER_METHODS:
        return "other"

    return "unknown"


def is_read_method(method: str) -> bool:
    """Check if method is a read operation.

    Args:
        method: HTTP method string

    Returns:
        True if this is a read method
    """
    return bucket_http_method(method) == "read"


def is_write_method(method: str) -> bool:
    """Check if method is a write operation.

    Args:
        method: HTTP method string

    Returns:
        True if this is a write method
    """
    return bucket_http_method(method) == "write"
