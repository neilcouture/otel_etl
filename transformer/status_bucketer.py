"""Status code bucketing - transforms HTTP/gRPC status codes into categories."""

import re
from typing import Literal

StatusBucket = Literal[
    "success",
    "redirect",
    "client_error",
    "server_error",
    "informational",
    "unknown",
]


def bucket_http_status(status: str | int) -> StatusBucket:
    """Bucket HTTP status code into category.

    Args:
        status: HTTP status code (numeric or string)

    Returns:
        Status bucket category
    """
    try:
        code = int(status)
    except (ValueError, TypeError):
        return _bucket_by_text(str(status))

    if 100 <= code < 200:
        return "informational"
    if 200 <= code < 300:
        return "success"
    if 300 <= code < 400:
        return "redirect"
    if 400 <= code < 500:
        return "client_error"
    if 500 <= code < 600:
        return "server_error"

    return "unknown"


GRPC_STATUS_MAP: dict[int, StatusBucket] = {
    0: "success",        # OK
    1: "client_error",   # CANCELLED
    2: "server_error",   # UNKNOWN
    3: "client_error",   # INVALID_ARGUMENT
    4: "server_error",   # DEADLINE_EXCEEDED
    5: "client_error",   # NOT_FOUND
    6: "client_error",   # ALREADY_EXISTS
    7: "client_error",   # PERMISSION_DENIED
    8: "server_error",   # RESOURCE_EXHAUSTED
    9: "client_error",   # FAILED_PRECONDITION
    10: "client_error",  # ABORTED
    11: "client_error",  # OUT_OF_RANGE
    12: "server_error",  # UNIMPLEMENTED
    13: "server_error",  # INTERNAL
    14: "server_error",  # UNAVAILABLE
    15: "server_error",  # DATA_LOSS
    16: "client_error",  # UNAUTHENTICATED
}


def bucket_grpc_status(status: str | int) -> StatusBucket:
    """Bucket gRPC status code into category.

    Args:
        status: gRPC status code (numeric or string)

    Returns:
        Status bucket category
    """
    try:
        code = int(status)
        return GRPC_STATUS_MAP.get(code, "unknown")
    except (ValueError, TypeError):
        return _bucket_by_text(str(status))


def _bucket_by_text(status: str) -> StatusBucket:
    """Bucket status by text content analysis.

    Args:
        status: Status string

    Returns:
        Status bucket category
    """
    lower = status.lower()

    if re.search(r"\b(ok|success|200|2\d\d)\b", lower):
        return "success"

    if re.search(r"\b(err|fail|exception|error)\b", lower):
        return "server_error"

    if re.search(r"\b(not.?found|unauthorized|forbidden|4\d\d)\b", lower):
        return "client_error"

    if re.search(r"\b(5\d\d|internal|unavailable)\b", lower):
        return "server_error"

    if re.search(r"\b(redirect|3\d\d)\b", lower):
        return "redirect"

    return "unknown"


def bucket_status_code(
    status: str | int,
    label_name: str | None = None,
) -> StatusBucket:
    """Smart status code bucketing based on label name hint.

    Args:
        status: Status code value
        label_name: Optional label name to help determine type

    Returns:
        Status bucket category
    """
    if label_name:
        lower_name = label_name.lower()
        if "grpc" in lower_name:
            return bucket_grpc_status(status)
        if "http" in lower_name:
            return bucket_http_status(status)

    try:
        code = int(status)
        if code >= 100:
            return bucket_http_status(code)
        else:
            return bucket_grpc_status(code)
    except (ValueError, TypeError):
        return _bucket_by_text(str(status))


def is_success(status: str | int, label_name: str | None = None) -> bool:
    """Check if status indicates success.

    Args:
        status: Status code value
        label_name: Optional label name hint

    Returns:
        True if status indicates success
    """
    return bucket_status_code(status, label_name) == "success"


def is_error(status: str | int, label_name: str | None = None) -> bool:
    """Check if status indicates an error (client or server).

    Args:
        status: Status code value
        label_name: Optional label name hint

    Returns:
        True if status indicates an error
    """
    bucket = bucket_status_code(status, label_name)
    return bucket in ("client_error", "server_error")
