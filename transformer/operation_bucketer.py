"""Operation bucketing - transforms DB/RPC operations into categories."""

import re
from typing import Literal

OperationBucket = Literal[
    "read",
    "write",
    "ddl",
    "transaction",
    "metadata",
    "stream",
    "publish",
    "consume",
    "ack",
    "other",
]


# SQL operation patterns (case-insensitive)
SQL_READ_PATTERNS = [
    r"^SELECT\b",
    r"^FIND\b",
    r"^GET\b",
    r"^READ\b",
    r"^QUERY\b",
    r"^FETCH\b",
    r"^LOAD\b",
    r"^LIST\b",
    r"^SEARCH\b",
    r"^COUNT\b",
]

SQL_WRITE_PATTERNS = [
    r"^INSERT\b",
    r"^UPDATE\b",
    r"^DELETE\b",
    r"^UPSERT\b",
    r"^SAVE\b",
    r"^PUT\b",
    r"^MERGE\b",
    r"^REPLACE\b",
    r"^REMOVE\b",
]

SQL_DDL_PATTERNS = [
    r"^CREATE\b",
    r"^ALTER\b",
    r"^DROP\b",
    r"^TRUNCATE\b",
    r"^RENAME\b",
    r"^GRANT\b",
    r"^REVOKE\b",
]

SQL_TRANSACTION_PATTERNS = [
    r"^BEGIN\b",
    r"^COMMIT\b",
    r"^ROLLBACK\b",
    r"^SAVEPOINT\b",
    r"^START\s+TRANSACTION\b",
]

SQL_METADATA_PATTERNS = [
    r"^EXPLAIN\b",
    r"^ANALYZE\b",
    r"^DESCRIBE\b",
    r"^SHOW\b",
    r"^SET\b",
]

# RPC/gRPC method patterns (case-insensitive prefix)
RPC_READ_PREFIXES = [
    "Get", "List", "Find", "Query", "Read", "Fetch", "Search", "Load", "Describe",
    "Check", "Validate", "Verify", "Lookup", "Resolve",
]

RPC_WRITE_PREFIXES = [
    "Create", "Update", "Delete", "Set", "Add", "Remove", "Put", "Insert",
    "Upsert", "Save", "Modify", "Patch", "Store", "Register", "Unregister",
]

RPC_STREAM_PREFIXES = [
    "Stream", "Watch", "Subscribe", "Listen", "Observe",
]

# Messaging operation patterns
MESSAGING_PUBLISH_PATTERNS = [
    r"\b(publish|send|produce|emit|dispatch)\b",
]

MESSAGING_CONSUME_PATTERNS = [
    r"\b(consume|receive|subscribe|poll|fetch)\b",
]

MESSAGING_ACK_PATTERNS = [
    r"\b(ack|nack|reject|commit|acknowledge)\b",
]


def bucket_sql_operation(operation: str) -> OperationBucket:
    """Bucket SQL operation into category.

    Args:
        operation: SQL operation/statement string

    Returns:
        Operation bucket category
    """
    if not operation:
        return "other"

    upper = operation.upper().strip()

    for pattern in SQL_READ_PATTERNS:
        if re.search(pattern, upper):
            return "read"

    for pattern in SQL_WRITE_PATTERNS:
        if re.search(pattern, upper):
            return "write"

    for pattern in SQL_DDL_PATTERNS:
        if re.search(pattern, upper):
            return "ddl"

    for pattern in SQL_TRANSACTION_PATTERNS:
        if re.search(pattern, upper):
            return "transaction"

    for pattern in SQL_METADATA_PATTERNS:
        if re.search(pattern, upper):
            return "metadata"

    return "other"


def bucket_rpc_operation(method_name: str) -> OperationBucket:
    """Bucket RPC/gRPC method name into category.

    Args:
        method_name: RPC method name (e.g., GetUser, CreateOrder)

    Returns:
        Operation bucket category
    """
    if not method_name:
        return "other"

    parts = method_name.split("/")
    method = parts[-1] if parts else method_name

    for prefix in RPC_STREAM_PREFIXES:
        if method.startswith(prefix):
            return "stream"

    for prefix in RPC_READ_PREFIXES:
        if method.startswith(prefix):
            return "read"

    for prefix in RPC_WRITE_PREFIXES:
        if method.startswith(prefix):
            return "write"

    return "other"


def bucket_messaging_operation(operation: str) -> OperationBucket:
    """Bucket messaging operation into category.

    Args:
        operation: Messaging operation string

    Returns:
        Operation bucket category
    """
    if not operation:
        return "other"

    lower = operation.lower()

    for pattern in MESSAGING_PUBLISH_PATTERNS:
        if re.search(pattern, lower):
            return "publish"

    for pattern in MESSAGING_CONSUME_PATTERNS:
        if re.search(pattern, lower):
            return "consume"

    for pattern in MESSAGING_ACK_PATTERNS:
        if re.search(pattern, lower):
            return "ack"

    return "other"


def bucket_operation(
    operation: str,
    operation_type: str | None = None,
) -> OperationBucket:
    """Smart operation bucketing based on type hint.

    Args:
        operation: Operation string
        operation_type: Hint about type: 'sql', 'rpc', 'messaging', or None

    Returns:
        Operation bucket category
    """
    if not operation:
        return "other"

    if operation_type == "sql" or operation_type == "db":
        return bucket_sql_operation(operation)

    if operation_type == "rpc" or operation_type == "grpc":
        return bucket_rpc_operation(operation)

    if operation_type == "messaging" or operation_type == "queue":
        return bucket_messaging_operation(operation)

    upper = operation.upper().strip()
    for pattern in SQL_READ_PATTERNS + SQL_WRITE_PATTERNS + SQL_DDL_PATTERNS:
        if re.search(pattern, upper):
            return bucket_sql_operation(operation)

    for prefix in RPC_READ_PREFIXES + RPC_WRITE_PREFIXES + RPC_STREAM_PREFIXES:
        if operation.startswith(prefix):
            return bucket_rpc_operation(operation)

    return "other"
