"""Route parameterizer - replaces dynamic path segments with placeholders."""

import re
from typing import Callable


# Parameterization patterns in order of specificity
ROUTE_PATTERNS: list[tuple[str, str]] = [
    # UUID: 8-4-4-4-12 hex format
    (
        r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",
        "{uuid}",
    ),
    # Long hex ID (16+ chars)
    (
        r"(?<=/)[a-fA-F0-9]{16,}(?=/|$)",
        "{hex_id}",
    ),
    # MongoDB ObjectId (24 hex chars)
    (
        r"(?<=/)[a-fA-F0-9]{24}(?=/|$)",
        "{object_id}",
    ),
    # Email address
    (
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "{email}",
    ),
    # ISO date (YYYY-MM-DD)
    (
        r"\d{4}-\d{2}-\d{2}",
        "{date}",
    ),
    # Timestamp (10 or 13 digits)
    (
        r"(?<=/)\d{10,13}(?=/|$)",
        "{timestamp}",
    ),
    # Numeric ID in path segment
    (
        r"(?<=/)\d+(?=/|$)",
        "{id}",
    ),
    # Long slug (20+ alphanumeric chars with underscores/hyphens)
    (
        r"(?<=/)[a-zA-Z0-9_-]{20,}(?=/|$)",
        "{slug}",
    ),
]


def parameterize_route(route: str, patterns: list[tuple[str, str]] | None = None) -> str:
    """Replace dynamic route segments with placeholders.

    Args:
        route: Original route/path string
        patterns: Optional custom patterns to use

    Returns:
        Parameterized route
    """
    if not route:
        return route

    patterns = patterns or ROUTE_PATTERNS
    result = route

    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    result = re.sub(r"/\{([^}]+)\}/\{([^}]+)\}", r"/{\1_\2}", result)

    return result


def get_route_template(route: str) -> str:
    """Get a normalized route template for grouping.

    This is more aggressive than parameterize_route, converting any path
    segment that looks dynamic into a placeholder.

    Args:
        route: Original route/path string

    Returns:
        Route template suitable for grouping
    """
    if not route:
        return route

    result = parameterize_route(route)

    segments = result.split("/")
    normalized_segments = []

    for segment in segments:
        if not segment:
            normalized_segments.append(segment)
            continue

        if segment.startswith("{") and segment.endswith("}"):
            normalized_segments.append(segment)
            continue

        if re.match(r"^\d+$", segment):
            normalized_segments.append("{id}")
        elif re.match(r"^[a-fA-F0-9]+$", segment) and len(segment) >= 8:
            normalized_segments.append("{hex_id}")
        elif re.match(r"^[a-zA-Z0-9_-]+$", segment) and len(segment) >= 20:
            normalized_segments.append("{slug}")
        else:
            normalized_segments.append(segment)

    return "/".join(normalized_segments)


def extract_route_patterns(routes: list[str]) -> dict[str, list[str]]:
    """Group routes by their templates.

    Args:
        routes: List of route strings

    Returns:
        Dict mapping template to list of original routes
    """
    templates: dict[str, list[str]] = {}

    for route in routes:
        template = get_route_template(route)
        if template not in templates:
            templates[template] = []
        templates[template].append(route)

    return templates


def create_custom_parameterizer(
    custom_patterns: list[tuple[str, str]],
) -> Callable[[str], str]:
    """Create a custom parameterizer with additional patterns.

    Args:
        custom_patterns: Additional patterns to apply after defaults

    Returns:
        Parameterizer function
    """
    combined_patterns = ROUTE_PATTERNS + custom_patterns

    def parameterizer(route: str) -> str:
        return parameterize_route(route, combined_patterns)

    return parameterizer
