"""Formatters for displaying BiasResult output."""

import json

from unbias_plus.schema import BiasResult


_SEVERITY_COLORS = {
    "high": "\033[91m",  # red
    "medium": "\033[93m",  # yellow
    "low": "\033[94m",  # blue
    "reset": "\033[0m",
}


def format_cli(result: BiasResult) -> str:
    """Format a BiasResult for CLI terminal display.

    Produces a human-readable, colored terminal output showing
    the bias label, severity, each biased segment with its
    replacement and reasoning, and the full unbiased rewrite.

    Parameters
    ----------
    result : BiasResult
        The bias analysis result to format.

    Returns
    -------
    str
        A human-readable colored string for terminal output.

    Examples
    --------
    >>> result = BiasResult(
    ...     binary_label="biased",
    ...     severity=3,
    ...     bias_found=True,
    ...     biased_segments=[],
    ...     unbiased_text="Neutral.",
    ... )
    >>> output = format_cli(result)
    >>> isinstance(output, str)
    True

    """
    lines = []
    lines.append("=" * 60)
    lines.append(
        f"BIAS DETECTED: {result.binary_label.upper()} "
        f"| Overall Severity: {result.severity}/5"
    )
    lines.append(f"Segments found: {len(result.biased_segments)}")
    lines.append("=" * 60)

    if result.biased_segments:
        lines.append("\nBIASED SEGMENTS:")
        for i, seg in enumerate(result.biased_segments, 1):
            color = _SEVERITY_COLORS.get(seg.severity, "")
            reset = _SEVERITY_COLORS["reset"]
            lines.append(f"\n  [{i}] {color}{seg.severity.upper()}{reset}")
            lines.append(f'  Original  : "{seg.original}"')
            lines.append(f'  Replace   : "{seg.replacement}"')
            lines.append(f"  Bias type : {seg.bias_type}")
            lines.append(f"  Reasoning : {seg.reasoning}")

    lines.append("\n" + "-" * 60)
    lines.append("NEUTRAL REWRITE:")
    lines.append(result.unbiased_text)
    lines.append("=" * 60)

    return "\n".join(lines)


def format_dict(result: BiasResult) -> dict:
    """Convert a BiasResult to a plain Python dictionary.

    Parameters
    ----------
    result : BiasResult
        The bias analysis result to convert.

    Returns
    -------
    dict
        Plain dictionary representation of the result.

    Examples
    --------
    >>> result = BiasResult(
    ...     binary_label="biased",
    ...     severity=3,
    ...     bias_found=True,
    ...     biased_segments=[],
    ...     unbiased_text="Neutral.",
    ... )
    >>> d = format_dict(result)
    >>> isinstance(d, dict)
    True

    """
    return result.model_dump()


def format_json(result: BiasResult) -> str:
    """Convert a BiasResult to a formatted JSON string.

    Parameters
    ----------
    result : BiasResult
        The bias analysis result to convert.

    Returns
    -------
    str
        Pretty-printed JSON string representation of the result.

    Examples
    --------
    >>> result = BiasResult(
    ...     binary_label="biased",
    ...     severity=3,
    ...     bias_found=True,
    ...     biased_segments=[],
    ...     unbiased_text="Neutral.",
    ... )
    >>> json_str = format_json(result)
    >>> isinstance(json_str, str)
    True

    """
    return json.dumps(result.model_dump(), indent=2)
