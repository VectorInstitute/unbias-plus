"""Formatters for displaying BiasResult output."""

from unbias_plus.schema import BiasResult


def format_cli(result: BiasResult) -> str:
    """Format a BiasResult for CLI terminal display.

    Parameters
    ----------
    result : BiasResult
        The bias analysis result to format.

    Returns
    -------
    str
        A human-readable string for terminal output.

    Examples
    --------
    >>> output = format_cli(result)
    >>> print(output)

    """
    pass


def format_dict(result: BiasResult) -> dict:  # type: ignore[type-arg]
    """Convert a BiasResult to a plain Python dictionary.

    Parameters
    ----------
    result : BiasResult
        The bias analysis result to convert.

    Returns
    -------
    dict
        Plain dictionary representation of the result.

    """
    pass


def format_json(result: BiasResult) -> str:
    """Convert a BiasResult to a JSON string.

    Parameters
    ----------
    result : BiasResult
        The bias analysis result to convert.

    Returns
    -------
    str
        JSON string representation of the result.

    """
    pass
