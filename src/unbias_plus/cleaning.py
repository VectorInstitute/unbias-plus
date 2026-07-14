"""Text cleaning helpers applied before inference.

Matches ``data_annotation/cleaning.py`` so serve-time input matches
the quote-stripped distribution used for annotation / SFT.
"""

import re


DOUBLE_QUOTES = re.compile(r'["\u201c\u201d\u00ab\u00bb\u201e\u201a]')
SINGLE_QUOTES = re.compile(
    r"(?<![A-Za-z0-9])['\u2018\u2019]|['\u2018\u2019](?![A-Za-z0-9])"
)


def strip_quotes(text: str) -> str:
    """Remove double and edge single quote characters from ``text``."""
    text = DOUBLE_QUOTES.sub("", text)
    return SINGLE_QUOTES.sub("", text)


def prepare_input(text: str) -> str:
    """Normalize article text before prompt construction / offset mapping."""
    return strip_quotes(text)
