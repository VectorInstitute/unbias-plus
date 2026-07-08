"""Text cleaning helpers applied before annotation."""

import re


DOUBLE_QUOTES = re.compile(r'["\u201c\u201d\u00ab\u00bb\u201e\u201a]')
SINGLE_QUOTES = re.compile(
    r"(?<![A-Za-z0-9])['\u2018\u2019]|['\u2018\u2019](?![A-Za-z0-9])"
)


def strip_quotes(text: str) -> str:
    text = DOUBLE_QUOTES.sub("", text)
    return SINGLE_QUOTES.sub("", text)
