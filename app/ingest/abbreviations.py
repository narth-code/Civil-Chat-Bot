"""Abbreviation definitions derived from the manuals' own text (lever 4).

A definition is a `Long Form (ABBR)` occurrence whose long form's initials, skipping small function words, spell ABBR.
Only the query side uses them: retrieval appends the expansion after the user's text; stored chunk text is untouched.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

DEFINITION = re.compile(r"((?:[A-Za-z][\w\-/&']*\s+){1,8})\(([A-Z][A-Za-z]{1,9})\)")
TOKEN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
FUNCTION_WORDS = {"of", "and", "the", "for", "in", "on", "to", "a", "an", "with"}


def _long_form(words: list[str], abbreviation: str) -> str | None:
    for count in range(max(2, len(abbreviation) - 1), min(len(words), len(abbreviation) + 3) + 1):
        span = words[-count:]
        initials = "".join(word[0].upper() for word in span if word.lower() not in FUNCTION_WORDS)
        if initials == abbreviation.upper():
            return " ".join(span)
    return None


def derive_definitions(texts: Iterable[str]) -> dict[str, str]:
    """Most frequent validated long form per abbreviation; abbreviations with competing forms are dropped."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for text in texts:
        for match in DEFINITION.finditer(text):
            long_form = _long_form(match.group(1).split(), match.group(2))
            if long_form:
                counts[match.group(2).upper()][long_form] += 1
    definitions: dict[str, str] = {}
    for abbreviation, forms in counts.items():
        ranked = forms.most_common(2)
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            definitions[abbreviation] = ranked[0][0]
    return definitions


def expand_query(query: str, definitions: dict[str, str]) -> str:
    """Append the long form of each uppercase abbreviation in the query, keeping the original wording."""
    added: list[str] = []
    for token in TOKEN.findall(query):
        long_form = definitions.get(token)
        if long_form and long_form not in added:
            added.append(long_form)
    return f"{query} {' '.join(added)}" if added else query
