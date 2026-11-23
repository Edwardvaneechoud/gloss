"""Finding the facts that mention a column or a table, for drift and re-keying.

Relationship references are exact (their columns are data). Metric references are a
word-match on SQL text: good enough to send a metric back to review, never used to
rewrite one.
"""

from __future__ import annotations

import re
from typing import Any


def expression_texts(expr: Any) -> list[str]:
    if isinstance(expr, dict):
        out = [expr.get("sql") or ""]
        out += [d.get("expression", "") for d in expr.get("dialects", [])]
        return out
    return [str(expr)]


def _word(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def short_name(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def mentions_dataset(expr: Any, key: str) -> bool:
    """Whether a metric expression names the dataset, by its key or as ``<short name>.``."""
    text = " ".join(expression_texts(expr))
    return key in text or re.search(rf"\b{re.escape(short_name(key))}\s*\.", text) is not None


def mentions_column(expr: Any, key: str, column: str, *, scoped: bool) -> bool:
    """Whether a metric expression uses ``column`` of dataset ``key``.

    ``scoped`` says the metric belongs to that dataset, so a bare column name is enough.
    """
    text = " ".join(expression_texts(expr))
    return bool(_word(column).search(text)) and (scoped or mentions_dataset(expr, key))


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    return name if _IDENT.match(name) else '"' + name.replace('"', '""') + '"'
