# topic_tree.py
"""
Builds a navigation tree from chunk breadcrumbs and provides helpers
for the cascading chapter/section dropdowns in the Gradio UI.

Breadcrumbs are expected in the form  "Book > Chapter > Section",
with any number of levels.  The tree is truncated at max_depth.
"""

from __future__ import annotations
import re
from config import TOPIC_TREE_MAX_DEPTH


def natural_sort(items: list[str]) -> list[str]:
    """
    Sort strings with embedded numbers in human order.

    "Chapter 2" < "Chapter 10" (not "Chapter 10" < "Chapter 2").
    """
    def key(s: str) -> list:
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
    return sorted(items, key=key)


class TopicTree:
    """
    Hierarchy built from chunk metadata breadcrumbs.

    _tree is a nested dict: each key is a path segment and the value
    is a dict of its children (empty dict = leaf node).
    """

    def __init__(self, chunks: list, max_depth: int = TOPIC_TREE_MAX_DEPTH):
        self.max_depth = max(1, max_depth)
        self._tree: dict = {}
        self._build(chunks)

    def _build(self, chunks: list) -> None:
        for chunk in chunks:
            meta  = chunk.get("metadata", {})
            crumb = meta.get("breadcrumb", "").strip()
            if not crumb:
                h1 = meta.get("h1", "").strip()
                if h1:
                    self._tree.setdefault(h1, {})
                continue
            parts = [p.strip() for p in crumb.split(">") if p.strip()]
            parts = parts[: self.max_depth]
            node  = self._tree
            for part in parts:
                node = node.setdefault(part, {})

    def root_choices(self) -> list[str]:
        """Naturally sorted top-level choices (sources/books)."""
        return natural_sort(self._tree.keys())

    def child_choices(self, *path: str) -> list[str]:
        """
        Naturally sorted choices one level below the given path.
        Returns an empty list if the path leads to a leaf or doesn't exist.
        """
        node = self._tree
        for segment in path:
            if segment not in node:
                return []
            node = node[segment]
        return natural_sort(node.keys())

    def has_children(self, *path: str) -> bool:
        return bool(self.child_choices(*path))

    def is_empty(self) -> bool:
        return not self._tree

    def single_root(self) -> str | None:
        """Return the single root value if there is exactly one, else None."""
        roots = self.root_choices()
        return roots[0] if len(roots) == 1 else None

    def breadcrumb_prefix(self, *selections: str) -> str:
        """
        Convert dropdown selections to a breadcrumb prefix, dropping empty
        or sentinel values.

            breadcrumb_prefix("Book", "Chapter 2", "")  -> "Book > Chapter 2"
            breadcrumb_prefix("Book", "", "")            -> "Book"
        """
        parts = [s for s in selections if s and not s.startswith("--")]
        return " > ".join(parts)