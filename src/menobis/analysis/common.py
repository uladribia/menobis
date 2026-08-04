"""Shared analysis helpers with no intra-package analysis imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from menobis.data.frames import EdgeTable


def node_count(edges: EdgeTable) -> int:
    """Derive the node count from an edge table (shared helper).

    All analysis modules must use this single helper so that the node-count
    derivation is defined once.
    """
    if len(edges) == 0:
        return 0
    return int(max(edges.source.max(), edges.target.max())) + 1


__all__ = ["node_count"]
