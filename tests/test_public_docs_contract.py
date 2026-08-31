"""Anti-drift contract tests for the public MENoBiS documentation.

These tests guard the public user/science/API pages against the stale
patterns listed in the documentation rewrite plan (§50, §74, §97). They
fail when a documentation page claims a capability is missing, uses an
obsolete API name, or presents a non-canonical schema.

Scope: the final public documentation surface — ``guide/``, ``science/``,
``performance/``, ``api/``, ``cli/``, ``examples/``, ``index.md`` and
``getting-started.md``. Developer-only directories (``development/``,
``decisions/``, ``_generated/`` and old transitional pages) are excluded:
historical decision records intentionally preserve old statements.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

PUBLIC_DIRS = (
    "guide",
    "science",
    "performance",
    "api",
    "cli",
    "examples",
)
PUBLIC_FILES = (
    "index.md",
    "getting-started.md",
)

# Patterns that must never appear in current public docs (§50, §74, §97).
STALE_PATTERNS: tuple[str, ...] = (
    # Obsolete fixed-pair API keyword.
    r"known_rate\s*=",
    # Claims that fixed-(s,k) or other routes are unimplemented.
    r"not currently implemented",
    # Obsolete analysis helper names.
    r"weighted_clustering_coefficient\s*\(",
    r"weight_distribution\s*\(",
    # CLI microcanonical contradiction.
    r"The CLI does not expose microcanonical sampling directly",
    # Legacy "four constraint families" wording.
    r"four constraint families",
    r"microcanonical sampling for four",
    # Architecture phase staging language in public pages.
    r"Phase\s*0",
)


def _public_markdown_files() -> list[Path]:
    """Return the current public documentation files (may be an empty list)."""
    files: list[Path] = []
    for directory in PUBLIC_DIRS:
        target = DOCS_DIR / directory
        if target.is_dir():
            files.extend(target.rglob("*.md"))
            files.extend(target.rglob("*.ipynb"))
    for name in PUBLIC_FILES:
        path = DOCS_DIR / name
        if path.is_file():
            files.append(path)
    return sorted(files)


@pytest.fixture(scope="module")
def public_files() -> list[Path]:
    """The public documentation files existing at test time."""
    return _public_markdown_files()


@pytest.mark.parametrize("pattern", STALE_PATTERNS)
def test_no_stale_patterns(public_files: list[Path], pattern: str) -> None:
    """No public page may contain a stale claim or obsolete API name."""
    hits: list[str] = []
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if re.search(pattern, line):
                hits.append(f"{path.relative_to(DOCS_DIR)}:{lineno}: {line.strip()}")
    assert not hits, f"stale pattern {pattern!r} found:\n" + "\n".join(hits)


def test_supported_models_page_is_registry_authoritative() -> None:
    """``guide/supported-models.md`` must defer to the capability registry."""
    page = DOCS_DIR / "guide" / "supported-models.md"
    if not page.exists():
        pytest.skip("guide/supported-models.md not created yet")
    text = page.read_text(encoding="utf-8")
    assert "capability registry" in text
    assert "authoritative" in text


def test_generated_capability_file_has_marker() -> None:
    """The generated capability table carries the do-not-edit marker."""
    generated = DOCS_DIR / "_generated" / "capabilities.md"
    if not generated.exists():
        pytest.skip("capability table not generated yet")
    assert generated.read_text(encoding="utf-8").startswith(
        "<!-- GENERATED FILE. DO NOT EDIT BY HAND. -->"
    )


def test_file_formats_canonical_schema() -> None:
    """The canonical CSV schema is ``source,target,occ_num``.

    ``weight`` may only appear as an input alias, never as the canonical
    writer schema (§3.4, §40, §97).
    """
    page = DOCS_DIR / "api" / "file-formats.md"
    if not page.exists():
        pytest.skip("api/file-formats.md not rewritten yet")
    text = page.read_text(encoding="utf-8")
    assert re.search(r"source\s*,\s*target\s*,\s*occ_num", text), (
        "canonical CSV header must use occ_num"
    )
    assert not re.search(r"source\s*,\s*target\s*,\s*weight\b", text), (
        "weight must not appear as the canonical CSV header"
    )


def test_microcanonical_guide_has_protected_philosophy() -> None:
    """The microcanonical user guide must not expose internal gate naming."""
    page = DOCS_DIR / "guide" / "microcanonical.md"
    if not page.exists():
        pytest.skip("guide/microcanonical.md not created yet")
    text = page.read_text(encoding="utf-8")
    for internal in ("Gate C", "Gate D", "STOP record"):
        assert internal not in text, f"internal harness language leaked: {internal}"
