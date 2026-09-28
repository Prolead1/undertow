"""Tests for the generated data dictionary (T16 tests #10 and #11).

#10 freshness: ``render_data_dictionary()`` must byte-equal the committed
``docs/data_dictionary.md``. #11 coverage: every stream in ``SCHEMA_REGISTRY``
has a section and every one of its columns appears as a table row with a
non-empty unit/convention cell equal to the schema's own ``unit:<name>``
metadata. The prose assertions pin the two conventions a reader must not get
wrong — swap signs and big-integer storage.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from undertow.data.dictionary import STREAM_SOURCES, render_data_dictionary
from undertow.data.schemas import POOL_SCOPED, SCHEMA_REGISTRY, SORT_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC_PATH = _REPO_ROOT / "docs" / "data_dictionary.md"

_REGENERATE_COMMAND = "uv run python -m undertow.data.dictionary"

# A stream heading is exactly ``### `swap` `` (single token, backticked).
_STREAM_HEADING = re.compile(r"^###\s+`([a-z_]+)`\s*$")
_HEADER_CELLS = ["column", "type", "unit / convention", "nullable"]
_CELL_SEPARATOR = re.compile(r"(?<!\\)\|")


def _split_row(line: str) -> list[str]:
    """Split a markdown table row on unescaped pipes and unescape cell content."""
    inner = line.strip()
    assert inner.startswith("|") and inner.endswith("|"), f"not a table row: {line!r}"
    cells = _CELL_SEPARATOR.split(inner[1:-1])
    return [cell.strip().replace("\\|", "|") for cell in cells]


def _parse_tables(markdown: str) -> dict[str, list[dict[str, str]]]:
    """Parse the per-stream column tables out of the generated markdown.

    Returns ``{stream_name: [row_dict, ...]}`` where each row maps
    ``column``/``type``/``unit / convention``/``nullable`` to its cell text.
    """
    tables: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("### "):
            heading = _STREAM_HEADING.match(lines[i])
            current = heading.group(1) if heading is not None else None
        if (
            current is not None
            and lines[i].startswith("|")
            and _split_row(lines[i]) == _HEADER_CELLS
        ):
            i += 2  # skip the header row and the |---|---| separator
            rows: list[dict[str, str]] = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(dict(zip(_HEADER_CELLS, _split_row(lines[i]), strict=True)))
                i += 1
            tables.setdefault(current, []).extend(rows)
            continue
        i += 1
    return tables


def _strip_code(cell: str) -> str:
    return cell.strip("`").strip()


# ---------------------------------------------------------------------------
# #10 — freshness
# ---------------------------------------------------------------------------


def test_dictionary_is_fresh() -> None:
    """The committed dictionary must be exactly what the generator produces."""
    committed = _DOC_PATH.read_text(encoding="utf-8")
    regenerated = render_data_dictionary()
    assert committed == regenerated, (
        "docs/data_dictionary.md is stale — regenerate it with:\n"
        f"    {_REGENERATE_COMMAND}"
    )


def test_dictionary_banner_marks_it_generated() -> None:
    doc = _DOC_PATH.read_text(encoding="utf-8")
    assert "GENERATED FILE" in doc
    assert "do not edit by hand" in doc.lower()
    assert _REGENERATE_COMMAND in doc


# ---------------------------------------------------------------------------
# #11 — coverage
# ---------------------------------------------------------------------------


def test_every_registry_stream_has_a_source_route() -> None:
    assert set(STREAM_SOURCES) == set(SCHEMA_REGISTRY)


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_every_column_documented_with_unit_and_type(stream: str) -> None:
    schema = SCHEMA_REGISTRY[stream]
    tables = _parse_tables(render_data_dictionary())

    assert stream in tables, f"stream {stream!r} has no table in the dictionary"
    rows = {_strip_code(row["column"]): row for row in tables[stream]}

    expected_columns = [field.name for field in schema]
    assert set(rows) == set(expected_columns), (
        f"dictionary stream {stream!r} column set mismatch: "
        f"missing={sorted(set(expected_columns) - set(rows))}, "
        f"extra={sorted(set(rows) - set(expected_columns))}"
    )

    for field in schema:
        row = rows[field.name]
        for cell_name in _HEADER_CELLS:
            assert row[cell_name] != "", (
                f"{stream}.{field.name}: empty {cell_name!r} cell"
            )
        # The unit cell is the schema metadata, decoded and pipe-unescaped.
        expected_unit = schema.metadata[f"unit:{field.name}".encode()].decode("utf-8")
        assert row["unit / convention"] == expected_unit, (
            f"{stream}.{field.name}: unit cell {row['unit / convention']!r} != "
            f"schema metadata {expected_unit!r}"
        )
        assert row["type"] == f"`{field.type}`"
        assert row["nullable"] == ("yes" if field.nullable else "no")


@pytest.mark.parametrize("stream", sorted(SCHEMA_REGISTRY))
def test_stream_provenance_shows_route_sort_key_and_scoping(stream: str) -> None:
    doc = render_data_dictionary()
    assert STREAM_SOURCES[stream] in doc, f"source route for {stream!r} missing from doc"
    sort_key = SORT_KEYS[stream]
    suffix = "," if len(sort_key) == 1 else ""
    rendered = f"({', '.join(sort_key)}{suffix})"
    assert f"**Sort key:** `{rendered}`" in doc
    scoped = "yes" if stream in POOL_SCOPED else "no"
    assert f"**Pool-scoped:** {scoped}" in doc


def test_pipe_bearing_units_round_trip_through_the_table() -> None:
    """A unit string containing ``|`` must survive markdown escaping intact."""
    tables = _parse_tables(render_data_dictionary())
    regime_row = next(
        row for row in tables["regime"] if _strip_code(row["column"]) == "regime"
    )
    assert regime_row["unit / convention"] == (
        "label: bull|bear|sideways|high_vol|unknown"
    )


# ---------------------------------------------------------------------------
# Prose — the conventions a reader must not get wrong
# ---------------------------------------------------------------------------


def test_prose_documents_swap_sign_convention() -> None:
    doc = render_data_dictionary()
    assert "positive = flowed into the pool" in doc
    assert "fee is" in doc and "input" in doc


def test_prose_documents_big_integer_convention() -> None:
    doc = render_data_dictionary()
    assert "decimal string" in doc
    assert "float64" in doc and "53 bits" in doc
    assert "encode_uint" in doc and "decode_uint" in doc


def test_prose_documents_price_orientation_and_collect_semantics() -> None:
    doc = render_data_dictionary()
    # T02's orientation sentence, quoted verbatim.
    assert "Human price of token1 denominated in token0 — USDC per WETH" in doc
    assert "USDC per WETH" in doc
    assert "Collect" in doc and "not the fees accrued in" in doc
    assert "Burn" in doc and "principal" in doc


def test_prose_documents_regime_thresholds_and_adr001() -> None:
    doc = render_data_dictionary()
    assert "ADR-001" in doc
    assert "0.80" in doc and "0.05" in doc
    assert "log(S_end/S_start)" in doc
    assert "window_complete" in doc


def test_prose_documents_gap_policy_and_known_limitations() -> None:
    doc = render_data_dictionary()
    assert "forward-filled and flagged" in doc
    assert "USDT as a USD proxy" in doc
    assert "NFT position manager" in doc
    assert "sampled, not per-block" in doc


def test_render_is_deterministic() -> None:
    """The generator is pure: two renders are byte-identical."""
    assert render_data_dictionary() == render_data_dictionary()