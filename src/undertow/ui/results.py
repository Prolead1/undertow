"""Parse ``results/`` evaluation artifacts into :class:`EvalMetrics` (U02).

The dashboard never evaluates; it only reads what the S15/S16 writers already
produced (`docs/plans/ui/PLAN.md` §0.6).  This module turns those markdown/JSON
files into the frozen :class:`~undertow.ui.schemas.EvalMetrics` shape.

Recognised inputs
-----------------
* ``regime_comparison.md`` / ``regime_matrix_<policy>.md`` — per-regime PnL
  (and its minimum, ``worst_regime_pnl``).
* ``summary.json`` — backtest metrics (``sharpe``, ``total_pnl``,
  ``annualized_return``, ``max_drawdown``); the numbers live under a ``result``
  object, but top-level keys are accepted too.
* ``results_summary.md`` — a best-effort markdown fallback for the same metrics.

Everything is defensive: an unknown markdown shape yields empty/``None`` fields
and never raises.  The returned :class:`EvalMetrics` records the relative
``source`` path the numbers came from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from undertow.ui.schemas import EvalMetrics

__all__ = [
    "load_eval_metrics",
    "parse_regime_pnl",
    "parse_results_summary_markdown",
    "parse_summary_json",
]

#: Filename (in preference order) for the per-regime PnL table.
_REGIME_FILENAMES: tuple[str, ...] = ("regime_comparison.md",)
_REGIME_GLOB = "regime_matrix_*.md"
_SUMMARY_JSON_FILENAME = "summary.json"
_SUMMARY_MD_FILENAME = "results_summary.md"

#: Metric names shared by ``summary.json`` / ``results_summary.md``.
_METRIC_KEYS: tuple[str, ...] = (
    "sharpe",
    "total_pnl",
    "annualized_return",
    "max_drawdown",
)

_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _parse_number(cell: str) -> float | None:
    """Pull the leading number out of a markdown cell (handles ``1,234.5 ± 3``)."""
    text = cell.split("±")[0].strip()
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:  # pragma: no cover - regex only matches parseable numbers
        return None


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Return every GFM pipe table as ``(headers, rows)`` pairs."""
    lines = text.splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            "|" in line
            and index + 1 < len(lines)
            and _SEPARATOR_RE.match(lines[index + 1])
        ):
            headers = _split_row(line)
            rows: list[list[str]] = []
            cursor = index + 2
            while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
                cells = _split_row(lines[cursor])
                if cells:
                    rows.append(cells)
                cursor += 1
            tables.append((headers, rows))
            index = cursor
        else:
            index += 1
    return tables


def _column_index(headers: list[str], names: frozenset[str]) -> int | None:
    for position, header in enumerate(headers):
        if header.strip().lower() in names:
            return position
    return None


def _read_text(path: Path) -> str | None:
    """Read a text artifact, returning ``None`` on any I/O/decoding failure."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None


def parse_regime_pnl(text: str, *, policy: str | None = None) -> dict[str, float]:
    """Return ``regime -> mean total_pnl`` from a per-regime markdown table.

    Handles both ``regime_comparison.md`` (``Policy | Regime | PnL | ...``) and
    ``regime_matrix_<policy>.md`` (``Regime | PnL | ...``).  The PnL cell may be
    ``"1,234.56 ± 12.0"``; only the mean is kept.  When ``policy`` is given,
    rows from the other policies are ignored (the comparison table holds every
    policy); otherwise the first occurrence of a regime wins.
    """
    pnl_names = frozenset({"pnl", "pnl (usdc)", "pnl (usd)", "total_pnl"})
    result: dict[str, float] = {}
    for headers, rows in _markdown_tables(text):
        regime_index = _column_index(headers, frozenset({"regime"}))
        pnl_index = _column_index(headers, pnl_names)
        if regime_index is None or pnl_index is None:
            continue
        policy_index = _column_index(headers, frozenset({"policy", "policy_name"}))
        for row in rows:
            if max(regime_index, pnl_index) >= len(row):
                continue
            if (
                policy is not None
                and policy_index is not None
                and (policy_index >= len(row) or row[policy_index].strip() != policy)
            ):
                continue
            regime = row[regime_index].strip()
            value = _parse_number(row[pnl_index])
            if regime and value is not None and regime not in result:
                result[regime] = value
    return result


def parse_summary_json(text: str) -> dict[str, float | None]:
    """Map ``summary.json`` onto the :class:`EvalMetrics` metric fields.

    The backtest writer nests the numbers under ``"result"``; a flat payload is
    accepted as well.  Malformed JSON yields an all-``None`` dict.
    """
    metrics: dict[str, float | None] = dict.fromkeys(_METRIC_KEYS, None)
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return metrics
    if not isinstance(payload, dict):
        return metrics

    nested = payload.get("result")
    sources: list[dict[str, object]] = []
    if isinstance(nested, dict):
        sources.append(nested)
    sources.append(payload)

    for key in _METRIC_KEYS:
        for source in sources:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)
                break
    return metrics


def parse_results_summary_markdown(text: str) -> dict[str, float | None]:
    """Best-effort extraction of metrics from the real ``results_summary.md``.

    The S16 writer emits a ``| Ablation | PnL (USDC) | Sharpe |`` table and a
    ``* Best PnL: **name** (1,234.56 USDC)`` bullet; there is no MaxDD or
    annualised figure in that file, so those stay ``None``.  The row with the
    highest PnL supplies both ``total_pnl`` and its matching ``sharpe``.
    """
    metrics: dict[str, float | None] = dict.fromkeys(_METRIC_KEYS, None)

    best_pnl: float | None = None
    best_sharpe: float | None = None
    for headers, rows in _markdown_tables(text):
        pnl_index = _column_index(
            headers, frozenset({"pnl", "pnl (usdc)", "pnl (usd)", "total_pnl"})
        )
        if pnl_index is None:
            continue
        sharpe_index = _column_index(headers, frozenset({"sharpe"}))
        for row in rows:
            if pnl_index >= len(row):
                continue
            value = _parse_number(row[pnl_index])
            if value is None or (best_pnl is not None and value <= best_pnl):
                continue
            best_pnl = value
            best_sharpe = (
                _parse_number(row[sharpe_index])
                if sharpe_index is not None and sharpe_index < len(row)
                else None
            )

    if best_pnl is None:
        match = re.search(
            r"best pnl[^0-9+\-]*([-+]?[\d,]+(?:\.\d+)?)\s*usdc", text.lower()
        )
        if match is not None:
            best_pnl = float(match.group(1).replace(",", ""))

    metrics["total_pnl"] = best_pnl
    metrics["sharpe"] = best_sharpe
    return metrics


def _first_existing(results_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = results_dir / name
        if candidate.is_file():
            return candidate
    return None


def _relative(results_dir: Path, path: Path, source_prefix: str | None) -> str:
    relative = path.relative_to(results_dir).as_posix()
    return f"{source_prefix}/{relative}" if source_prefix else relative


def load_eval_metrics(
    results_dir: Path | str,
    *,
    source_prefix: str | None = None,
    policy: str | None = None,
) -> EvalMetrics | None:
    """Parse a results directory into :class:`EvalMetrics`.

    Returns ``None`` when the directory does not exist or contains none of the
    recognised artifacts.  When artifacts exist but are unrecognised or
    undecodable, the affected fields stay ``None`` and no exception escapes.
    ``policy`` selects the run's own rows (and prefers ``regime_matrix_<policy>.md``)
    when the results root holds more than one policy.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        return None

    regime_path: Path | None = None
    if policy is not None:
        regime_path = _first_existing(directory, (f"regime_matrix_{policy}.md",))
    if regime_path is None:
        regime_path = _first_existing(directory, _REGIME_FILENAMES)
    if regime_path is None:
        matrix_candidates = sorted(directory.glob(_REGIME_GLOB))
        regime_path = matrix_candidates[0] if matrix_candidates else None

    summary_json_path = _first_existing(directory, (_SUMMARY_JSON_FILENAME,))
    summary_md_path = _first_existing(directory, (_SUMMARY_MD_FILENAME,))
    if regime_path is None and summary_json_path is None and summary_md_path is None:
        return None

    per_regime: dict[str, float] = {}
    source: str | None = None
    if regime_path is not None:
        regime_text = _read_text(regime_path)
        if regime_text is not None:
            per_regime = parse_regime_pnl(regime_text, policy=policy)
        source = _relative(directory, regime_path, source_prefix)

    raw: dict[str, float | None] = dict.fromkeys(_METRIC_KEYS, None)
    if summary_json_path is not None:
        summary_text = _read_text(summary_json_path)
        if summary_text is not None:
            raw = parse_summary_json(summary_text)
        if source is None:
            source = _relative(directory, summary_json_path, source_prefix)
    if summary_md_path is not None and any(value is None for value in raw.values()):
        markdown_text = _read_text(summary_md_path)
        if markdown_text is not None:
            for key, value in parse_results_summary_markdown(markdown_text).items():
                if raw[key] is None and value is not None:
                    raw[key] = value
        if source is None:
            source = _relative(directory, summary_md_path, source_prefix)

    worst = min(per_regime.values()) if per_regime else None
    return EvalMetrics(
        sharpe=raw["sharpe"],
        total_pnl=raw["total_pnl"],
        annualized_return=raw["annualized_return"],
        max_drawdown=raw["max_drawdown"],
        worst_regime_pnl=worst,
        per_regime=per_regime,
        source=source,
    )
