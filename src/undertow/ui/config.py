"""Configuration for ``undertow.ui`` — the live policy cockpit.

Implements `CONTRACTS.md` §3 exactly: one frozen, slotted :class:`UiConfig` plus
a stdlib-only TOML loader.  This module itself imports no web framework, so it
can be used (and tested) without FastAPI installed.  Importing the package
``undertow.ui`` does eagerly import the FastAPI factory via ``__init__``; the
sim/data packages still never acquire a UI dependency (`docs/plans/ui/PLAN.md`
§0.6).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

__all__ = ["UiConfig", "UiConfigError"]

#: The fields that hold a filesystem path (``web_dist`` is optional, handled separately).
_PATH_FIELDS = frozenset({"runs_dir", "results_dir"})


class UiConfigError(ValueError):
    """Raised when a UI TOML config is missing, malformed, or uses unknown keys."""


@dataclass(frozen=True, slots=True)
class UiConfig:
    """Resolved settings for the dashboard server (CONTRACTS §3).

    The defaults are the localhost research-tool defaults: bind loopback only,
    poll telemetry once a second, and downsample any series to 2000 points.
    ``cors_origins`` is empty by design — development goes through the Vite
    proxy, so the API is same-origin only unless a deployer opts in.
    """

    runs_dir: Path = Path("runs")
    results_dir: Path = Path("results")
    host: str = "127.0.0.1"
    port: int = 8050
    telemetry_poll_seconds: float = 1.0
    max_points: int = 2000
    cors_origins: tuple[str, ...] = ()
    web_dist: Path | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> UiConfig:
        """Load a :class:`UiConfig` from a flat TOML file.

        Every top-level key must name a field of this dataclass; omitted keys
        keep their default.  Unknown keys raise :class:`UiConfigError` so a
        typo cannot silently do nothing — the same contract as
        :func:`undertow.sim.config.load_sim_config`.
        """
        toml_path = Path(path)
        try:
            with open(toml_path, "rb") as fh:
                raw = tomllib.load(fh)
        except FileNotFoundError as exc:
            raise UiConfigError(f"config file not found: {toml_path}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise UiConfigError(f"invalid TOML in {toml_path}: {exc}") from exc

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise UiConfigError(
                f"unknown UI config key(s): {', '.join(unknown)}; "
                f"known keys: {', '.join(sorted(known))}"
            )
        values = {key: _coerce(key, value) for key, value in raw.items()}
        # ``_coerce`` validates each value against its field type; mypy cannot
        # see through the dynamic mapping, so the expansion is declared here.
        return cls(**values)  # type: ignore[arg-type]


def _coerce(key: str, value: object) -> object:
    """Validate and convert one TOML value to its :class:`UiConfig` field type."""
    if key in _PATH_FIELDS or key == "web_dist":
        if not isinstance(value, str):
            raise UiConfigError(f"{key} must be a string path")
        return Path(value)
    if key == "cors_origins":
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(origin, str) for origin in value
        ):
            raise UiConfigError(f"{key} must be a list of origin strings")
        return tuple(value)
    if key == "host":
        if not isinstance(value, str):
            raise UiConfigError(f"{key} must be a string")
        return value
    if key in ("port", "max_points"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise UiConfigError(f"{key} must be an integer")
        return value
    if key == "telemetry_poll_seconds":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UiConfigError(f"{key} must be a number")
        return float(value)
    raise UiConfigError(f"unknown UI config key: {key}")  # pragma: no cover - guarded above
