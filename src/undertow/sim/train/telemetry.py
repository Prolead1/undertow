"""Training telemetry sidecars (U00, ``CONTRACTS.md`` §1).

The dashboard's Live Cockpit needs a machine-readable view of training that it
can tail *while a run is in progress*. This module adds two sidecars next to the
existing manifest/checkpoint artifacts:

* ``telemetry.jsonl`` — one :class:`dict` per ``log_freq_steps`` timesteps,
  appended in a single ``write()`` of ``json.dumps(point) + "\\n"`` so a
  concurrent reader never observes a torn line.
* ``status.json`` — the run's lifecycle state (``running`` → ``done``/
  ``failed``), written atomically with a temp file + :func:`os.replace`.

Nothing here changes training behaviour or any pre-existing artifact format; the
callback only observes what SB3 already computes in
``model.logger.name_to_value`` and the model's episode info buffer.

Metric extraction is deliberately a straight prefix strip: keys under
``train/`` / ``time/`` become ``point["train"]`` / ``point["time"]`` entries
with the prefix removed, unknown keys are preserved, and missing keys are
omitted (the reader treats them as ``null``). The one additive exception is
SB3's ``train/policy_gradient_loss``, which is also mirrored to the wire
schema's ``policy_loss`` field (see ``_TRAIN_KEY_ALIASES``). See
``CONTRACTS.md`` §1.1.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

__all__ = [
    "SCHEMA_VERSION",
    "STATUS_FILENAME",
    "TELEMETRY_FILENAME",
    "RunStatus",
    "TelemetryCallback",
    "write_status",
]

#: Sidecar schema version (``CONTRACTS.md`` §0: every artifact carries this).
SCHEMA_VERSION: int = 1

#: Filename of the newline-delimited telemetry stream (one point per line).
TELEMETRY_FILENAME: str = "telemetry.jsonl"

#: Filename of the atomically-written run status.
STATUS_FILENAME: str = "status.json"

_TRAIN_PREFIX: str = "train/"
_TIME_PREFIX: str = "time/"

#: SB3 v2 records PPO's policy loss as ``train/policy_gradient_loss`` while the
#: frozen wire schema names the field ``policy_loss`` (CONTRACTS.md §1.1/§4).
#: The raw key is preserved and the canonical alias is added additively.
_TRAIN_KEY_ALIASES: dict[str, str] = {"policy_gradient_loss": "policy_loss"}


@dataclass(frozen=True, slots=True)
class RunStatus:
    """Lifecycle state of a single training run (``CONTRACTS.md`` §1.2)."""

    schema_version: int
    run_id: str
    seed: int
    state: Literal["running", "done", "failed"]
    timestep: int
    total_timesteps: int
    updated_at: datetime
    pid: int
    error: str | None = None


def _iso_z(moment: datetime) -> str:
    """Render ``moment`` as an ISO-8601 UTC string ending in ``Z``."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _status_payload(status: RunStatus) -> dict[str, Any]:
    """Convert a :class:`RunStatus` to its JSON-serialisable payload."""
    return {
        "schema_version": status.schema_version,
        "run_id": status.run_id,
        "seed": status.seed,
        "state": status.state,
        "timestep": status.timestep,
        "total_timesteps": status.total_timesteps,
        "updated_at": _iso_z(status.updated_at),
        "pid": status.pid,
        "error": status.error,
    }


def write_status(path: Path, status: RunStatus) -> None:
    """Atomically write ``status`` to ``path`` (``CONTRACTS.md`` §1.2).

    The body is written to a temporary file in the *same* directory and then
    :func:`os.replace`-d into place, so a reader only ever sees a complete
    previous or complete new file. Keys are sorted and the file ends in a
    trailing newline.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_status_payload(status), sort_keys=True) + "\n"

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _to_jsonable(value: object) -> object:
    """Convert numpy scalars/arrays to strict-JSON-serialisable Python objects.

    Non-finite floats (SB3 reports ``explained_variance=NaN`` before the first
    update) are mapped to ``None``: ``NaN``/``Infinity`` are not valid strict
    JSON and a browser's ``JSON.parse`` would reject the line. Arrays and
    sequences are converted element-wise so a non-finite entry *inside* a numpy
    array is neutralised too, rather than surviving ``ndarray.tolist()`` and
    being emitted by :func:`json.dumps` as a bare ``NaN``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_jsonable(value.item())
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _episode_metrics(model: object) -> tuple[float | None, float | None]:
    """Windowed mean reward and mean length from the info buffer.

    ``None`` means "not yet known" (no episode has finished): the empty-buffer
    case must not be reported as ``0.0`` because zero is a legitimate reward.
    The *cumulative* episode count is tracked separately by the callback, since
    ``ep_info_buffer`` saturates at SB3's ``maxlen`` (100) and would undercount
    a long run.
    """
    buffer = getattr(model, "ep_info_buffer", None)
    if not buffer:
        return None, None
    mean_reward = float(np.mean([ep["r"] for ep in buffer]))
    mean_length = float(np.mean([ep["l"] for ep in buffer]))
    return mean_reward, mean_length


class TelemetryCallback(BaseCallback):
    """Append a telemetry point and refresh ``status.json`` every N timesteps.

    One point is emitted per ``log_freq_steps`` and the status heartbeat runs at
    the same cadence. ``_on_training_end`` writes a terminal ``done`` status; the
    trainer re-writes it after the manifest/checkpoint exists so that
    ``state="done"`` implies the run's artifacts are on disk.
    """

    def __init__(
        self,
        run_id: str,
        seed: int,
        total_timesteps: int,
        log_freq_steps: int,
        status_path: Path,
        telemetry_path: Path,
    ) -> None:
        super().__init__(verbose=0)
        self.run_id = run_id
        self.seed = int(seed)
        self.total_timesteps = int(total_timesteps)
        self.log_freq_steps = max(1, int(log_freq_steps))
        self.status_path = Path(status_path)
        self.telemetry_path = Path(telemetry_path)
        self._next_eval = self.log_freq_steps
        self._start_time = time.monotonic()
        # SB3's ``Logger.dump`` clears ``name_to_value`` at the end of every
        # logging iteration, and it dumps *after* the rollout callback, so the
        # ``time/*`` metrics are never observable from a callback. We shadow
        # them by wrapping ``logger.record`` (calling through unchanged).
        self._train_cache: dict[str, Any] = {}
        self._time_cache: dict[str, Any] = {}
        self._original_record: Any | None = None
        # Cumulative count of completed episodes. ``model.ep_info_buffer`` is a
        # windowed ``deque(maxlen=100)``; counting finished envs instead keeps
        # this monotonic over an arbitrarily long run.
        self._episode_count: int = 0

    # -- logging shadow -----------------------------------------------------

    def _cached_record(self, key: str, value: object, exclude: object = None) -> None:
        """Mirror ``Logger.record`` into a cache that survives ``dump()``."""
        # A real runtime check, not ``assert``: assertions are stripped under
        # ``python -O`` and this must still fail loudly if the wrapper is ever
        # installed without ``_init_callback`` having run.
        if self._original_record is None:
            raise RuntimeError("TelemetryCallback.record called before _init_callback")
        self._original_record(key, value, exclude)
        if key.startswith(_TRAIN_PREFIX):
            self._train_cache[key[len(_TRAIN_PREFIX) :]] = _to_jsonable(value)
        elif key.startswith(_TIME_PREFIX):
            self._time_cache[key[len(_TIME_PREFIX) :]] = _to_jsonable(value)

    def _init_callback(self) -> None:
        logger = self.model.logger
        if not hasattr(logger, "_undertow_original_record"):
            setattr(logger, "_undertow_original_record", logger.record)  # noqa: B010
        self._original_record = getattr(logger, "_undertow_original_record")  # noqa: B009
        setattr(logger, "record", self._cached_record)  # noqa: B010

    # -- status helpers -----------------------------------------------------

    def make_status(
        self,
        state: Literal["running", "done", "failed"],
        *,
        error: str | None = None,
    ) -> RunStatus:
        """Build a :class:`RunStatus` for the current timestep."""
        return RunStatus(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            seed=self.seed,
            state=state,
            timestep=int(self.num_timesteps),
            total_timesteps=self.total_timesteps,
            updated_at=datetime.now(UTC),
            pid=os.getpid(),
            error=error,
        )

    def write_running_status(self) -> None:
        """Write a ``running`` heartbeat at the current timestep."""
        write_status(self.status_path, self.make_status("running"))

    def write_done_status(self) -> None:
        """Write the terminal ``done`` status at the current timestep."""
        write_status(self.status_path, self.make_status("done"))

    def write_failed_status(self, error: str) -> None:
        """Write the terminal ``failed`` status with ``error``."""
        write_status(self.status_path, self.make_status("failed", error=error))

    # -- telemetry point ----------------------------------------------------

    def _build_point(self) -> dict[str, Any]:
        """Build the JSON object for the current timestep."""
        ep_rew_mean, ep_len_mean = _episode_metrics(self.model)
        train: dict[str, Any] = dict(self._train_cache)
        timing: dict[str, Any] = dict(self._time_cache)
        # ``_train_cache``/``_time_cache`` shadow the last ``logger.record``
        # calls, so a value can lag ``name_to_value`` by one SB3 iteration; the
        # fresh values read below take precedence when present.
        logger = getattr(getattr(self, "model", None), "logger", None)
        metrics = getattr(logger, "name_to_value", {})
        for key, value in metrics.items():
            if key.startswith(_TRAIN_PREFIX):
                train[key[len(_TRAIN_PREFIX) :]] = _to_jsonable(value)
            elif key.startswith(_TIME_PREFIX):
                timing[key[len(_TIME_PREFIX) :]] = _to_jsonable(value)
        for source, alias in _TRAIN_KEY_ALIASES.items():
            if source in train and alias not in train:
                train[alias] = train[source]
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "timestep": int(self.num_timesteps),
            "wall_time_s": time.monotonic() - self._start_time,
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            # ``None`` (not ``0``) until the first episode finishes, matching
            # the contract's "null until known".
            "episodes": self._episode_count or None,
            "train": train,
            "time": timing,
        }

    def _write_point(self) -> None:
        """Append the current point as one complete JSON line."""
        point = self._build_point()
        # One ``write`` of the whole line + newline: a concurrent reader never
        # sees a torn line (CONTRACTS.md §0 "JSON discipline").
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            # ``allow_nan=False`` makes any non-finite value that slipped past
            # ``_to_jsonable`` fail loudly instead of emitting invalid JSON.
            handle.write(json.dumps(point, sort_keys=True, allow_nan=False) + "\n")

    # -- SB3 callback hooks -------------------------------------------------

    def _on_step(self) -> bool:
        # SB3 exposes each finished env's info dict in ``locals()["infos"]``;
        # ``VecMonitor`` adds an ``info["episode"]`` key exactly when that env
        # completed an episode. Count those so the reported total keeps rising
        # past the model's 100-episode rolling buffer.
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_count += 1
        # A ``while`` (not ``if``) coalesces any skipped multiples in the same
        # callback invocation into separate point writes, matching
        # ``_LearningCurveCallback``.
        while self.num_timesteps >= self._next_eval:
            self._write_point()
            self.write_running_status()
            self._next_eval += self.log_freq_steps
        return True

    def _restore_logger_record(self) -> None:
        """Undo the ``logger.record`` wrapper installed by ``_init_callback``.

        Without this the patched bound method keeps the callback (and through
        it the model) alive after training finishes.
        """
        logger = getattr(getattr(self, "model", None), "logger", None)
        if logger is not None:
            original = self._original_record or getattr(
                logger, "_undertow_original_record", None
            )
            if original is not None:
                setattr(logger, "record", original)  # noqa: B010
            if hasattr(logger, "_undertow_original_record"):
                delattr(logger, "_undertow_original_record")
        self._original_record = None

    def _on_training_end(self) -> None:
        self._restore_logger_record()
        # The trainer re-writes the terminal status once the manifest and
        # checkpoint exist (so ``done`` implies artifacts). This callback-level
        # ``done`` is deliberately redundant: it covers the case where a
        # trainer never reaches the post-``learn`` bookkeeping.
        self.write_done_status()
