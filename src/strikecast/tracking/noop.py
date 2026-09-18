"""The tracker that does nothing (plan §5.5).

This is the default whenever tracking is switched off, and it is what makes the
"the run store is the source of truth" rule testable: a pipeline run with a
:class:`NoopTracker` must produce byte-identical run-store output to the same
run with a :class:`~strikecast.tracking.wandb_tracker.WandbTracker`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from strikecast.store.run_store import RunKey

__all__ = ["NoopTracker"]


class NoopTracker:
    """A :class:`~strikecast.tracking.protocol.Tracker` that records nothing.

    Every method accepts the full signature and returns ``None``. It keeps no
    state at all, so it cannot drift from the run store.
    """

    def start(
        self,
        run_key: RunKey | Any,
        config: Mapping[str, Any],
        tags: Sequence[str] = (),
        *,
        stage: str = "run",
    ) -> str | None:
        return None

    def log_fold(self, step: int, metrics: Mapping[str, float]) -> None:
        return None

    def log_trial(self, number: int, value: float, params: Mapping[str, Any]) -> None:
        return None

    def log_tables(
        self,
        tables: Mapping[str, Any],
        *,
        stage: str | None = None,
    ) -> None:
        return None

    def log_artifact(
        self,
        path: str | Path,
        kind: str = "artifacts",
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def finish(self, status: str = "success") -> None:
        return None

    # -- convenience ---------------------------------------------------------
    def __enter__(self) -> NoopTracker:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def __repr__(self) -> str:
        return "NoopTracker()"
