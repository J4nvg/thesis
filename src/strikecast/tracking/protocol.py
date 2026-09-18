"""The tracking contract (plan §5.5).

The local run store (:mod:`strikecast.store.run_store`) is the **source of
truth**. A tracker is a *mirror*: it may hold nothing the store does not also
hold, so every method here takes data that has already been, or is about to be,
written under ``runs/``. Dropping the tracker entirely (``NoopTracker``) must
never change what a run produces.

One tracked run corresponds to one run-store run directory, i.e. one
``(experiment, model_variant, paradigm, seed)`` tuple -- exactly
:class:`strikecast.store.run_store.RunKey`. The stage (``cv`` / ``test`` /
``tune``) is the W&B ``job_type``, so the two stages of one run key share a
group and a config but are separate tracked runs, just as they are separate
sub-directories in the store.

Nothing in this module imports ``wandb`` or the run store: the protocol is
structural, and any object with these five methods is a tracker. ``RunKey`` is
only a type hint; at runtime a tracker needs no more than the four attributes
``experiment``, ``model``, ``paradigm`` and ``seed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from strikecast.store.run_store import RunKey

__all__ = ["ARTIFACT_KINDS", "Tracker", "run_name"]

#: The artifact kinds the pipeline mirrors, used as the W&B artifact ``type``.
#: They name what the run store wrote, never a new artefact: ``predictions``
#: are the parquet parts under ``<stage>/predictions/``, ``metrics`` the
#: ``evaluate_long`` views under ``<stage>/metrics/``, ``config`` the
#: ``config.yaml`` snapshot, ``tuning`` the Optuna ``best_params.json`` /
#: ``trials.csv``, and ``artifacts`` whatever ``RunStore.write_artifact`` put
#: under ``artifacts/`` (importances, calibrators, plots).
ARTIFACT_KINDS: tuple[str, ...] = (
    "predictions",
    "metrics",
    "config",
    "tuning",
    "artifacts",
)


def run_name(run_key: RunKey | Any) -> str:
    """The tracked run's name: ``<experiment>/<model>/<paradigm>/seed=<s>``.

    This is exactly ``str(RunKey.relative())``, so a W&B run name and the run
    directory it mirrors are the same string. It is computed from the four
    attributes rather than by calling ``relative()`` so that any object with
    those attributes works, which keeps this package independent of the store.
    """
    return (
        f"{run_key.experiment}/{run_key.model}/{run_key.paradigm}/seed={int(run_key.seed)}"
    )


@runtime_checkable
class Tracker(Protocol):
    """Mirror of one run-store run directory.

    Lifecycle: :meth:`start` once, then any number of :meth:`log_fold`,
    :meth:`log_trial`, :meth:`log_tables` and :meth:`log_artifact` calls, then
    :meth:`finish` exactly once. Calling a log method before :meth:`start` or
    after :meth:`finish` is a no-op, never an error -- a pipeline must not have
    to guard its tracking calls.
    """

    def start(
        self,
        run_key: RunKey | Any,
        config: Mapping[str, Any],
        tags: Sequence[str] = (),
        *,
        stage: str = "run",
    ) -> str | None:
        """Open the tracked run and return its backend id, if any.

        ``config`` is the **resolved** experiment config (§5.5), the same
        mapping ``RunStore.write_config`` snapshots. ``tags`` is
        ``[family, kind]`` from the model spec. ``stage`` becomes the
        ``job_type``. The returned id is what a caller records in the store's
        ``env.json`` so a run directory can be traced to its mirror.
        """
        ...

    def log_fold(self, step: int, metrics: Mapping[str, float]) -> None:
        """Running global metrics after fold ``step`` (``step=Fold.index``)."""
        ...

    def log_trial(self, number: int, value: float, params: Mapping[str, Any]) -> None:
        """One finished Optuna trial: its number, its objective value, its params."""
        ...

    def log_tables(
        self,
        tables: Mapping[str, Any],
        *,
        stage: str | None = None,
    ) -> None:
        """Mirror frames (leaderboard, per-view metrics) as backend tables."""
        ...

    def log_artifact(
        self,
        path: str | Path,
        kind: str = "artifacts",
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Mirror a file or directory the run store already wrote."""
        ...

    def finish(self, status: str = "success") -> None:
        """Close the tracked run. Idempotent."""
        ...
