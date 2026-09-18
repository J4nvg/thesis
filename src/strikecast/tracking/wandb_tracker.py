"""Weights & Biases mirror of the local run store (plan §5.5).

One W&B run per ``(experiment, model_variant, paradigm, seed)`` and stage:

=================  ==========================================================
W&B field          Value
=================  ==========================================================
``name``           ``<experiment>/<model>/<paradigm>/seed=<s>`` -- exactly
                   ``str(RunKey.relative())``, the run-store directory
``group``          the experiment (``TrackingConfig.group`` overrides it)
``job_type``       the stage: ``cv``, ``test`` or ``tune``
``tags``           ``[family, kind]`` from the model spec
``config``         the resolved experiment config, plus the run key and stage
``id``             a deterministic digest of name + stage, so re-running a
                   stage resumes its mirror instead of creating a second one
=================  ==========================================================

Cluster nodes have internet, so runs are **online by default**; ``mode:
offline`` in :class:`~strikecast.config.schema.TrackingConfig` switches to
offline logging, which also exports ``WANDB_MODE=offline`` for anything wandb
spawns. Offline directories are synced afterwards with ``scripts/wandb_sync.sh``.

``wandb`` is imported lazily inside :meth:`WandbTracker.start`, so this module
-- and therefore ``strikecast.tracking`` -- imports fine without wandb
installed. Once a run is open, a failing wandb call never aborts the pipeline:
it is logged at WARNING level and the tracker disables itself, because the run
store already holds everything the tracker was mirroring. ``strict=True``
re-raises instead, which is what the smoke test of §8 P4 uses.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .protocol import run_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from strikecast.store.run_store import RunKey

__all__ = ["WandbTracker", "wandb_run_id"]

logger = logging.getLogger(__name__)

#: Status strings understood by :meth:`WandbTracker.finish`, mapped to the
#: ``exit_code`` wandb records.
_EXIT_CODES: Mapping[str, int] = {"success": 0, "failed": 1, "killed": 1}


def wandb_run_id(name: str, stage: str) -> str:
    """Deterministic W&B run id for one run-store stage.

    W&B ids may not contain ``/``, and we want re-running a stage to resume its
    mirror rather than pile up duplicates, so the id is a digest of the run
    name and the stage. It is derived only from run-store coordinates, which
    keeps the mapping run directory -> W&B run reproducible from disk alone.
    """
    digest = hashlib.sha1(f"{name}#{stage}".encode()).hexdigest()
    return digest[:16]


class WandbTracker:
    """A :class:`~strikecast.tracking.protocol.Tracker` backed by W&B.

    Parameters mirror :class:`~strikecast.config.schema.TrackingConfig`, which
    is what :func:`strikecast.tracking.factory.make_tracker` passes in.
    ``dir`` has no config field today (see the Stream C note in
    ``docs/REFACTOR_PROGRESS.md``); it defaults to ``$WANDB_DIR`` and is the
    scratch directory offline runs are written to on the cluster.
    """

    def __init__(
        self,
        project: str = "strikecast",
        entity: str | None = None,
        mode: str = "online",
        group: str | None = None,
        *,
        dir: str | Path | None = None,
        strict: bool = False,
        resume: str = "allow",
    ) -> None:
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"unknown wandb mode {mode!r}")
        self.project = project
        self.entity = entity
        self.mode = mode
        self.group = group
        self.dir = Path(dir) if dir is not None else None
        self.strict = strict
        self.resume = resume
        self.run: Any | None = None
        self.run_id: str | None = None
        self.name: str | None = None
        self.stage: str | None = None
        self._disabled = False

    def __repr__(self) -> str:
        return (
            f"WandbTracker(project={self.project!r}, entity={self.entity!r}, "
            f"mode={self.mode!r}, run={self.name!r})"
        )

    # -- Tracker -------------------------------------------------------------
    def start(
        self,
        run_key: RunKey | Any,
        config: Mapping[str, Any],
        tags: Sequence[str] = (),
        *,
        stage: str = "run",
    ) -> str | None:
        """Open (or resume) the W&B run that mirrors ``run_key``'s ``stage``."""
        if self.run is not None:
            raise RuntimeError(f"tracker already started for {self.name!r}")
        wandb = self._import_wandb()

        name = run_name(run_key)
        self.name = name
        self.stage = stage
        self.run_id = wandb_run_id(name, stage)

        if self.mode == "offline":
            # Exported as well as passed, so that anything wandb spawns (and a
            # SLURM epilogue reading the environment) sees the same decision.
            os.environ["WANDB_MODE"] = "offline"

        init_kwargs: dict[str, Any] = {
            "project": self.project,
            "entity": self.entity,
            "name": name,
            "id": self.run_id,
            "group": self.group or run_key.experiment,
            "job_type": stage,
            "tags": list(tags),
            "config": self._config_payload(run_key, config, stage),
            "mode": self.mode,
            "resume": self.resume,
        }
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            init_kwargs["dir"] = str(self.dir)

        try:
            self.run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover - exercised via strict=True
            if self.strict:
                raise
            logger.warning("wandb.init failed for %s (%s); tracking disabled", name, exc)
            self._disabled = True
            return None
        return self.run_id

    def log_fold(self, step: int, metrics: Mapping[str, float]) -> None:
        """Running global metrics after fold ``step``, logged at ``step=fold``."""
        if not self._active() or not metrics:
            return
        payload = {str(k): v for k, v in metrics.items()}
        payload.setdefault("fold", int(step))
        self._guard("log_fold", lambda: self.run.log(payload, step=int(step)))

    def log_trial(self, number: int, value: float, params: Mapping[str, Any]) -> None:
        """One finished Optuna trial. Trials live on the ``tune`` run, whose
        step axis is the trial number, so it never collides with ``log_fold``."""
        if not self._active():
            return
        payload: dict[str, Any] = {
            "trial/number": int(number),
            "trial/value": float(value),
        }
        payload.update({f"trial/params/{k}": v for k, v in params.items()})
        self._guard("log_trial", lambda: self.run.log(payload, step=int(number)))

    def log_tables(
        self,
        tables: Mapping[str, Any],
        *,
        stage: str | None = None,
    ) -> None:
        """Mirror frames as ``wandb.Table``s, keyed ``<stage>/<name>``.

        Values are dataframes (the ``evaluate_long`` views, a leaderboard).
        Anything that is already a ``wandb`` object is passed through.
        """
        if not self._active() or not tables:
            return
        wandb = self._import_wandb()
        prefix = f"{stage}/" if stage else ""
        payload: dict[str, Any] = {}
        for key, value in tables.items():
            if value is None:
                continue
            if hasattr(value, "columns") and hasattr(value, "to_dict"):
                value = wandb.Table(dataframe=value)
            payload[f"{prefix}{key}"] = value
        if payload:
            self._guard("log_tables", lambda: self.run.log(payload))

    def log_artifact(
        self,
        path: str | Path,
        kind: str = "artifacts",
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Mirror a file or directory that the run store already wrote."""
        if not self._active():
            return
        wandb = self._import_wandb()
        target = Path(path)
        if not target.exists():
            logger.warning("wandb artifact %s does not exist; not logged", target)
            return
        artifact_name = name or self._artifact_name(target, kind)

        def _log() -> None:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=kind,
                metadata=dict(metadata) if metadata else None,
            )
            if target.is_dir():
                artifact.add_dir(str(target))
            else:
                artifact.add_file(str(target))
            self.run.log_artifact(artifact)

        self._guard("log_artifact", _log)

    def finish(self, status: str = "success") -> None:
        """Close the W&B run. Idempotent, and safe after a failed start."""
        run, self.run = self.run, None
        if run is None:
            return
        exit_code = _EXIT_CODES.get(status, 1)
        try:
            run.finish(exit_code=exit_code)
        except Exception as exc:  # pragma: no cover - exercised via strict=True
            if self.strict:
                raise
            logger.warning("wandb finish failed for %s (%s)", self.name, exc)

    # -- convenience ---------------------------------------------------------
    def __enter__(self) -> WandbTracker:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.finish("success" if exc_type is None else "failed")

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _import_wandb() -> Any:
        try:
            import wandb  # noqa: PLC0415 -- lazy: strikecast imports without wandb
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "tracking.backend='wandb' needs the wandb package; install it or "
                "set tracking.backend='noop'"
            ) from exc
        return wandb

    def _config_payload(
        self,
        run_key: RunKey | Any,
        config: Mapping[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        """Resolved config plus the run-store coordinates, for W&B filtering.

        The four coordinates and the stage are already in the run directory's
        path and ``config.yaml``, so this adds no state the store lacks.
        """
        payload = dict(config)
        payload["run"] = {
            "experiment": run_key.experiment,
            "model": run_key.model,
            "paradigm": run_key.paradigm,
            "seed": int(run_key.seed),
            "stage": stage,
            "dir": run_name(run_key),
        }
        return payload

    def _artifact_name(self, target: Path, kind: str) -> str:
        """W&B artifact names allow ``[A-Za-z0-9_.-]`` only, so the run name's
        separators are flattened rather than dropped."""
        base = (self.name or "run").replace("/", "__").replace("=", "-")
        return f"{base}__{self.stage or 'run'}__{kind}"

    def _active(self) -> bool:
        return self.run is not None and not self._disabled

    def _guard(self, what: str, fn: Any) -> None:
        """Run a wandb call; on failure warn and disable, unless ``strict``.

        Tracking is a mirror, so a broken mirror must not take the run with it.
        """
        try:
            fn()
        except Exception as exc:
            if self.strict:
                raise
            logger.warning("wandb %s failed for %s (%s); tracking disabled", what, self.name, exc)
            self._disabled = True
