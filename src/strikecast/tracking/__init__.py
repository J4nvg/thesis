"""Experiment tracking (plan §5.5, phase P4).

The local run store is the source of truth; a tracker only mirrors it, and a
run with :class:`NoopTracker` must produce exactly the same run directory as a
run with :class:`WandbTracker`. Comet ML, which the thesis used, is dropped.

``wandb`` is imported lazily, so importing this package never requires it::

    from strikecast.tracking import TrackerFoldHook, make_tracker

    tracker = make_tracker(cfg.tracking)
    tracker.start(run_key, cfg.model_dump(), tags=(spec.family, spec.kind), stage="cv")
    engine.run(..., hooks=[persist_hook, TrackerFoldHook(tracker, score, prefix="cv/")])
    tracker.log_tables(metric_views, stage="cv")
    tracker.log_artifact(store.predictions_dir(run_key, "cv"), "predictions")
    tracker.finish()
"""

from .factory import make_tracker
from .hook import TrackerFoldHook, numeric_metrics
from .noop import NoopTracker
from .protocol import ARTIFACT_KINDS, Tracker, run_name
from .wandb_tracker import WandbTracker, wandb_run_id

__all__ = [
    "ARTIFACT_KINDS",
    "NoopTracker",
    "Tracker",
    "TrackerFoldHook",
    "WandbTracker",
    "make_tracker",
    "numeric_metrics",
    "run_name",
    "wandb_run_id",
]
