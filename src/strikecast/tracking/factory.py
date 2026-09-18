"""``make_tracker``: the one place a :class:`TrackingConfig` becomes a tracker.

Keeping the mapping here means the pipeline never branches on
``cfg.tracking.backend``; it asks for a tracker and gets one. ``backend:
"noop"`` (and an unreachable/misconfigured wandb) yields a
:class:`~strikecast.tracking.noop.NoopTracker`, which changes nothing about
what the run store records.

:class:`~strikecast.config.schema.TrackingConfig` is duck-typed rather than
imported, so this package does not depend on the config package (which another
stream owns) and any object with ``backend``, ``mode``, ``project``, ``entity``
and ``group`` works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .noop import NoopTracker
from .wandb_tracker import WandbTracker

if TYPE_CHECKING:
    from pathlib import Path

    from strikecast.config.schema import TrackingConfig

    from .protocol import Tracker

__all__ = ["make_tracker"]


def make_tracker(
    cfg: TrackingConfig | Any | None = None,
    *,
    dir: str | Path | None = None,
    strict: bool = False,
) -> Tracker:
    """Build the tracker ``cfg`` describes.

    ``cfg=None`` -- tracking not configured -- is a :class:`NoopTracker`.
    ``dir`` is the directory offline runs are written to (cluster scratch);
    ``TrackingConfig`` has no field for it yet, so the CLI passes it in.
    """
    if cfg is None:
        return NoopTracker()
    backend = getattr(cfg, "backend", "noop")
    if backend == "noop":
        return NoopTracker()
    if backend != "wandb":
        raise ValueError(f"unknown tracking backend {backend!r}")
    return WandbTracker(
        project=getattr(cfg, "project", "strikecast"),
        entity=getattr(cfg, "entity", None),
        mode=getattr(cfg, "mode", "online"),
        group=getattr(cfg, "group", None),
        dir=dir,
        strict=strict,
    )
