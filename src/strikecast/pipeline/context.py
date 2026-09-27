"""Per-run values and the two interfaces the pipeline borrows from elsewhere.

``RunContext`` is NOT redefined here: it already lives in
:mod:`strikecast.models.spec` (plan §5.2, "``RunContext``: ``seed``, ``device``,
``threads``") and this module only re-exports it, so a builder and a stage
receive the very same class.

Everything else in this module exists because the orchestration layer has to
work *before* the streams that own those modules land:

``get_spec``
    the registry (``strikecast.models.registry``, Stream A) imports every spec
    module and exposes ``get_spec``. Until it exists, :func:`get_spec` imports
    the spec modules itself and falls back to
    :func:`strikecast.models.spec.get_spec`.
``make_tracker`` / :class:`NoopTracker`
    ``strikecast.tracking`` (Stream C) owns the ``Tracker`` protocol and
    ``make_tracker(cfg)``. Until it exists, :func:`make_tracker` returns the
    local :class:`NoopTracker` below, which is a stub of our own and is
    deliberately never exported as *the* tracker: the moment
    ``strikecast.tracking`` is importable, its objects are used instead.

Tracking is a mirror, never a source of truth (§5.5), so :func:`track` swallows
every tracker error: a W&B outage must not fail a run whose predictions are
already on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strikecast.models.spec import RunContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strikecast.config.schema import ExperimentConfig
    from strikecast.models.spec import ModelSpec

__all__ = [
    "NoopTracker",
    "RunContext",
    "get_spec",
    "is_noop_tracker",
    "make_fold_hook",
    "make_run_context",
    "make_tracker",
    "resolve_store_root",
    "track",
    "tracker_tags",
]

logger = logging.getLogger(__name__)

#: The spec modules the registry imports. Listed here only for the fallback
#: path; when ``strikecast.models.registry`` exists it is authoritative.
_SPEC_MODULES = (
    "strikecast.models.gbm",
    "strikecast.models.rnn",
    "strikecast.models.classical",
    "strikecast.models.classifiers",
)

#: Spec modules the registry does NOT import, loaded here in every case (audit
#: C15). ``strikecast.models.chronos`` registers ``chronos2_zero_shot`` and
#: ``chronos2_fine_tuned``; it keeps every AutoGluon import inside functions, so
#: importing it is free in the main environment. Without this, ``make_jobs
#: experiment=chronos2`` and the CLI raised ``KeyError: 'chronos2_zero_shot'``.
_EXTRA_SPEC_MODULES = ("strikecast.models.chronos",)

_specs_loaded = False


def _load_spec_modules() -> None:
    """Import every spec module once, so the registry is populated.

    In ``envs/autogluon`` the registry itself is not importable (it imports
    ``strikecast.models.classifiers``, which needs ``imbalanced-ensemble``);
    the per-module fallback below then registers whatever that env can import,
    which includes both Chronos-2 specs.
    """
    global _specs_loaded
    if _specs_loaded:
        return
    import importlib  # noqa: PLC0415

    try:  # Stream A's registry does this for us, and owns the canonical list.
        importlib.import_module("strikecast.models.registry")
    except ImportError:
        for name in _SPEC_MODULES:
            try:
                importlib.import_module(name)
            except ImportError as exc:  # a family whose deps are absent
                logger.debug("spec module %s not importable: %s", name, exc)
    for name in _EXTRA_SPEC_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover - chronos imports no extra deps
            logger.debug("spec module %s not importable: %s", name, exc)
    _specs_loaded = True


def get_spec(name: str, experiment: str | None = None) -> ModelSpec:
    """Look up a model spec by registry name, preferring Stream A's registry."""
    _load_spec_modules()
    try:
        from strikecast.models.registry import get_spec as _get  # noqa: PLC0415
    except ImportError:
        from strikecast.models.spec import get_spec as _get  # noqa: PLC0415
    return _get(name, experiment)


def make_run_context(
    cfg: ExperimentConfig, model_name: str, seed: int | None = None
) -> RunContext:
    """The :class:`RunContext` a builder takes for one model under one seed.

    Thin wrapper over :meth:`ExperimentConfig.to_run_context` so that stages
    never reach into the config for the device policy themselves (§1, F9).
    """
    return cfg.to_run_context(model_name, seed)


def resolve_store_root(cfg: ExperimentConfig, override: str | Path | None = None) -> Path:
    """The run-store root as an ABSOLUTE path (§9, "Hydra working-directory").

    Hydra is configured with ``hydra.job.chdir=false``, but a relative store
    root would still be resolved against whatever directory the process happens
    to be in. Every path the pipeline writes is therefore anchored here.
    """
    root = Path(override if override is not None else cfg.store.root).expanduser()
    return root.resolve()


class NoopTracker:
    """A tracker that records nothing; the stand-in until Stream C's lands.

    Mirrors ``strikecast.tracking.protocol.Tracker`` exactly (§5.5), so a
    pipeline written against it works unchanged once
    :func:`strikecast.tracking.make_tracker` is importable -- which it is
    whenever ``strikecast.tracking`` exists, in which case THIS class is never
    instantiated.
    """

    name = "noop"

    def start(
        self,
        run_key: Any,
        config: Any,
        tags: Any = (),
        *,
        stage: str = "run",
    ) -> str | None:
        return None

    def log_fold(self, step: int, metrics: Any) -> None:
        return None

    def log_trial(self, number: int, value: float, params: Any) -> None:
        return None

    def log_tables(self, tables: Any, *, stage: str | None = None) -> None:
        return None

    def log_artifact(
        self,
        path: Any,
        kind: str = "artifacts",
        *,
        name: str | None = None,
        metadata: Any = None,
    ) -> None:
        return None

    def finish(self, status: str = "success") -> None:
        return None

    def __enter__(self) -> NoopTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()


def make_tracker(cfg: ExperimentConfig | Any, **kwargs: Any) -> Any:
    """``strikecast.tracking.make_tracker(cfg.tracking)`` when importable.

    Accepts either an :class:`ExperimentConfig` or a bare ``TrackingConfig``.
    A ``noop`` backend short-circuits without importing anything, which is what
    the tests and the CLI's ``tracking=noop`` override rely on.

    ``TrackingConfig.dir`` and ``TrackingConfig.strict`` are keyword arguments
    of ``strikecast.tracking.make_tracker`` rather than fields it reads, so they
    are forwarded here (an explicit ``dir=``/``strict=`` keyword still wins).
    Both default to the values that preserve current behaviour: no ``dir`` is
    passed to ``wandb.init`` and a broken mirror disables itself instead of
    failing the run (§5.5).
    """
    tracking = getattr(cfg, "tracking", cfg)
    if tracking is None or getattr(tracking, "backend", "noop") == "noop":
        return NoopTracker()
    try:
        from strikecast.tracking import make_tracker as _make  # noqa: PLC0415
    except ImportError:
        logger.info("strikecast.tracking is not available; using the no-op tracker")
        return NoopTracker()
    options: dict[str, Any] = {}
    directory = getattr(tracking, "dir", None)
    if directory:
        options["dir"] = directory
    if getattr(tracking, "strict", False):
        options["strict"] = True
    options.update(kwargs)
    try:
        return _make(tracking, **options)
    except Exception as exc:  # pragma: no cover - a tracker must never fail a run
        if options.get("strict"):
            raise
        logger.warning("tracker construction failed (%s); falling back to no-op", exc)
        return NoopTracker()


def tracker_tags(cfg: ExperimentConfig | Any, spec: ModelSpec | Any) -> tuple[str, ...]:
    """``[family, kind]`` from the spec, then ``TrackingConfig.tags`` (§5.5).

    Duplicates are dropped and order is preserved, so adding a tag that the
    spec already supplies changes nothing.
    """
    tags: list[str] = [str(getattr(spec, "family", "")), str(getattr(spec, "kind", ""))]
    tracking = getattr(cfg, "tracking", cfg)
    tags.extend(str(t) for t in (getattr(tracking, "tags", None) or ()))
    seen: dict[str, None] = {}
    for tag in tags:
        if tag:
            seen.setdefault(tag, None)
    return tuple(seen)


def is_noop_tracker(tracker: Any) -> bool:
    """True for a tracker that records nothing, whoever defines it.

    Both this module's stand-in and ``strikecast.tracking.NoopTracker`` are
    no-ops, and the second one is what the pipeline actually gets whenever
    ``strikecast.tracking`` is importable. Recognising both is what keeps
    ``tracking=noop`` from paying for per-fold scoring nobody reads.
    """
    if isinstance(tracker, NoopTracker):
        return True
    return getattr(tracker, "name", None) == "noop" or type(tracker).__name__ == "NoopTracker"


def make_fold_hook(tracker: Any, metrics_fn: Any, *, prefix: str = "", every: int = 1) -> Any:
    """``TrackerFoldHook`` when Stream C is importable, else ``None``.

    ``None`` is a legal absence: the engine simply gets one hook fewer. A no-op
    tracker gets no hook either -- the hook re-scores every fold it mirrors, and
    that cost must not be paid for metrics that go nowhere.
    """
    if is_noop_tracker(tracker):
        return None
    try:
        from strikecast.tracking import TrackerFoldHook  # noqa: PLC0415
    except ImportError:
        return None
    return TrackerFoldHook(tracker, metrics_fn, prefix=prefix, every=every)


def track(tracker: Any, method: str, *args: Any, **kwargs: Any) -> None:
    """Call ``tracker.<method>(...)`` if it exists, swallowing every failure.

    The run store is the source of truth (§5.5); a tracker that is missing a
    method, or that raises because the network is down, must never abort a
    stage whose artefacts are already written.
    """
    if tracker is None:
        return
    fn = getattr(tracker, method, None)
    if fn is None:
        return
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive by design
        logger.warning("tracker.%s failed: %s", method, exc)
