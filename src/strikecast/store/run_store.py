"""The local run store: the source of truth for everything a run produces.

Plan §5.3. One directory tree under ``root`` holds every experiment, and W&B
(P4) only ever mirrors what is written here. Nothing in this module pickles
anything: predictions are parquet, metrics are JSON and CSV, config is YAML,
state is JSON.

Layout, exactly as ``docs/REFACTOR_PLAN.md`` §5.3 draws it::

    runs/
      <experiment>/
        shared/
          panel.<hash>.parquet
          series.<hash>/                       SeriesBundle (parquet + manifest.json)
          feature_selection.<hash>.json
        tuning/<model_variant>/
          optuna.sqlite3  best_params.json  trials.csv
        <model_variant>/<paradigm>/seed=<s>/
          config.yaml  env.json  state.json
          cv/predictions/part-<fold_from>-<fold_to>.parquet
          cv/metrics/{global.json, per_region.csv, per_horizon.csv,
                      per_region_horizon.csv, per_activity_level.csv,
                      per_activity_horizon.csv}
          test/...
          artifacts/                           importances, calibrators.json, plots
        report/                                cross-seed aggregates (§7)

The metric file names are the keys of the legacy
``src/evaluation_tools.py::evaluate_long`` result dict, so a stored view can be
compared with a legacy one by name.

Resume (§5.3, §8 P3)
--------------------
A stage is identified by :func:`stage_hash` over the resolved stage config, the
upstream hashes it consumes and the seed. :meth:`RunStore.is_complete` skips a
finished stage. A partial stage is resumable because
:class:`PersistHook` flushes a prediction part at every retrain boundary: the
folds of the window that was in flight when the process died are the only ones
lost. :meth:`RunStore.resume_point` returns the fold index those parts end at,
which is always a retrain boundary and therefore a legal restart point for the
expanding-window schedule.

The engine does **not** support starting mid-schedule today
(``ExpandingWindowBacktest.iter_folds`` always runs the full schedule), so the
number is informational: the P3 CLI either re-runs the stage from fold 0 or,
once the engine grows a ``start_fold`` argument, restarts at
:meth:`resume_point` and recomputes **at most one retrain window** (six folds
with the legacy ``predict_stride=1``/``retrain_stride=7``).

Atomic writes
-------------
Every write goes to a temporary file in the destination directory and is then
``os.replace``'d into place, the same rule
``strikecast.data.cache.cached_parquet`` and ``PredictionSet.to_parquet``
follow. Unlike those two, the helpers here delete the temporary file if the
write raises, so a crash mid-write leaves neither a partial part nor a stray
``.tmp`` beside the finished ones.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from strikecast.backtest.predictions import COLUMNS, PredictionSet
from strikecast.data.cache import content_hash

if TYPE_CHECKING:
    from darts import TimeSeries

    from strikecast.backtest.protocols import FoldResult

__all__ = [
    "METRIC_VIEWS",
    "PartWriter",
    "PersistHook",
    "RunKey",
    "RunStore",
    "StageState",
    "Status",
    "stage_hash",
]

Status = Literal["pending", "running", "complete", "failed"]

#: The ``evaluate_long`` views, in the order §5.3 lists their files. ``global``
#: is a single row and is written as JSON; the rest are frames written as CSV.
METRIC_VIEWS: tuple[str, ...] = (
    "global",
    "per_region",
    "per_horizon",
    "per_region_horizon",
    "per_activity_level",
    "per_activity_horizon",
)

#: Default file extension per ``shared`` artefact kind (§5.3). ``series`` is a
#: directory (parquet files plus ``manifest.json``), hence the empty string.
_SHARED_EXT: Mapping[str, str] = {
    "panel": ".parquet",
    "series": "",
    "feature_selection": ".json",
}

_PART_RE = re.compile(r"^part-(\d+)-(\d+)\.parquet$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+=-]*$")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    """UTC timestamp, second resolution, ISO-8601 with a ``Z`` suffix."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-safe primitives.

    Mirrors ``scripts/convert_legacy_artifacts.py::jsonable`` so that a metric
    row written here and one converted from a legacy pickle look the same.
    """
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    for attr in ("item", "isoformat"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:  # best effort: fall through to str()
                pass
    return str(obj)


def _atomic_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically, removing the temp file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    return _atomic_text(path, json.dumps(_jsonable(payload), indent=2) + "\n")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    """Write ``frame`` to ``path`` atomically, removing the temp file on failure.

    A crash inside ``to_parquet`` leaves the destination untouched and no
    ``.tmp`` file behind, so a resumed run sees exactly the parts that were
    completed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    return _atomic_text(path, frame.to_csv(index=False))


def _check_name(name: str, what: str) -> str:
    if not _SAFE_NAME.match(name) or "/" in name or "\\" in name:
        raise ValueError(f"unsafe {what} {name!r}: must be a plain path segment")
    return name


def stage_hash(
    stage_config: Mapping[str, Any],
    upstream: Sequence[str] = (),
    seed: int | None = None,
) -> str:
    """Stage identity: ``content_hash(resolved stage config, upstream hashes, seed)``.

    ``stage_config`` is the *resolved* config of this stage (backtest window,
    model parameters, transform, paradigm -- whatever the CLI resolved), which
    hashes order-insensitively through canonical JSON. ``upstream`` are the
    hashes of the shared artefacts the stage consumes (panel, series bundle,
    feature selection), in a fixed order. ``seed`` is the evaluation seed of
    §5.4, so the same stage under two seeds gets two identities and both are
    kept.
    """
    return content_hash(dict(stage_config), list(upstream), seed)


# --------------------------------------------------------------------------- #
# keys and state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RunKey:
    """The four coordinates that name one run directory (§5.3, §5.4)."""

    experiment: str
    model: str
    paradigm: str
    seed: int

    def relative(self) -> Path:
        return Path(
            _check_name(self.experiment, "experiment"),
            _check_name(self.model, "model"),
            _check_name(self.paradigm, "paradigm"),
            f"seed={int(self.seed)}",
        )


@dataclass
class StageState:
    """``state.json``: what a stage has durably produced so far.

    ``folds_done``
        Number of folds whose predictions are on disk, i.e. ``parts[-1].to + 1``
        (folds are 0-based and contiguous). Folds computed but not yet flushed
        do not count.
    ``last_retrain_fold``
        Index of the retrain fold that opened the part currently being filled,
        i.e. the first fold of the next part. ``None`` before the first flush
        and after the final one.
    ``parts``
        One entry per written part: ``{"path", "from", "to", "rows"}`` with
        ``path`` relative to the stage's ``predictions`` directory.
    """

    stage: str
    status: Status = "pending"
    stage_hash: str | None = None
    folds_done: int = 0
    last_retrain_fold: int | None = None
    parts: list[dict[str, Any]] = field(default_factory=list)
    started: str | None = None
    finished: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageState:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"state.json has unknown fields {sorted(unknown)}")
        return cls(**{k: v for k, v in payload.items() if k in known})


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
class RunStore:
    """Directory layout, config/state snapshots, prediction parts and metrics.

    Every method takes ``run`` either as a :class:`RunKey` or as a run
    directory :class:`~pathlib.Path` (what :meth:`run_dir` returned), so a
    caller that already holds the path never has to rebuild the key.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"RunStore({str(self.root)!r})"

    # -- layout -------------------------------------------------------------
    def experiment_dir(self, name: str) -> Path:
        return self.root / _check_name(name, "experiment")

    def shared_path(
        self, experiment: str, kind: str, hash: str, ext: str | None = None
    ) -> Path:
        """``<experiment>/shared/<kind>.<hash><ext>``.

        ``ext`` defaults to the §5.3 extension of ``kind`` (``.parquet`` for
        ``panel``, ``.json`` for ``feature_selection``, none for ``series``,
        which is a directory). An unknown ``kind`` without an explicit ``ext``
        gets no extension, i.e. it is treated as a directory.
        """
        suffix = _SHARED_EXT.get(kind, "") if ext is None else ext
        _check_name(kind, "shared kind")
        _check_name(hash, "hash")
        return self.experiment_dir(experiment) / "shared" / f"{kind}.{hash}{suffix}"

    def tuning_dir(self, experiment: str, model: str) -> Path:
        return self.experiment_dir(experiment) / "tuning" / _check_name(model, "model")

    def report_dir(self, experiment: str) -> Path:
        return self.experiment_dir(experiment) / "report"

    def run_dir(self, experiment: str, model: str, paradigm: str, seed: int) -> Path:
        return self.root / RunKey(experiment, model, paradigm, seed).relative()

    def resolve(self, run: RunKey | str | Path) -> Path:
        """The run directory for a :class:`RunKey` or an existing path."""
        if isinstance(run, RunKey):
            return self.root / run.relative()
        return Path(run)

    def stage_dir(self, run: RunKey | str | Path, stage: str) -> Path:
        return self.resolve(run) / _check_name(stage, "stage")

    def predictions_dir(self, run: RunKey | str | Path, stage: str) -> Path:
        return self.stage_dir(run, stage) / "predictions"

    def metrics_dir(self, run: RunKey | str | Path, stage: str) -> Path:
        return self.stage_dir(run, stage) / "metrics"

    def artifacts_dir(self, run: RunKey | str | Path) -> Path:
        return self.resolve(run) / "artifacts"

    # -- config and environment ---------------------------------------------
    def write_config(self, run: RunKey | str | Path, cfg_dict: Mapping[str, Any]) -> Path:
        """Snapshot the resolved config as ``config.yaml``.

        Written with ``yaml.safe_dump(sort_keys=False)`` so the resolved
        pydantic order survives; values go through the same JSON-safe
        conversion the rest of the store uses, so a ``Path`` or a numpy scalar
        cannot make the file unreadable.
        """
        import yaml  # noqa: PLC0415 -- only needed when a config is actually written

        text = yaml.safe_dump(_jsonable(dict(cfg_dict)), sort_keys=False, allow_unicode=True)
        return _atomic_text(self.resolve(run) / "config.yaml", text)

    def read_config(self, run: RunKey | str | Path) -> dict[str, Any]:
        import yaml  # noqa: PLC0415

        return yaml.safe_load((self.resolve(run) / "config.yaml").read_text(encoding="utf-8"))

    def write_env(self, run: RunKey | str | Path, env_dict: Mapping[str, Any]) -> Path:
        """Snapshot the environment as ``env.json``.

        The caller decides what goes in; F16 requires at least ``PYTHONHASHSEED``
        to be recorded, because the legacy covariate column order depends on it.
        """
        return _atomic_json(self.resolve(run) / "env.json", dict(env_dict))

    def read_env(self, run: RunKey | str | Path) -> dict[str, Any]:
        return json.loads((self.resolve(run) / "env.json").read_text(encoding="utf-8"))

    # -- stage state --------------------------------------------------------
    def state_path(self, run: RunKey | str | Path) -> Path:
        return self.resolve(run) / "state.json"

    def read_states(self, run: RunKey | str | Path) -> dict[str, StageState]:
        """All stage states of a run, keyed by stage name (empty if none yet)."""
        path = self.state_path(run)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {name: StageState.from_dict(body) for name, body in payload.items()}

    def read_state(self, run: RunKey | str | Path, stage: str) -> StageState | None:
        return self.read_states(run).get(stage)

    def write_state(self, run: RunKey | str | Path, state: StageState) -> Path:
        """Replace one stage's entry in ``state.json``, keeping the others."""
        states = self.read_states(run)
        states[state.stage] = state
        payload = {name: s.to_dict() for name, s in states.items()}
        return _atomic_json(self.state_path(run), payload)

    def start_stage(
        self,
        run: RunKey | str | Path,
        stage: str,
        stage_hash: str | None = None,
    ) -> StageState:
        """Mark a stage ``running``.

        A stage whose recorded hash differs from ``stage_hash`` is a different
        stage under the same name, so its parts are discarded and the state
        starts over. This is the only place the store deletes anything.
        """
        state = self.read_state(run, stage)
        if state is not None and stage_hash is not None and state.stage_hash != stage_hash:
            self._clear_parts(run, stage)
            state = None
        if state is None:
            state = StageState(stage=stage, stage_hash=stage_hash)
        state.status = "running"
        state.stage_hash = stage_hash if stage_hash is not None else state.stage_hash
        state.started = state.started or _now()
        state.finished = None
        state.error = None
        self.write_state(run, state)
        return state

    def complete_stage(self, run: RunKey | str | Path, stage: str) -> StageState:
        state = self.read_state(run, stage) or StageState(stage=stage)
        state.status = "complete"
        state.last_retrain_fold = None
        state.finished = _now()
        state.error = None
        self.write_state(run, state)
        return state

    def fail_stage(self, run: RunKey | str | Path, stage: str, error: str) -> StageState:
        state = self.read_state(run, stage) or StageState(stage=stage)
        state.status = "failed"
        state.finished = _now()
        state.error = error
        self.write_state(run, state)
        return state

    def is_complete(
        self,
        run: RunKey | str | Path,
        stage: str,
        stage_hash: str | None = None,
    ) -> bool:
        """True when the stage finished under the same identity (§5.3: "skip")."""
        state = self.read_state(run, stage)
        if state is None or state.status != "complete":
            return False
        if stage_hash is None:
            return True
        return state.stage_hash == stage_hash

    def resume_point(self, run: RunKey | str | Path, stage: str) -> int | None:
        """Fold index to restart at, or ``None`` when there is nothing to resume.

        The value is ``last written fold + 1``. Parts are flushed only when the
        next retrain fold begins (:class:`PersistHook`), so that index is always
        a retrain boundary and therefore a legal start of an expanding window:
        restarting there refits on exactly the data the crashed run refit on.

        Returns ``None`` when the stage is complete, absent, or has no part yet.
        The engine cannot start mid-schedule today, so the P3 CLI uses this
        number to decide how much work a resume saves; with the legacy
        ``retrain_stride=7`` a resumed run recomputes at most one retrain
        window, the six folds that were in flight.
        """
        state = self.read_state(run, stage)
        if state is None or state.status == "complete" or not state.parts:
            return None
        return int(state.parts[-1]["to"]) + 1

    def _clear_parts(self, run: RunKey | str | Path, stage: str) -> None:
        directory = self.predictions_dir(run, stage)
        if not directory.exists():
            return
        for path in directory.iterdir():
            if _PART_RE.match(path.name) or path.name.endswith(".tmp"):
                path.unlink()

    # -- predictions --------------------------------------------------------
    def part_writer(self, run: RunKey | str | Path, stage: str) -> PartWriter:
        return PartWriter(self, run, stage)

    def persist_hook(
        self,
        run: RunKey | str | Path,
        stage: str,
        retrain_stride: int | None = 7,
        *,
        actuals: Sequence[TimeSeries],
        region_names: Sequence[str],
    ) -> PersistHook:
        return PersistHook(
            self, run, stage, retrain_stride, actuals=actuals, region_names=region_names
        )

    def part_paths(self, run: RunKey | str | Path, stage: str) -> list[Path]:
        """Existing parts, ordered by their first fold.

        Read from the directory, not from ``state.json``, so a part that landed
        just before a crash is picked up even if the state write did not.
        """
        directory = self.predictions_dir(run, stage)
        if not directory.exists():
            return []
        found: list[tuple[int, Path]] = []
        for path in directory.iterdir():
            match = _PART_RE.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return [path for _, path in sorted(found, key=lambda item: item[0])]

    def load_predictions(
        self,
        run: RunKey | str | Path,
        stage: str,
        *,
        legacy_order: bool = False,
    ) -> PredictionSet:
        """Concatenate the stage's parts, in part order, into one PredictionSet.

        Row order is part-major, then the legacy region-major/fold/horizon order
        *within* a part. That is NOT the legacy ``collect_predictions_long``
        order, which is region-major across the whole run; a part boundary cuts
        every region's block. ``legacy_order=True`` restores it with a stable
        sort on ``(channel, region, fold, horizon)``, which is what a level-E
        comparison against a stored ``predictions_long_*.parquet`` needs.
        Every consumer in ``evaluation`` only groups, so the default order is
        fine for metrics.
        """
        paths = self.part_paths(run, stage)
        if not paths:
            # `concat` of nothing is PredictionSet's own typed empty frame.
            return PredictionSet.concat([])
        combined = PredictionSet.concat(PredictionSet.from_parquet(p) for p in paths)
        if not legacy_order:
            return combined
        frame = combined.frame.sort_values(
            ["channel", "region", "fold", "horizon"], kind="stable"
        ).reset_index(drop=True)
        return PredictionSet(frame)

    # -- metrics and artifacts ----------------------------------------------
    def write_metrics(
        self,
        run: RunKey | str | Path,
        stage: str,
        views: Mapping[str, Any],
    ) -> dict[str, Path]:
        """Write the ``evaluate_long`` views under ``<stage>/metrics/``.

        ``views`` is the dict ``evaluate_long`` returns: ``global`` (one row,
        a dict or a Series) goes to ``global.json``; every other key is a frame
        and goes to ``<key>.csv`` with the legacy file name (``per_region.csv``,
        ``per_horizon.csv``, ``per_region_horizon.csv``,
        ``per_activity_level.csv``, ``per_activity_horizon.csv``). Extra keys
        are allowed and written the same way -- the hurdle family adds named
        row populations such as ``count_head@all_days`` (F65).
        """
        directory = self.metrics_dir(run, stage)
        written: dict[str, Path] = {}
        ordered = [v for v in METRIC_VIEWS if v in views]
        ordered += [v for v in views if v not in METRIC_VIEWS]
        for name in ordered:
            value = views[name]
            _check_name(name, "metric view")
            if name == "global":
                written[name] = _atomic_json(directory / "global.json", value)
                continue
            if not isinstance(value, pd.DataFrame):
                raise TypeError(
                    f"metric view {name!r} must be a DataFrame, got {type(value).__name__}"
                )
            written[name] = _atomic_csv(directory / f"{name}.csv", value)
        return written

    def read_metrics(self, run: RunKey | str | Path, stage: str) -> dict[str, Any]:
        """Read back whatever :meth:`write_metrics` wrote."""
        directory = self.metrics_dir(run, stage)
        views: dict[str, Any] = {}
        if not directory.exists():
            return views
        for path in sorted(directory.iterdir()):
            if path.name == "global.json":
                views["global"] = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".csv":
                views[path.stem] = pd.read_csv(path)
        return views

    def write_artifact(self, run: RunKey | str | Path, name: str, obj: Any) -> Path:
        """Write one artifact under ``artifacts/``. JSON or parquet, never pickle.

        The extension decides the format. Without one, a ``DataFrame`` becomes
        parquet and everything else JSON. Any other extension is refused: the
        plan's rule is that no darts, Optuna or sklearn object is ever pickled
        (§5.2), so an artifact that cannot be expressed as a frame or as JSON
        does not belong in the store.
        """
        _check_name(name, "artifact name")
        suffix = Path(name).suffix
        if not suffix:
            suffix = ".parquet" if isinstance(obj, pd.DataFrame) else ".json"
            name = f"{name}{suffix}"
        path = self.artifacts_dir(run) / name
        if suffix == ".parquet":
            if not isinstance(obj, pd.DataFrame):
                raise TypeError(f"{name}: parquet artifacts must be a DataFrame")
            return _atomic_parquet(path, obj)
        if suffix == ".json":
            return _atomic_json(path, obj)
        raise ValueError(
            f"{name}: only .json and .parquet artifacts are supported (never pickle)"
        )


# --------------------------------------------------------------------------- #
# prediction parts
# --------------------------------------------------------------------------- #
class PartWriter:
    """Appends prediction parts to one stage and keeps ``state.json`` in step.

    A part is ``cv/predictions/part-<from>-<to>.parquet`` with the fold indices
    it covers, inclusive, zero-padded to six digits so a plain ``ls`` sorts
    them. The write is atomic (temp file in the same directory plus
    ``os.replace``) and the state is updated only after the parquet has landed,
    so the on-disk parts are always a prefix of what the state claims, never the
    other way round.
    """

    def __init__(self, store: RunStore, run: RunKey | str | Path, stage: str) -> None:
        self.store = store
        self.run = run
        self.stage = stage

    @staticmethod
    def part_name(fold_from: int, fold_to: int) -> str:
        return f"part-{int(fold_from):06d}-{int(fold_to):06d}.parquet"

    def write(
        self,
        predictions: PredictionSet,
        fold_from: int,
        fold_to: int,
        *,
        next_retrain_fold: int | None = None,
    ) -> Path:
        """Write one part and record it in ``state.json``.

        ``next_retrain_fold`` is the fold that opens the NEXT part (the retrain
        fold that triggered this flush), or ``None`` at the end of a run.
        """
        if fold_to < fold_from:
            raise ValueError(f"empty part range {fold_from}..{fold_to}")
        directory = self.store.predictions_dir(self.run, self.stage)
        path = directory / self.part_name(fold_from, fold_to)
        _atomic_parquet(path, predictions.frame)

        state = self.store.read_state(self.run, self.stage) or StageState(stage=self.stage)
        state.parts = [p for p in state.parts if p["path"] != path.name]
        state.parts.append(
            {
                "path": path.name,
                "from": int(fold_from),
                "to": int(fold_to),
                "rows": int(len(predictions)),
            }
        )
        state.parts.sort(key=lambda p: int(p["from"]))
        state.folds_done = int(state.parts[-1]["to"]) + 1
        state.last_retrain_fold = next_retrain_fold
        if state.status == "pending":
            state.status = "running"
        self.store.write_state(self.run, state)
        return path


class PersistHook:
    """A :class:`~strikecast.backtest.protocols.FoldHook` that streams predictions to disk.

    It buffers :class:`FoldResult`\\ s and flushes them as one part when the next
    retrain fold arrives, and once more at :meth:`close`. With the legacy
    schedule (``predict_stride=1``, ``retrain_stride=7``) that is one part per
    week of folds, so a crash costs at most the six folds since the last
    retrain -- which is what §5.3's "recomputes at most six folds" means.

    The hook only reads ``result.predictions``; the cumulative snapshot the
    engine hands it is ignored, so it never retains the engine's state (F61 is
    moot here, but the independence is deliberate: the buffered ``TimeSeries``
    objects are the per-fold ones, already level-space and inverse-transformed).

    ``actuals`` are the LEVEL-space target series, in the same region order as
    the list handed to the engine, and ``region_names`` their names: both are
    what :meth:`PredictionSet.from_fold_preds` needs to attach ``y_true`` and
    ``region``. The engine does not carry them, hence the keyword arguments.

    Fold numbering: ``PredictionSet.from_fold_preds`` numbers folds from 0
    within the list it is given, so each part's ``fold`` column is shifted by
    the part's first fold index. A loaded run therefore has globally correct,
    contiguous fold numbers.
    """

    def __init__(
        self,
        store: RunStore,
        run: RunKey | str | Path,
        stage: str,
        retrain_stride: int | None = 7,
        *,
        actuals: Sequence[TimeSeries],
        region_names: Sequence[str],
    ) -> None:
        if len(actuals) != len(region_names):
            raise ValueError(
                f"{len(actuals)} actual series for {len(region_names)} region names"
            )
        self.store = store
        self.run = run
        self.stage = stage
        self.retrain_stride = retrain_stride
        self.actuals = list(actuals)
        self.region_names = list(region_names)
        self.writer = PartWriter(store, run, stage)
        self.parts: list[Path] = []
        self._buffer: list[FoldResult] = []

    # -- FoldHook -----------------------------------------------------------
    def on_fold(
        self,
        result: FoldResult,
        cumulative: dict[str, list[list[TimeSeries]]],
    ) -> None:
        if result.fold.retrain and self._buffer:
            self._flush(next_retrain_fold=result.fold.index)
        self._buffer.append(result)

    def close(self) -> Path | None:
        """Flush whatever is buffered. Returns the final part, if any."""
        if not self._buffer:
            return None
        return self._flush(next_retrain_fold=None)

    def __enter__(self) -> PersistHook:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()

    # -- internals ----------------------------------------------------------
    def _flush(self, *, next_retrain_fold: int | None) -> Path:
        buffered = self._buffer
        self._buffer = []
        fold_from = buffered[0].fold.index
        fold_to = buffered[-1].fold.index

        channels = list(buffered[0].predictions)
        per_channel: dict[str, list[list[TimeSeries]]] = {
            channel: [[] for _ in self.region_names] for channel in channels
        }
        for res in buffered:
            for channel, region_preds in res.predictions.items():
                for r_idx, pred in enumerate(region_preds):
                    per_channel[channel][r_idx].append(pred)

        predictions = PredictionSet.from_fold_preds(
            self.actuals, per_channel, self.region_names
        )
        frame = predictions.frame
        if fold_from and len(frame):
            frame = frame.copy()
            frame["fold"] = frame["fold"] + int(fold_from)
            predictions = PredictionSet(frame.loc[:, list(COLUMNS)])

        path = self.writer.write(
            predictions, fold_from, fold_to, next_retrain_fold=next_retrain_fold
        )
        self.parts.append(path)
        return path


def iter_parts(paths: Iterable[Path]) -> Iterable[PredictionSet]:
    """Read parts lazily, in the order given."""
    for path in paths:
        yield PredictionSet.from_parquet(path)
