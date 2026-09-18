"""Build the leaderboard frame from a run store (plan sec. 5.3, sec. 7).

This module is **additive** (plan sec. 1): it reads the ``global.json`` metric
files a finished stage already wrote and never touches a pipeline. Nothing here
recomputes a metric; the arithmetic lives in
:mod:`strikecast.evaluation.metrics` and :mod:`strikecast.evaluation.aggregate`
and ran when the stage ran.

Layout consumed (plan sec. 5.3)::

    runs/<experiment>/<model>/<paradigm>/seed=<s>/{cv,test}/metrics/global.json

``model`` is the plan's ``model_variant`` and is named ``model`` here because
that is what :class:`strikecast.store.RunKey` calls it and what the legacy
``leaderboard.csv`` column is called.

Behaviour-preservation notes
----------------------------

``Q1`` **The frame, its columns and its sort come from the legacy code.**
    :func:`build_leaderboard` assembles the same ``lb_rows`` the three legacy
    scripts assemble (``{"split", "paradigm", "model", **global_metrics}``) and
    hands them to :func:`strikecast.evaluation.aggregate.leaderboard`, which is
    the verbatim port of ``pd.DataFrame(lb_rows).sort_values(["split",
    "MASE_mean"]).reset_index(drop=True)``. The column order is therefore the
    insertion order of ``global.json``, which is the insertion order of
    ``base_metrics`` followed by ``scaled_metrics`` -- exactly
    ``golden/results/{gbdt,diff}/leaderboard.csv``.

``Q2`` **One seed per leaderboard.** The legacy leaderboard has no seed column,
    so :func:`build_leaderboard` selects a single seed (default 42, the thesis
    seed) and the frame is byte-comparable with the golden files. Cross-seed
    output is :mod:`strikecast.evaluation.seeds`, which adds its own columns.

``Q3`` **``split`` is the stage name.** The run store calls the two stages
    ``cv`` and ``test``; the legacy column is called ``split`` and holds the
    same two strings, so no mapping is needed.

``Q4`` **Missing metrics stay missing.** A stage that wrote a ``global.json``
    without, say, ``MASE_mean`` becomes a NaN cell and sorts last within its
    split, which is the legacy behaviour (``na_position="last"``). Nothing is
    filled in.

``Q5`` **The golden reader is a separate entry point.**
    :func:`golden_metric_rows` parses the flat
    ``global_<split>_<paradigm>_<model>.json`` naming of
    ``golden/results/<family>/`` so the same builder can be pointed at the
    thesis artefacts. It is a compatibility shim for the equivalence tests, not
    a run-store layout.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregate import leaderboard as _leaderboard_frame

__all__ = [
    "GOLDEN_PARADIGMS",
    "LEADERBOARD_KEYS",
    "MetricRow",
    "STAGES",
    "build_leaderboard",
    "build_leaderboard_from_rows",
    "collect_metric_rows",
    "discover_metric_files",
    "golden_metric_rows",
    "leaderboard_rows",
    "metric_frame",
    "read_global_metrics",
    "write_leaderboard",
]

#: The two stages a run store writes metrics for, in leaderboard sort order.
STAGES: tuple[str, ...] = ("cv", "test")

#: The three leading columns of ``leaderboard.csv``, before the metric block.
LEADERBOARD_KEYS: tuple[str, ...] = ("split", "paradigm", "model")

#: Paradigm names, needed to split ``global_<split>_<paradigm>_<model>.json``
#: because both the split and the paradigm can be the literal ``global``.
GOLDEN_PARADIGMS: tuple[str, ...] = ("global", "activity", "local")

_SEED_DIR = re.compile(r"^seed=(-?\d+)$")
_GOLDEN_FILE = re.compile(r"^global_(cv|test)_(.+)\.json$")
_RESERVED = frozenset({"shared", "tuning", "report"})


@dataclass(frozen=True)
class MetricRow:
    """One ``global.json`` with the coordinates that name it.

    ``metrics`` keeps the file's key order, which is what makes the leaderboard
    column order match the legacy CSV (Q1).
    """

    experiment: str
    model: str
    paradigm: str
    seed: int
    stage: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        """``(experiment, model, paradigm, stage)`` -- everything but the seed."""
        return (self.experiment, self.model, self.paradigm, self.stage)


def read_global_metrics(path: str | Path) -> dict[str, Any]:
    """Read one ``global.json``, preserving key order.

    A stage may write the global view as a one-row mapping (what
    :meth:`strikecast.store.RunStore.write_metrics` does with a dict or a
    ``Series``); anything else is refused rather than coerced, because a silent
    coercion would reorder the columns.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: global.json must hold an object, got {type(payload).__name__}")
    return dict(payload)


def discover_metric_files(
    root: str | Path,
    *,
    experiment: str | None = None,
    stages: Sequence[str] = STAGES,
) -> Iterator[tuple[str, str, str, int, str, Path]]:
    """Walk a run store and yield ``(experiment, model, paradigm, seed, stage, path)``.

    Only directories that match the full plan sec. 5.3 layout are yielded, so the
    ``shared/``, ``tuning/`` and ``report/`` siblings of a model directory are
    skipped (they have no ``seed=<s>`` level anyway). Results are sorted so the
    walk is deterministic across filesystems.
    """
    root = Path(root)
    wanted = tuple(stages)
    found: list[tuple[str, str, str, int, str, Path]] = []
    if experiment is not None:
        experiments = [root / experiment]
    elif root.is_dir():
        experiments = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        experiments = []
    for exp_dir in experiments:
        if not exp_dir.is_dir():
            continue
        for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            if model_dir.name in _RESERVED:
                continue
            for paradigm_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for seed_dir in sorted(p for p in paradigm_dir.iterdir() if p.is_dir()):
                    m = _SEED_DIR.match(seed_dir.name)
                    if m is None:
                        continue
                    seed = int(m.group(1))
                    for stage in wanted:
                        path = seed_dir / stage / "metrics" / "global.json"
                        if path.is_file():
                            found.append(
                                (
                                    exp_dir.name,
                                    model_dir.name,
                                    paradigm_dir.name,
                                    seed,
                                    stage,
                                    path,
                                )
                            )
    found.sort(key=lambda t: (t[0], t[1], t[2], t[3], wanted.index(t[4])))
    yield from found


def collect_metric_rows(
    root: str | Path,
    *,
    experiment: str | None = None,
    stages: Sequence[str] = STAGES,
    seeds: Iterable[int] | None = None,
    models: Iterable[str] | None = None,
    paradigms: Iterable[str] | None = None,
) -> list[MetricRow]:
    """Every ``global.json`` in the store as a :class:`MetricRow`, filtered.

    ``seeds``, ``models`` and ``paradigms`` are optional whitelists; ``None``
    means "everything present".
    """
    seed_set = None if seeds is None else {int(s) for s in seeds}
    model_set = None if models is None else set(models)
    paradigm_set = None if paradigms is None else set(paradigms)
    rows: list[MetricRow] = []
    for exp, model, paradigm, seed, stage, path in discover_metric_files(
        root, experiment=experiment, stages=stages
    ):
        if seed_set is not None and seed not in seed_set:
            continue
        if model_set is not None and model not in model_set:
            continue
        if paradigm_set is not None and paradigm not in paradigm_set:
            continue
        rows.append(
            MetricRow(
                experiment=exp,
                model=model,
                paradigm=paradigm,
                seed=seed,
                stage=stage,
                metrics=read_global_metrics(path),
                path=path,
            )
        )
    return rows


def leaderboard_rows(rows: Iterable[MetricRow]) -> list[dict[str, Any]]:
    """``MetricRow`` -> the legacy ``lb_rows`` dicts (Q1).

    ``{"split": stage, "paradigm": paradigm, "model": model, **metrics}``, in
    that order, which is the column order of ``leaderboard.csv``.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "split": row.stage,
                "paradigm": row.paradigm,
                "model": row.model,
                **dict(row.metrics),
            }
        )
    return out


def build_leaderboard_from_rows(rows: Iterable[MetricRow]) -> pd.DataFrame:
    """The leaderboard frame for an already-selected set of rows.

    The caller is responsible for having picked one seed (Q2); this function
    does not check, because the golden reader has no seed to pick.
    """
    return _leaderboard_frame(leaderboard_rows(rows))


def build_leaderboard(
    root: str | Path,
    *,
    experiment: str | None = None,
    seed: int = 42,
    stages: Sequence[str] = STAGES,
    models: Iterable[str] | None = None,
    paradigms: Iterable[str] | None = None,
) -> pd.DataFrame:
    """``leaderboard.csv`` for one seed, read out of a run store.

    With ``seed=42`` and the thesis configs this reproduces
    ``golden/results/<family>/leaderboard.csv`` column for column and row for
    row, modulo the undefined order of exactly-tied ``MASE_mean`` values that
    the legacy quicksort also leaves undefined.
    """
    rows = collect_metric_rows(
        root,
        experiment=experiment,
        stages=stages,
        seeds=[seed],
        models=models,
        paradigms=paradigms,
    )
    return build_leaderboard_from_rows(rows)


def metric_frame(rows: Iterable[MetricRow]) -> pd.DataFrame:
    """The long ``(experiment, model, paradigm, seed, stage, metric, value)`` frame.

    This is what :mod:`strikecast.evaluation.seeds` aggregates over. Only
    numeric metric values are kept: a ``global.json`` may carry a string field
    and a mean over strings is not a thing.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        for metric, value in row.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            records.append(
                {
                    "experiment": row.experiment,
                    "model": row.model,
                    "paradigm": row.paradigm,
                    "seed": row.seed,
                    "stage": row.stage,
                    "metric": metric,
                    "value": float(value),
                }
            )
    frame = pd.DataFrame.from_records(
        records,
        columns=["experiment", "model", "paradigm", "seed", "stage", "metric", "value"],
    )
    return frame


def write_leaderboard(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the leaderboard exactly as the legacy scripts did: ``to_csv(index=False)``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


# --------------------------------------------------------------------------- #
# golden compatibility (Q5)
# --------------------------------------------------------------------------- #
def _split_golden_stem(rest: str, paradigms: Sequence[str]) -> tuple[str, str]:
    """``"global_lightgbm_poisson_tuned"`` -> ``("global", "lightgbm_poisson_tuned")``."""
    for paradigm in paradigms:
        prefix = f"{paradigm}_"
        if rest.startswith(prefix):
            return paradigm, rest[len(prefix) :]
    raise ValueError(f"cannot split {rest!r} into a paradigm from {list(paradigms)} plus a model")


def golden_metric_rows(
    directory: str | Path,
    *,
    experiment: str | None = None,
    seed: int = 42,
    paradigms: Sequence[str] = GOLDEN_PARADIGMS,
    stages: Sequence[str] = STAGES,
) -> list[MetricRow]:
    """Read ``golden/results/<family>/global_<split>_<paradigm>_<model>.json``.

    The thesis wrote one flat directory per family instead of the run-store
    tree, but the file name carries the same four coordinates, so the same
    builder can be pointed at it (Q5). ``seed`` is a label only: the thesis ran
    a single seed and did not record it.
    """
    directory = Path(directory)
    name = experiment if experiment is not None else directory.name
    wanted = tuple(stages)
    rows: list[MetricRow] = []
    for path in sorted(directory.glob("global_*.json")):
        m = _GOLDEN_FILE.match(path.name)
        if m is None:
            continue
        stage, rest = m.group(1), m.group(2)
        if stage not in wanted:
            continue
        try:
            paradigm, model = _split_golden_stem(rest, paradigms)
        except ValueError:
            continue
        rows.append(
            MetricRow(
                experiment=name,
                model=model,
                paradigm=paradigm,
                seed=seed,
                stage=stage,
                metrics=read_global_metrics(path),
                path=path,
            )
        )
    return rows
