"""Feature importance as a pipeline stage (audit C14; thesis Figs 20, 21, 23).

``strikecast importance experiment=<e> model=<m> paradigm=<p> seed=<s>``
reproduces what the thesis computed outside the pipeline:

GBDT (``_regression_GBDT.ipynb`` cells 57-58)
    per paradigm group (``global``: one model; ``activity``: one per tier, in
    ``sorted`` order), build the TUNED model (``resolve_params``: the
    experiment entry, else ``tuning/<model>/best_params.json``, else the
    spec defaults -- the same parameters the test stage uses), fit it on the
    group's FULL series (``bundle.target_full`` with the post-selection
    ``past_covs``/``future_covs``, exactly ``fit(series=target_full, ...)`` in
    cell 57), then :func:`strikecast.evaluation.importance.gbm_importances`
    (gain + ``permutation_importance(n_repeats=5, random_state=42,
    n_jobs=-1)`` per horizon). The row label is the legacy one:
    ``global_<model>`` / ``activity_<tier>_<model>`` (``local_<region>_<model>``
    for the local paradigm, which the thesis never computed).
Chronos-2 (``_chronos2.py:526, 686-688``)
    load the predictor the test stage fitted (``<run>/artifacts/predictor/``,
    :mod:`strikecast.pipeline.chronos_stage`) and call
    ``predictor.feature_importance(full frame, model=<its model>,
    relative_scores=True)``. Needs ``envs/autogluon``.

Outputs, next to the run they explain (run-store layout, §5.3)::

    <exp>/<model>/<paradigm>/seed=<s>/
      state.json                      stage "importance" (complete/failed + hash)
      importance/importance.csv       GBDT: importance_all schema, this run's rows
                                      Chronos: AutoGluon's frame (index = feature)
      importance/category_shares.csv  model, metric, category, share, top_n
      importance/timings.json         seconds per group and part (gain/design/perm)

and :func:`collect_importance` gathers every completed run of one seed into::

    <exp>/report/importance/seed=<s>/importance_all.csv     (GBDT, = results/gbdt/importance_all.csv)
    <exp>/report/importance/seed=<s>/category_shares.csv
    <exp>/report/importance/seed=<s>/feature_importance_<model>.csv   (Chronos)

The stage is resumable in the run-store sense: it is identified by
``stage_hash(resolved importance config, upstream hashes, seed)`` and a
complete stage with the same identity is skipped (``--force`` recomputes).
A GBDT activity run writes nothing until all its groups are done; a killed job
recomputes the run (minutes to ~2 h, see ``impl_stream2.md``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from strikecast.pipeline.context import get_spec, make_run_context, resolve_store_root
from strikecast.seeds import record_env, seed_everything
from strikecast.store import RunKey, RunStore, stage_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strikecast.config.schema import ExperimentConfig
    from strikecast.pipeline.data_stage import DataArtifacts

__all__ = [
    "DEFAULT_JOBS",
    "GBDT_FAMILIES",
    "IMPORTANCE_STAGE",
    "ImportanceOutcome",
    "collect_importance",
    "default_jobs",
    "importance_dir",
    "run_importance",
]

logger = logging.getLogger(__name__)

#: The stage name in ``state.json`` and the directory under the run.
IMPORTANCE_STAGE = "importance"

#: Families whose darts model wraps one sklearn-API booster per horizon.
GBDT_FAMILIES = ("lightgbm", "xgboost", "catboost")

#: What the thesis computed, per experiment: ``{model: [paradigm, ...]}``.
#: ``count``: ``importance_all.csv`` holds ``global_lightgbm_poisson`` and the
#: six ``activity_{1,2,3}_{lightgbm_poisson,catboost_tweedie}`` rows (Figs 20,
#: 23); cell 57 as saved also fits ``global_catboost_tweedie``
#: (``importance_global_catboost_tweedie_2.csv``). ``chronos2``: both
#: predictors (``feature_importance_chronos2_{zs,ft}.csv``; Fig 21 uses ft).
DEFAULT_JOBS: dict[str, dict[str, tuple[str, ...]]] = {
    "count": {
        "lightgbm_poisson": ("global", "activity"),
        "catboost_tweedie": ("global", "activity"),
    },
    "chronos2": {
        "chronos2_fine_tuned": ("global",),
        "chronos2_zero_shot": ("global",),
    },
}

#: Legacy permutation settings (cell 57).
N_REPEATS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class ImportanceOutcome:
    """What one importance run produced (or skipped)."""

    run_key: RunKey
    stage_hash: str
    skipped: bool
    path: Path | None = None
    n_rows: int = 0
    seconds: float = 0.0
    labels: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        key = self.run_key
        return f"{key.experiment}/{key.model}/{key.paradigm}/seed={key.seed}/{IMPORTANCE_STAGE}"


def default_jobs(experiment: str) -> dict[str, tuple[str, ...]]:
    """The thesis's importance jobs for one experiment (empty when it had none)."""
    return dict(DEFAULT_JOBS.get(experiment, {}))


def importance_dir(store: RunStore, key: RunKey) -> Path:
    """``<run>/importance/``."""
    return store.stage_dir(key, IMPORTANCE_STAGE)


def _atomic_csv(path: Path, frame: pd.DataFrame, *, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=index)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def group_label(paradigm: str, group: str, model: str) -> str:
    """``global_<m>`` / ``activity_<tier>_<m>`` / ``local_<region>_<m>`` (cell 57)."""
    if paradigm == "global":
        return f"global_{model}"
    return f"{paradigm}_{group}_{model}"


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def run_importance(
    cfg: ExperimentConfig,
    model_name: str,
    paradigm: str,
    seed: int,
    data: DataArtifacts,
    *,
    store: RunStore | None = None,
    force: bool = False,
    permutation: bool = True,
    n_repeats: int = N_REPEATS,
    n_jobs: int | None = -1,
    random_state: int = RANDOM_STATE,
) -> ImportanceOutcome:
    """Compute (or skip) the importance of one ``(model, paradigm, seed)``."""
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    if n_jobs is None or n_jobs < 1:
        # Legacy passed n_jobs=-1 on Colab, where "all CPUs" meant 12 = the family's
        # pinned thread count. On a 64-CPU cluster node -1 means 64 loky workers,
        # each holding the model + design matrix and running CatBoost's own 12
        # threads: workers were killed with SIGABRT (cluster jobs 64558/64559).
        # Permutation seeds are drawn up front from random_state, so the result
        # does not depend on n_jobs, and n_jobs is not part of the stage identity.
        n_jobs = int(cfg.resolved_threads or 1)
    spec = get_spec(model_name, cfg.name)
    key = RunKey(cfg.name, model_name, str(paradigm), int(seed))

    if spec.kind == "chronos":
        resolved: dict[str, Any] = {"family": "chronos", "method": "autogluon_permutation",
                                    "relative_scores": True}
        test_state = store.read_state(key, "test")
        # The importance explains the predictor the test stage fitted, so its
        # identity follows that stage's.
        upstream = [*data.upstream, getattr(test_state, "stage_hash", None) or ""]
    elif spec.kind == "global" and spec.family in GBDT_FAMILIES:
        from strikecast.pipeline.run_stage import resolve_params  # noqa: PLC0415

        params, params_source = resolve_params(cfg, spec, model_name, store)
        resolved = {
            "family": spec.family,
            "params": params,
            "params_source": params_source,
            "permutation": bool(permutation),
            "n_repeats": int(n_repeats),
            "random_state": int(random_state),
            "device": cfg.device_for(model_name),
            "threads": cfg.threads,
            "common_kwargs": cfg.common_kwargs.model_dump(mode="json"),
        }
        upstream = list(data.upstream)
    else:
        raise ValueError(
            f"feature importance is defined for the GBDT families ({', '.join(GBDT_FAMILIES)}) "
            f"and Chronos-2 only; {model_name!r} is kind={spec.kind!r} family={spec.family!r}"
        )

    resolved = {
        "experiment": cfg.name,
        "model": model_name,
        "paradigm": str(paradigm),
        "stage": IMPORTANCE_STAGE,
        **resolved,
    }
    digest = stage_hash(resolved, upstream, int(seed))
    if not force and store.is_complete(key, IMPORTANCE_STAGE, digest):
        logger.info("skip %s: complete under the same identity", key.relative())
        return ImportanceOutcome(key, digest, skipped=True, path=importance_dir(store, key))

    seed_everything(int(seed))
    store.start_stage(key, IMPORTANCE_STAGE, digest)
    run_dir = store.resolve(key)
    if not (run_dir / "config.yaml").exists():
        store.write_config(key, cfg.model_dump(mode="json"))
    if not (run_dir / "env.json").exists():
        store.write_env(key, record_env())
    start = time.perf_counter()
    try:
        if spec.kind == "chronos":
            path, frame, labels, timings = _chronos(cfg, key, data, store)
        else:
            path, frame, labels, timings = _gbdt(
                cfg,
                spec,
                model_name,
                str(paradigm),
                int(seed),
                data,
                store,
                key,
                params=resolved["params"],
                permutation=permutation,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        seconds = time.perf_counter() - start
        _atomic_json(
            importance_dir(store, key) / "timings.json",
            {"total_s": seconds, "groups": timings, "n_jobs": n_jobs,
             "cpu_count": os.cpu_count()},
        )
        store.complete_stage(key, IMPORTANCE_STAGE)
    except KeyboardInterrupt as exc:  # SIGTERM/SIGUSR1 via strikecast.pipeline.interrupt (C21)
        mark = getattr(store, "interrupt_stage", None)
        if mark is not None:
            mark(key, IMPORTANCE_STAGE, str(exc) or type(exc).__name__)
        raise
    except Exception as exc:
        store.fail_stage(key, IMPORTANCE_STAGE, f"{type(exc).__name__}: {exc}")
        raise
    logger.info("%s: %d rows in %.1f s -> %s", key.relative(), len(frame), seconds, path)
    return ImportanceOutcome(key, digest, False, path, len(frame), seconds, labels)


def _gbdt(
    cfg: ExperimentConfig,
    spec: Any,
    model_name: str,
    paradigm: str,
    seed: int,
    data: DataArtifacts,
    store: RunStore,
    key: RunKey,
    *,
    params: dict[str, Any],
    permutation: bool,
    n_repeats: int,
    n_jobs: int | None,
    random_state: int,
) -> tuple[Path, pd.DataFrame, list[str], dict[str, Any]]:
    from strikecast.backtest.grouping import partition, take  # noqa: PLC0415
    from strikecast.evaluation.importance import (  # noqa: PLC0415
        category_shares_long,
        gbm_importances,
    )

    bundle = data.bundle
    targets = list(bundle.target_full)
    past = list(bundle.past_covs)
    future = list(bundle.future_covs)
    groups = partition(data.region_names, paradigm, data.activity_by_region)  # type: ignore[arg-type]
    ctx = make_run_context(cfg, model_name, seed)

    frames: list[pd.DataFrame] = []
    labels: list[str] = []
    timings: dict[str, Any] = {}
    for group in groups:
        label = group_label(paradigm, group.label, model_name)
        logger.info("importance %s (%d regions)", label, len(group.indices))
        ts = [targets[i] for i in group.indices]
        pc = take(past, group.indices)
        fc = take(future, group.indices)
        t0 = time.perf_counter()
        model = spec.build(params, ctx)
        model.fit(series=ts, past_covariates=pc, future_covariates=fc)
        fit_s = time.perf_counter() - t0
        group_timings: dict[str, float] = {"fit_s": fit_s}
        frame = gbm_importances(
            model,
            spec.family,
            series=ts,
            past_covs=pc,
            fut_covs=fc,
            permutation=permutation,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
            timings=group_timings,
        )
        frame["model"] = label
        frames.append(frame)
        labels.append(label)
        timings[label] = group_timings

    importance = pd.concat(frames, ignore_index=True)
    directory = importance_dir(store, key)
    path = _atomic_csv(directory / "importance.csv", importance)
    _atomic_csv(directory / "category_shares.csv", category_shares_long(importance))
    return path, importance, labels, timings


def _chronos(
    cfg: ExperimentConfig,
    key: RunKey,
    data: DataArtifacts,
    store: RunStore,
) -> tuple[Path, pd.DataFrame, list[str], dict[str, Any]]:
    from strikecast.evaluation.importance import (  # noqa: PLC0415
        category_shares,
        chronos_importance,
    )
    from strikecast.pipeline.chronos_stage import (  # noqa: PLC0415
        ChronosData,
        load_stage_predictor,
    )

    if not isinstance(data, ChronosData):
        raise TypeError("Chronos-2 importance needs ChronosData (prepare_data on chronos2)")
    state = store.read_state(key, "test")
    if state is None or state.status != "complete":
        raise RuntimeError(
            f"{key.relative()}: the test stage is not complete; its fitted predictor is "
            "what the importance explains (run the test stage first)"
        )
    t0 = time.perf_counter()
    predictor = load_stage_predictor(store, key)
    # `test_data_ag` in `_chronos2.py` is the FULL frame: `train_test_split`
    # returns (train, full). AutoGluon scores its last `prediction_length` steps.
    fi = chronos_importance(predictor, data.tsdf)
    seconds = time.perf_counter() - t0
    directory = importance_dir(store, key)
    # Written like `fi_ft.to_csv(...)`: the feature names are the (unnamed) index.
    path = _atomic_csv(directory / "importance.csv", fi, index=True)
    two_col = pd.DataFrame({"Feature": list(fi.index), "importance": fi["importance"].to_numpy()})
    shares = category_shares(two_col)
    rows = [
        {"model": key.model, "metric": "perm", "category": c, "share": float(v), "top_n": 15}
        for c, v in shares.items()
    ]
    _atomic_csv(
        directory / "category_shares.csv",
        pd.DataFrame(rows, columns=["model", "metric", "category", "share", "top_n"]),
    )
    return path, fi, [key.model], {key.model: {"feature_importance_s": seconds}}


# --------------------------------------------------------------------------- #
# experiment-level collection
# --------------------------------------------------------------------------- #
_PARADIGM_ORDER = {"global": 0, "activity": 1, "local": 2}


def collect_importance(
    store: RunStore,
    cfg: ExperimentConfig,
    seed: int,
    *,
    models: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Gather every COMPLETE importance run of ``seed`` into ``report/importance/seed=<s>/``.

    GBDT rows are concatenated in (paradigm, model-list, group) order into
    ``importance_all.csv`` -- the file ``results/analyse_results.ipynb`` cells
    44/47 read (``global_lightgbm_poisson`` then ``activity_<k>_<model>``).
    Chronos runs are copied to ``feature_importance_<model>.csv`` in
    AutoGluon's own format (``results/chronos2/feature_importance_chronos2_*.csv``).
    """
    names = list(models) if models else list(cfg.model_names)
    out_dir = store.report_dir(cfg.name) / IMPORTANCE_STAGE / f"seed={int(seed)}"
    gbdt: list[tuple[tuple[int, int], pd.DataFrame]] = []
    shares: list[pd.DataFrame] = []
    written: dict[str, Path] = {}
    for m_idx, model in enumerate(names):
        for paradigm in ("global", "activity", "local"):
            key = RunKey(cfg.name, model, paradigm, int(seed))
            state = store.read_state(key, IMPORTANCE_STAGE) if store.resolve(key).exists() else None
            if state is None or state.status != "complete":
                continue
            directory = importance_dir(store, key)
            share_path = directory / "category_shares.csv"
            if share_path.exists():
                shares.append(pd.read_csv(share_path))
            spec = get_spec(model, cfg.name)
            if spec.kind == "chronos":
                frame = pd.read_csv(directory / "importance.csv", index_col=0)
                frame.index.name = None
                written[f"feature_importance_{model}"] = _atomic_csv(
                    out_dir / f"feature_importance_{model}.csv", frame, index=True
                )
            else:
                gbdt.append(((_PARADIGM_ORDER[paradigm], m_idx), pd.read_csv(directory / "importance.csv")))
    if gbdt:
        by_paradigm: dict[int, list[tuple[int, pd.DataFrame]]] = {}
        for (p_rank, m_rank), frame in gbdt:
            by_paradigm.setdefault(p_rank, []).append((m_rank, frame))
        ordered: list[pd.DataFrame] = []
        for p_rank in sorted(by_paradigm):
            runs = sorted(by_paradigm[p_rank], key=lambda item: item[0])
            # Legacy cell 57 loops groups OUTSIDE models (tier 1: lgbm, cb; tier 2: ...).
            labels_per_run = [list(dict.fromkeys(f["model"])) for _, f in runs]
            n_groups = max(len(labels) for labels in labels_per_run)
            for g in range(n_groups):
                for (_, frame), labels in zip(runs, labels_per_run, strict=True):
                    if g < len(labels):
                        ordered.append(frame[frame["model"] == labels[g]])
        written["importance_all"] = _atomic_csv(
            out_dir / "importance_all.csv", pd.concat(ordered, ignore_index=True)
        )
    if shares:
        written["category_shares"] = _atomic_csv(
            out_dir / "category_shares.csv", pd.concat(shares, ignore_index=True)
        )
    return written
