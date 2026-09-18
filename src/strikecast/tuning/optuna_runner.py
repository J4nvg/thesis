"""Optuna tuning, reproducing the legacy studies exactly (plan §5.3, §8 P3).

The legacy tuning cells are ``_regression_GBDT.py`` lines 985-1130,
``_regression_LSTM.py`` lines 1145-1180 and ``_diff_regression.py`` lines
1120-1260. All three build the study the same way::

    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner    = optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

and optimise with::

    study.optimize(
        make_gbm_objective(variant),
        n_trials          = n_trials,
        timeout           = OPTUNA_TIMEOUT_S,   # None in every in-scope run
        show_progress_bar = True,
        n_jobs            = 1,
    )

(the NN branch drops ``show_progress_bar`` and ``n_jobs``, which are only
cosmetic and the single-worker default). ``RANDOM_STATE`` is 42 and
``OPTUNA_N_TRIALS`` is 50 in the count and diff scripts; the count script's NN
branch overrides it to 25 (``n_trials = 25 if is_nn else OPTUNA_N_TRIALS``),
which is why ``n_trials`` is a per-spec value here (``ModelSpec.n_trials``) and
not a global.

The objective itself is not this module's business. It is the loop in
``make_gbm_objective``: run the expanding-window CV fold by fold, score the
CUMULATIVE fold predictions with ``RMSSE_mean``, report to the trial, prune,
and return the last score. That is exactly
:class:`strikecast.backtest.hooks.PruningHook` driving
:meth:`strikecast.backtest.engine.ExpandingWindowBacktest.iter_folds`, and the
caller wires it: :func:`tune` takes a ``run_trial(params, trial) -> float``
callable and stays ignorant of the engine, the data and darts.

What this module adds over the legacy cells
-------------------------------------------
* **SQLite storage with ``load_if_exists``** (§5.3). The legacy code pickled
  ``(best_params, study)`` only after all 50 trials finished, so a crash at
  trial 49 lost everything (§3 item 5). Here every trial lands in
  ``tuning/<model>/optuna.sqlite3`` as it completes and a rerun continues to
  exactly ``n_trials`` finished trials.
* **Durable artefacts**: ``best_params.json`` and ``trials.csv`` in the exact
  format ``scripts/convert_legacy_artifacts.py::convert_tuning`` produced for
  ``golden/converted/tuning/*/``, so a fresh study and a converted legacy one
  are read by the same loader.

Flag F11 (documented, not fixed)
--------------------------------
Resuming a study from SQLite does **not** restore the TPE sampler's RNG state.
``TPESampler(seed=42)`` is re-seeded at construction, so a study that is
interrupted after ``k`` trials and resumed does not produce the same trial
sequence as an uninterrupted ``n_trials`` run: the sampler's internal random
state is fresh while its observation set is not. The sampled points stay valid
(TPE conditions on the completed trials it reads back from the storage) and the
study stays reproducible *given the same interruption pattern*, but an
uninterrupted run and a resumed one can differ. Golden comparisons therefore
use the stored ``best_params``, never a re-tuned study (§6, §5.3).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BEST_PARAMS_KEYS",
    "RunTrial",
    "TuningResult",
    "TuningSettings",
    "load_best_params",
    "load_tuning_meta",
    "make_objective",
    "tune",
]

logger = logging.getLogger(__name__)

#: The keys ``best_params.json`` carries, in the order
#: ``scripts/convert_legacy_artifacts.py::convert_tuning`` writes them. A fresh
#: study writes exactly these, so ``golden/converted/tuning/<v>/best_params.json``
#: and ``runs/<exp>/tuning/<v>/best_params.json`` are interchangeable.
BEST_PARAMS_KEYS: tuple[str, ...] = (
    "variant",
    "source_dir",
    "source_file",
    "best_params",
    "best_value",
    "n_trials",
    "n_pruned",
    "n_complete",
    "n_failed",
    "sampler",
    "pruner",
    "direction",
    "study_name",
)

#: ``(params, trial) -> objective value``. The caller builds the forecaster
#: from ``params``, runs the backtest with a
#: :class:`~strikecast.backtest.hooks.PruningHook` bound to ``trial`` and
#: returns the final score. Raising ``optuna.TrialPruned`` from inside (which is
#: what the hook does) prunes the trial, exactly as the legacy objective does.
RunTrial = Callable[[Mapping[str, Any], Any], float]


class SpecLike(Protocol):
    """The part of :class:`strikecast.models.spec.ModelSpec` tuning needs."""

    name: str
    n_trials: int | None

    @property
    def search_space(self) -> Callable[[Any], dict[str, Any]] | None: ...


class TuningLike(Protocol):
    """The tuning configuration :func:`tune` reads.

    The P3 config layer supplies a pydantic ``TuningConfig`` with these fields;
    :class:`TuningSettings` is the plain-dataclass stand-in used by the tests
    and by any caller that has no Hydra config yet.
    """

    metric: str
    tuning_seed: int
    direction: str
    n_warmup_steps: int
    n_trials: int | None
    timeout: float | None
    catch: tuple[type[BaseException], ...]
    gc_after_trial: bool
    show_progress_bar: bool
    n_jobs: int


@dataclass(frozen=True)
class TuningSettings:
    """Legacy-faithful defaults for everything except the objective metric.

    ``metric`` has **no default**: F39 records that the two legacy
    ``_score_fold_preds`` copies disagree on their default metric
    (``MASE_mean`` in the GBDT script, ``RMSSE_mean`` in the diff script) and
    that both objectives passed ``RMSSE_mean`` explicitly, so the defaults never
    fired. The plan's treatment is to make it an explicit config field with no
    default; this dataclass is where that rule lives.

    Every other default is the legacy value:

    ``tuning_seed=42``
        ``RANDOM_STATE`` (§5.4: the sampler uses the tuning seed only).
    ``direction="minimize"``, ``n_warmup_steps=5``
        the ``create_study`` call above.
    ``timeout=None``
        ``OPTUNA_TIMEOUT_S`` is ``None`` in all three scripts.
    ``catch=()`` and ``gc_after_trial=False``
        Optuna's defaults; NO in-scope study passed either keyword, so a
        failing trial aborted the study rather than being recorded as FAIL.
        (``_chronos2.py:614`` does pass ``gc_after_trial=True``, but that study
        has no pruner and belongs to P5.)
    ``n_jobs=1``
        passed explicitly by the count and diff GBDT branches; it is also
        Optuna's default, so the NN branches behave identically.
    ``show_progress_bar=False``
        the ONE deliberate deviation: the legacy GBDT branches passed ``True``.
        A tqdm bar on a SLURM log is noise and it cannot change any sampled
        value. Set it back to ``True`` to get the notebook's output.
    """

    metric: str
    tuning_seed: int = 42
    direction: str = "minimize"
    n_warmup_steps: int = 5
    n_trials: int | None = None
    timeout: float | None = None
    catch: tuple[type[BaseException], ...] = ()
    gc_after_trial: bool = False
    show_progress_bar: bool = False
    n_jobs: int = 1


@dataclass(frozen=True)
class TuningResult:
    """What a finished (or resumed-and-finished) study produced."""

    variant: str
    study_name: str
    best_params: dict[str, Any]
    best_value: float | None
    n_trials: int
    n_pruned: int
    n_complete: int
    n_failed: int
    sampler: str
    pruner: str
    direction: str
    storage_path: Path
    best_params_path: Path
    trials_path: Path
    trials: pd.DataFrame = field(repr=False)
    study: Any = field(repr=False, default=None)

    def meta(self) -> dict[str, Any]:
        """The ``best_params.json`` payload, with :data:`BEST_PARAMS_KEYS`."""
        return {
            "variant": self.variant,
            "source_dir": self.storage_path.parent.name,
            "source_file": self.storage_path.name,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "n_complete": self.n_complete,
            "n_failed": self.n_failed,
            "sampler": self.sampler,
            "pruner": self.pruner,
            "direction": self.direction,
            "study_name": self.study_name,
        }


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def make_objective(spec: SpecLike, run_trial: RunTrial) -> Callable[[Any], float]:
    """Build ``objective(trial) -> float`` from a spec's search space.

    This is the ``make_gbm_objective`` / ``make_nn_objective`` shape with the
    two halves separated: ``spec.search_space(trial)`` is the legacy
    ``suggester(trial, objective_kind)`` and ``run_trial(params, trial)`` is
    everything after it (build the model, run the backtest, score, prune,
    return). The order matters and is the legacy one: every parameter is
    suggested BEFORE the first fold runs, so a pruned trial still has a
    complete parameter record.
    """
    search_space = spec.search_space
    if search_space is None:
        raise ValueError(f"model {spec.name!r} has no search space; it is not tunable")

    def _objective(trial: Any) -> float:
        params = search_space(trial)
        return float(run_trial(params, trial))

    return _objective


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def tune(
    spec: SpecLike,
    run_trial: RunTrial,
    cfg: TuningLike,
    storage_path: str | Path,
    study_name: str | None = None,
    n_trials: int | None = None,
    *,
    out_dir: str | Path | None = None,
    write_artifacts: bool = True,
) -> TuningResult:
    """Run (or continue) one Optuna study and write its durable artefacts.

    Parameters
    ----------
    spec:
        The model variant. ``spec.search_space`` supplies the parameters and
        ``spec.n_trials`` the trial budget (50 for the GBDT and diff variants,
        25 for the count family's NN variants).
    run_trial:
        ``(params, trial) -> float``. The caller wires the engine and the
        :class:`~strikecast.backtest.hooks.PruningHook` in here; this module
        never imports the engine.
    cfg:
        Sampler seed, direction, pruner warm-up and the ``optimize`` options.
        See :class:`TuningSettings`.
    storage_path:
        ``tuning/<model_variant>/optuna.sqlite3``
        (:meth:`strikecast.store.RunStore.tuning_dir`). The parent directory is
        created; the SQLite URL is built from the resolved absolute path.
    study_name:
        Defaults to ``spec.name``. It is the resume key, so it must be stable
        across runs -- unlike the legacy studies, which had no name and show up
        in the converted artefacts as ``no-name-<uuid>``.
    n_trials:
        Overrides ``spec.n_trials`` / ``cfg.n_trials``. This is the TOTAL number
        of finished trials the study should end with, not the number to add: a
        study that already holds 30 finished trials optimises 20 more.

    Notes
    -----
    Finished trials are ``COMPLETE``, ``PRUNED`` and ``FAIL`` -- the same set
    the legacy ``n_trials`` count in ``best_params.json`` reports, so "50
    trials" means the same thing before and after this refactor. See F11 for
    what resuming does not restore.
    """
    import optuna  # noqa: PLC0415 -- keep `strikecast.tuning` importable without optuna

    storage_path = Path(storage_path).resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(out_dir).resolve() if out_dir is not None else storage_path.parent
    name = study_name or spec.name

    total = _resolve_n_trials(spec, cfg, n_trials)

    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction=cfg.direction,
        sampler=optuna.samplers.TPESampler(seed=cfg.tuning_seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=cfg.n_warmup_steps),
    )
    # Study-level (never trial-level) user attributes: trial attributes would
    # add columns to `trials.csv` and break the golden format.
    study.set_user_attr("tuning_seed", int(cfg.tuning_seed))
    study.set_user_attr("metric", cfg.metric)
    study.set_user_attr("n_trials_requested", int(total))

    done = _n_finished(study)
    remaining = max(0, total - done)
    if remaining:
        if done:
            logger.info(
                "study %r: %d/%d trials already finished, running %d more (F11: the "
                "TPE RNG state is NOT restored)",
                name,
                done,
                total,
                remaining,
            )
        study.optimize(
            make_objective(spec, run_trial),
            n_trials=remaining,
            timeout=cfg.timeout,
            n_jobs=cfg.n_jobs,
            catch=cfg.catch,
            gc_after_trial=cfg.gc_after_trial,
            show_progress_bar=cfg.show_progress_bar,
        )
    else:
        logger.info("study %r: %d trials already finished, nothing to do", name, done)

    result = _result_from_study(study, spec.name, storage_path, directory)
    if write_artifacts:
        _write_artifacts(result)
    return result


def _resolve_n_trials(spec: SpecLike, cfg: TuningLike, override: int | None) -> int:
    """``n_trials`` argument > ``spec.n_trials`` > ``cfg.n_trials``.

    The spec wins over the config because the budget is per variant in the
    legacy code (``n_trials = 25 if is_nn else OPTUNA_N_TRIALS``).
    """
    for candidate in (override, spec.n_trials, cfg.n_trials):
        if candidate is not None:
            if candidate < 1:
                raise ValueError(f"n_trials must be >= 1, got {candidate}")
            return int(candidate)
    raise ValueError(
        f"no trial budget for {spec.name!r}: set ModelSpec.n_trials or TuningConfig.n_trials"
    )


def _n_finished(study: Any) -> int:
    """COMPLETE + PRUNED + FAIL, the states the legacy ``n_trials`` counted."""
    return sum(1 for trial in study.trials if trial.state.is_finished())


def _result_from_study(
    study: Any,
    variant: str,
    storage_path: Path,
    directory: Path,
) -> TuningResult:
    trials = list(study.trials)

    def count(state_name: str) -> int:
        return sum(1 for t in trials if t.state.name == state_name)

    try:
        best_value = float(study.best_value)
        best_params = dict(study.best_params)
    except Exception as exc:  # no completed trial, or multi-objective
        logger.warning("study %r has no best trial (%s)", study.study_name, exc)
        best_value = None  # type: ignore[assignment]
        best_params = {}

    try:
        directions = [d.name for d in study.directions]
        direction = directions[0] if len(directions) == 1 else directions
    except Exception:
        direction = str(getattr(study, "direction", None))

    return TuningResult(
        variant=variant,
        study_name=study.study_name,
        best_params=best_params,
        best_value=best_value,
        n_trials=len(trials),
        n_pruned=count("PRUNED"),
        n_complete=count("COMPLETE"),
        n_failed=count("FAIL"),
        sampler=type(study.sampler).__name__,
        pruner=type(study.pruner).__name__,
        direction=direction,  # type: ignore[arg-type]
        storage_path=storage_path,
        best_params_path=directory / "best_params.json",
        trials_path=directory / "trials.csv",
        trials=study.trials_dataframe(),
        study=study,
    )


def _write_artifacts(result: TuningResult) -> None:
    """``best_params.json`` + ``trials.csv``, in the converted-golden format.

    ``trials.csv`` is ``study.trials_dataframe().to_csv(index=False)``, the same
    call ``convert_legacy_artifacts.py`` made, so the header is
    ``number,value,datetime_start,datetime_complete,duration,params_*,state``.
    Both files are written atomically through the run store's helpers.
    """
    from strikecast.store.run_store import _atomic_json, _atomic_text  # noqa: PLC0415

    _atomic_json(result.best_params_path, result.meta())
    _atomic_text(result.trials_path, result.trials.to_csv(index=False))


# --------------------------------------------------------------------------- #
# reading back
# --------------------------------------------------------------------------- #
def load_tuning_meta(directory: str | Path) -> dict[str, Any]:
    """The whole ``best_params.json`` payload of a tuning directory.

    Works for a run-store ``tuning/<model_variant>/`` directory and for a
    ``golden/converted/tuning/<source>/<variant>/`` one: both carry
    :data:`BEST_PARAMS_KEYS`. A path to the JSON file itself is accepted too.
    """
    path = Path(directory)
    if path.is_dir():
        path = path / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(f"no best_params.json at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("variant", "best_params") if k not in payload]
    if missing:
        raise ValueError(f"{path} is not a tuning artefact: missing {missing}")
    return payload


def load_best_params(directory: str | Path) -> dict[str, Any]:
    """The tuned parameters of a variant, from the store or from golden.

    The returned mapping is the flat ``study.best_params`` the legacy code
    pickled; ``ModelSpec.from_best_params`` turns it into builder keyword
    arguments (``build_gbm_from_params`` / ``_build_lstm_from_best``).
    """
    return dict(load_tuning_meta(directory)["best_params"])


def find_tuning_dir(roots: Sequence[str | Path], variant: str) -> Path:
    """First directory under ``roots`` that holds ``<variant>/best_params.json``.

    Lets the CLI prefer a freshly tuned study and fall back to
    ``golden/converted/tuning/``, which is the only surviving record of the
    thesis's own studies (§8 P0: the pickles were converted once).
    """
    for root in roots:
        candidate = Path(root) / variant
        if (candidate / "best_params.json").exists():
            return candidate
    raise FileNotFoundError(
        f"no best_params.json for {variant!r} under {[str(r) for r in roots]}"
    )
