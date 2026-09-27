"""The Chronos-2 fine-tune study: ``_chronos2.py`` §8 on the shared Optuna runner.

The legacy study (``_chronos2.py:548-633``)::

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(chronos2_objective, n_trials=12, timeout=None,
                   show_progress_bar=True, gc_after_trial=True)

    def chronos2_objective(trial):
        fine_tune_lr    = trial.suggest_float("fine_tune_lr", 1e-6, 1e-4, log=True)
        fine_tune_steps = trial.suggest_int("fine_tune_steps", 200, 3000, step=100)
        pred = TimeSeriesPredictor(..., path=<tempdir>, verbosity=0).fit(
            train_data, hyperparameters={"Chronos2": {"fine_tune": True, ...,
            "ag_args": {"name_suffix": "FT"}}}, enable_ensemble=False,
            num_val_windows=3, random_seed=RANDOM_STATE, verbosity=0)
        return -float(pred.leaderboard().iloc[0]["score_val"])

is reproduced here with the pieces that already exist:

* the search space is :func:`strikecast.models.chronos.search_space` (same
  names, bounds, ``log=True`` and ``step=100``, suggested in the same order);
* the fit is :func:`strikecast.models.chronos.fit_predictor` with
  :func:`~strikecast.models.chronos.fine_tune_hyperparameters` (``name_suffix
  "FT"``, ``verbosity=0``) and the objective is
  :func:`~strikecast.models.chronos.internal_validation_score` (F129, F130);
* the study is :func:`strikecast.tuning.optuna_runner.tune`: SQLite storage
  under ``tuning/<model>/``, ``TPESampler(seed=tuning_seed)``, resume to exactly
  ``n_trials`` finished trials, ``best_params.json`` + ``trials.csv``.

Pruning. ``create_study`` in the legacy cell passes no pruner, so Optuna
installs its default ``MedianPruner()`` -- which is exactly what
:func:`~strikecast.tuning.optuna_runner.tune` installs too (with
``n_warmup_steps=0`` here). The objective never calls ``trial.report``, so in
both cases **no trial can be pruned**; the pruner object is inert (audit B12).

Differences from the legacy cell, none of which can change a sampled value or
a score:

* the trial predictor is fit into ``tuning/<model>/trials/trial_<n>/`` instead
  of ``tempfile.mkdtemp()``, and is deleted when the trial ends (the legacy
  cell kept the best trial's directory and never read it again: §9 re-fits the
  winner from scratch). ``keep_predictors=True`` keeps them all;
* ``show_progress_bar=False`` (a tqdm bar is noise in a SLURM log);
* no ``trial.set_user_attr("predictor_path", ...)``: a trial attribute would
  add a column to ``trials.csv`` (see ``optuna_runner._write_artifacts``).

AutoGluon is imported only inside :func:`strikecast.models.chronos.fit_predictor`,
so this module imports in the main environment; running a study needs
``envs/autogluon``.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strikecast.config.schema import ExperimentConfig
    from strikecast.tuning.optuna_runner import RunTrial, SpecLike, TuningResult

__all__ = [
    "CHRONOS_OBJECTIVE",
    "ChronosTuningSettings",
    "make_chronos_run_trial",
    "settings_from_config",
    "tune_chronos",
]

logger = logging.getLogger(__name__)

#: The objective's name in ``configs/experiment/chronos2.yaml`` (F129): AutoGluon's
#: internal 3-window MASE, deliberately not called ``MASE_mean``/``RMSSE_mean``.
CHRONOS_OBJECTIVE = "internal_MASE"

#: ``name_suffix`` of a trial predictor (``_chronos2.py:596``); the model is
#: therefore called ``Chronos2FT`` inside every trial predictor.
TRIAL_NAME_SUFFIX = "FT"


@dataclass(frozen=True)
class ChronosTuningSettings:
    """The ``TuningLike`` view of the Chronos study (``_chronos2.py:604-615``).

    ``n_trials=12`` is ``OPTUNA_N_TRIALS``; ``gc_after_trial=True`` is the one
    ``optimize`` option the Chronos cell passes that the GBDT/RNN cells do not;
    ``n_warmup_steps=0`` gives Optuna's default ``MedianPruner()``, which the
    legacy ``create_study`` also got by passing no pruner (never consulted).
    ``catch=()`` keeps the legacy behaviour that a failing fit aborts the study.
    """

    metric: str = CHRONOS_OBJECTIVE
    tuning_seed: int = 42
    direction: str = "minimize"
    n_warmup_steps: int = 0
    n_trials: int | None = 12
    timeout: float | None = None
    catch: tuple[type[BaseException], ...] = ()
    gc_after_trial: bool = True
    show_progress_bar: bool = False
    n_jobs: int = 1


def settings_from_config(cfg: ExperimentConfig) -> ChronosTuningSettings:
    """:class:`ChronosTuningSettings` from ``configs/experiment/chronos2.yaml``.

    ``tuning.n_trials`` is keyed by model kind; the Chronos entry is
    ``chronos`` (12). ``tuning.objective`` must be ``internal_MASE`` (F129).
    """
    tuning = cfg.tuning
    if tuning is None:
        raise ValueError(f"experiment {cfg.name!r} has no tuning block")
    if tuning.objective != CHRONOS_OBJECTIVE:
        raise ValueError(
            f"the Chronos-2 study scores AutoGluon's internal validation MASE; "
            f"tuning.objective must be {CHRONOS_OBJECTIVE!r}, got {tuning.objective!r} (F129)"
        )
    if tuning.pruner.kind not in ("none", "median"):
        raise ValueError(f"unsupported pruner {tuning.pruner.kind!r} for the Chronos study")
    return ChronosTuningSettings(
        tuning_seed=int(cfg.seeds.tuning_seed),
        direction=tuning.direction,
        n_trials=tuning.n_trials.get("chronos", 12),
        timeout=None if tuning.timeout_s is None else float(tuning.timeout_s),
    )


def make_chronos_run_trial(
    train_data: Any,
    *,
    target: str,
    known_covariates: Sequence[str],
    seed: int,
    trial_root: str | Path,
    fit: Callable[..., Any] | None = None,
    score: Callable[[Any], float] | None = None,
    keep_predictors: bool = False,
    verbosity: int | None = 0,
) -> RunTrial:
    """``(params, trial) -> internal MASE``: the body of ``chronos2_objective``.

    ``fit`` and ``score`` default to :func:`~strikecast.models.chronos.fit_predictor`
    and :func:`~strikecast.models.chronos.internal_validation_score`; tests
    inject fakes so no AutoGluon is needed.
    """
    from strikecast.models import chronos  # noqa: PLC0415

    fit_fn = fit or chronos.fit_predictor
    score_fn = score or chronos.internal_validation_score
    root = Path(trial_root)

    def run_trial(params: Mapping[str, Any], trial: Any) -> float:
        builder = chronos.Chronos2Builder(
            hyperparameters=chronos.fine_tune_hyperparameters(
                params["fine_tune_lr"], params["fine_tune_steps"], TRIAL_NAME_SUFFIX
            ),
            random_seed=int(seed),
            fine_tune=True,
            params=dict(params),
        )
        number = int(getattr(trial, "number", 0))
        trial_dir = root / f"trial_{number:03d}"
        if trial_dir.exists():  # a crashed earlier attempt of the same trial number
            shutil.rmtree(trial_dir, ignore_errors=True)
        trial_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            predictor = fit_fn(
                train_data,
                builder=builder,
                target=target,
                known_covariates_names=list(known_covariates),
                path=trial_dir,
                verbosity=verbosity,
            )
            value = float(score_fn(predictor))
        except Exception as exc:  # `_chronos2.py:589-593`: clean up, then re-raise
            logger.warning("[trial %d] FAILED: %s: %s", number, type(exc).__name__, exc)
            shutil.rmtree(trial_dir, ignore_errors=True)
            raise
        if not keep_predictors:
            shutil.rmtree(trial_dir, ignore_errors=True)
        logger.info(
            "[trial %d] lr=%.3e steps=%d -> %s=%.6f",
            number,
            float(params["fine_tune_lr"]),
            int(params["fine_tune_steps"]),
            CHRONOS_OBJECTIVE,
            value,
        )
        return value

    return run_trial


def tune_chronos(
    spec: SpecLike,
    train_data: Any,
    *,
    target: str,
    known_covariates: Sequence[str],
    settings: ChronosTuningSettings,
    directory: str | Path,
    n_trials: int | None = None,
    callbacks: Sequence[Callable[[Any, Any], None]] | None = None,
    fit: Callable[..., Any] | None = None,
    score: Callable[[Any], float] | None = None,
    keep_predictors: bool = False,
) -> TuningResult:
    """Run (or resume) the Chronos-2 fine-tune study into ``directory``.

    ``directory`` is ``RunStore.tuning_dir("chronos2", "chronos2_fine_tuned")``;
    it receives ``optuna.sqlite3``, ``best_params.json`` (the bare
    ``{fine_tune_lr, fine_tune_steps}`` pair under ``best_params``, which
    ``spec.from_best_params`` maps back) and ``trials.csv``. ``n_trials`` is the
    TOTAL the study should end with (default ``settings.n_trials``, 12).
    """
    from strikecast.tuning.optuna_runner import tune  # noqa: PLC0415

    directory = Path(directory)
    run_trial = make_chronos_run_trial(
        train_data,
        target=target,
        known_covariates=known_covariates,
        seed=settings.tuning_seed,
        trial_root=directory / "trials",
        fit=fit,
        score=score,
        keep_predictors=keep_predictors,
    )
    # `spec.n_trials` (12) would win over the settings inside `tune`; the
    # explicit argument wins over both, and the settings value is the config's.
    total = n_trials if n_trials is not None else settings.n_trials
    result = tune(
        spec,
        run_trial,
        settings,
        directory / "optuna.sqlite3",
        study_name=spec.name,
        n_trials=total,
        out_dir=directory,
        callbacks=callbacks,
    )
    trials_dir = directory / "trials"
    if not keep_predictors and trials_dir.is_dir() and not any(trials_dir.iterdir()):
        trials_dir.rmdir()
    return result
