"""The Chronos-2 family: a fixed AutoGluon predictor on the shared schedule (P5).

Port of ``_chronos2.py`` §7-§10.  Two model variants:

``chronos2_zero_shot``
    AutoGluon's Chronos-2 with no fine-tuning (``_chronos2.py:508-527``).
``chronos2_fine_tuned``
    the same model fine-tuned once, with the ``(fine_tune_lr,
    fine_tune_steps)`` pair Optuna picked (``_chronos2.py:648-670``); the
    thesis' winner is ``lr = 8.721349828452045e-05``, ``steps = 1500``
    (``golden/checkpoints/chronos2_best/best_params.json``, trial 10 of 12).

Both are **fixed predictors**: they are fit once, before the backtest, and the
rolling evaluation only ever calls ``predict``.  That is exactly the plan's
``retrains=False`` shape (§5.2) with ``retrain_stride=None`` (§5.2, "``None``
means *never retrain*, and is the Chronos-2 shape"), so the family runs on the
same :class:`~strikecast.backtest.engine.ExpandingWindowBacktest` and the same
:func:`~strikecast.backtest.schedule.fold_schedule` as every other family and
lands in the same :class:`~strikecast.backtest.predictions.PredictionSet`.
``chronos2_rolling_long`` disappears; :class:`Chronos2Forecaster` is all that is
left of it.

Environment
-----------
This module imports in the MAIN environment: every ``autogluon`` import is
inside a function.  Running it needs the separate environment of plan §5.7::

    uv run --project envs/autogluon python -m ...     # from the repo root

See ``envs/autogluon/README.md``.

Registry hook
-------------
``models/registry.py`` (another stream's file) needs exactly one line added to
its import block::

    from . import chronos  # noqa: F401

and nothing else: the two specs register themselves on import, under the
experiment name ``chronos2``.  ``pipeline/run_stage.py::make_forecaster`` needs
one branch, because these specs carry ``kind="chronos"``::

    if spec.kind == "chronos":
        from strikecast.models.chronos import make_forecaster as make_chronos
        return make_chronos(spec, params, ctx, data=..., predictor_dir=...)

Until that branch exists, ``make_forecaster``'s final
``raise ValueError(... which run_stage cannot drive)`` fires, which is the
loud failure we want rather than a Chronos spec silently being built as a
``GlobalDartsForecaster``.

Methodology flags raised here (continuing ``docs/REFACTOR_PLAN.md`` §4; the
adapter's F123-F126 are in :mod:`strikecast.data.autogluon`)
---------------------------------------------------------------------------

F127
    **The two variants are fit on different amounts of data than they are
    evaluated against, and the zero-shot one is not fit at all.**  Both
    ``TimeSeriesPredictor.fit`` calls receive ``train_data`` (678 of 847 steps,
    F34/F126) with ``num_val_windows=3``, but zero-shot Chronos-2 has no
    trainable parameters, so its "fit" only computes the three internal
    validation scores that the leaderboard prints.  The rolling backtest then
    conditions BOTH predictors on the growing full-length context.  Preserved;
    the consequence is that ``chronos2_zero_shot`` is deterministic
    (``stochastic=False``, §5.4) and only ``chronos2_fine_tuned`` needs a seed
    sweep.

F128
    **``num_val_windows=3`` produces three fine-tuned adapters, and which one
    predicts is AutoGluon's choice, not the script's.**
    ``golden/checkpoints/chronos2_best/models/Chronos2FT_best/`` holds ``W0``,
    ``W1`` and ``W2``, each with its own ``fine-tuned-ckpt/``.  The script never
    calls ``refit_full``, so the predictor used for all 164 folds is whichever
    window AutoGluon keeps as the model to predict with.  Preserved (the port
    loads the stored predictor rather than re-deciding), but it means the
    reported model is fine-tuned on a window that ends before the end of
    ``train_data``.  Publication: worth one sentence.

F129
    **Optuna tunes on AutoGluon's internal validation, not on the shared CV
    schedule.**  ``chronos2_objective`` reads ``-leaderboard().iloc[0]
    ["score_val"]`` -- the mean MASE over the three internal backtest windows
    AutoGluon chose -- while every other family's objective is the shared
    expanding-window CV (``RMSSE_mean`` over 79 folds, F39).  The two numbers
    are not comparable, and the Chronos study is 12 trials against the others'
    50 (``OPTUNA_N_TRIALS``).  Preserved: the objective is AutoGluon's and is
    named ``internal_MASE`` in ``configs/experiment/chronos2.yaml`` so no
    leaderboard silently compares it with ``RMSSE_mean``.

F130
    **``leaderboard().iloc[0]`` is row 0, not the best row.**  With
    ``enable_ensemble=False`` and a single ``hyperparameters`` entry there is
    exactly one model, so row 0 IS the model; the expression would silently
    pick an arbitrary row if a second model were ever added.  Preserved as
    written, with an assertion that there is exactly one row.

F131
    **The Chronos family has no CV stage.**  ``_chronos2.py`` evaluates only the
    held-out 20% test segment; the other families evaluate a validation CV
    stage as well.  ``configs/experiment/chronos2.yaml`` therefore declares
    ``stages.test`` only, and a leaderboard that merges the families will have
    no ``split == "cv"`` row for Chronos.  Preserved.

F132
    **The naive floors in ``golden/results/chronos2/`` are a fourth
    implementation.**  ``naive_rolling_long`` is not ported (flag F41 already
    records this); the shared
    :mod:`strikecast.backtest.naive` produces the identical frames once sorted,
    which is what golden level E asserts.  ``chronos2.yaml`` therefore lists the
    two naive baselines by their registry names.

F133
    **The plan calls this family's paradigm "local"; it is global.**  §2.1's
    table says "local (regions as multivariate)".  In the partition sense of
    :mod:`strikecast.backtest.grouping` -- which is what ``paradigm`` means in
    the run store -- there is exactly ONE ``TimeSeriesPredictor`` and it is
    handed all 20 items in a single ``predict`` call, so the Chronos family is
    ``global`` and nothing else.  ``configs/experiment/chronos2.yaml`` pins
    ``paradigms: [global]``; §2.1 should be corrected, not the code.
    (Chronos-2 does model the items jointly in context, which is presumably
    what "regions as multivariate" meant, but that is the model's internals,
    not the backtest's partitioning.)

F134
    **A replayed prediction is not bit-identical to the stored one.**  The
    thesis' Chronos numbers were produced on a CUDA node; replaying the SAME
    saved predictor on CPU/MPS agrees to ``max |Δ| ≈ 5e-3``, ``mean |Δ| ≈
    1e-4`` and a Pearson correlation of 0.9999999 over 280 points, on a target
    that counts events.  That is float non-determinism inside a foundation
    model, not a port difference, so the Chronos golden comparison is a
    level-F family tolerance (plan §6) and not level E's 1e-6.  The metric
    half of that test, which does no model arithmetic, still holds at 1e-9.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strikecast.data.autogluon import LongFrames, predictions_to_series
from strikecast.models.spec import ModelKind, ModelSpec, RunContext, register

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from darts import TimeSeries

    from strikecast.backtest.predictions import PredictionSet

logger = logging.getLogger(__name__)

__all__ = [
    "CHRONOS_EXPERIMENT",
    "DEFAULT_FINE_TUNE_PARAMS",
    "FINE_TUNED",
    "ZERO_SHOT",
    "Chronos2Builder",
    "Chronos2Forecaster",
    "fine_tune_hyperparameters",
    "fit_predictor",
    "level_targets_from_frames",
    "load_or_fit_predictor",
    "make_forecaster",
    "run_backtest",
    "search_space",
    "zero_shot_hyperparameters",
]

#: The experiment family these specs register under.
CHRONOS_EXPERIMENT = "chronos2"

#: Registry names.
ZERO_SHOT = "chronos2_zero_shot"
FINE_TUNED = "chronos2_fine_tuned"

#: ``kind`` is a new adapter class, not one of the four darts ones.  It is
#: spelled through ``cast`` because ``ModelKind`` lives in ``models/spec.py``,
#: which belongs to another stream; adding ``"chronos"`` to that ``Literal`` is
#: the one-line follow-up.
CHRONOS_KIND: ModelKind = cast("ModelKind", "chronos")

#: ``OUTPUT_CHUNK_LEN`` (``_chronos2.py:101``).
HORIZON = 7
#: ``NUM_VAL_WINDOWS`` (``_chronos2.py:116``).
NUM_VAL_WINDOWS = 3
#: ``eval_metric`` and ``freq`` of every ``TimeSeriesPredictor`` in the script.
EVAL_METRIC = "MASE"
FREQ = "D"
#: ``OPTUNA_N_TRIALS`` (``_chronos2.py:119``).
N_TRIALS = 12

#: The winning trial, ``golden/checkpoints/chronos2_best/best_params.json``.
DEFAULT_FINE_TUNE_PARAMS: Mapping[str, Any] = {
    "fine_tune_lr": 8.721349828452045e-05,
    "fine_tune_steps": 1500,
}


# --------------------------------------------------------------------------- #
# AutoGluon hyperparameters
# --------------------------------------------------------------------------- #
def zero_shot_hyperparameters(name_suffix: str = "ZeroShot") -> dict[str, Any]:
    """``_chronos2.py:521``, verbatim -- note the value is a LIST of one dict.

    The fine-tuned call passes a bare dict instead (:func:`fine_tune_hyper\
    parameters`).  AutoGluon accepts both; the asymmetry is the script's.
    """
    return {"Chronos2": [{"ag_args": {"name_suffix": name_suffix}}]}


def fine_tune_hyperparameters(
    fine_tune_lr: float,
    fine_tune_steps: int,
    name_suffix: str = "FT_best",
) -> dict[str, Any]:
    """``_chronos2.py:659-666`` (``FT_best``) / ``:592-599`` (a trial, ``FT``)."""
    return {
        "Chronos2": {
            "fine_tune": True,
            "fine_tune_lr": float(fine_tune_lr),
            "fine_tune_steps": int(fine_tune_steps),
            "ag_args": {"name_suffix": name_suffix},
        }
    }


def search_space(trial: Any) -> dict[str, Any]:
    """``chronos2_objective``'s two suggestions (``_chronos2.py:585-586``).

    ``fine_tune_lr``: log-uniform ``[1e-6, 1e-4]``;
    ``fine_tune_steps``: integer ``[200, 3000]`` with ``step=100``.
    """
    return {
        "fine_tune_lr": trial.suggest_float("fine_tune_lr", 1e-6, 1e-4, log=True),
        "fine_tune_steps": trial.suggest_int("fine_tune_steps", 200, 3000, step=100),
    }


def _from_best_params(best: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a bare Optuna ``best_params`` or the stored JSON sidecar.

    ``golden/checkpoints/chronos2_best/best_params.json`` nests the pair under
    ``"best_params"`` together with provenance; a fresh study's
    ``study.best_params`` is the bare pair.
    """
    payload = best.get("best_params", best) if isinstance(best, Mapping) else best
    return {
        "fine_tune": True,
        "fine_tune_lr": float(payload["fine_tune_lr"]),
        "fine_tune_steps": int(payload["fine_tune_steps"]),
    }


# --------------------------------------------------------------------------- #
# what ModelSpec.build returns
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chronos2Builder:
    """Everything needed to obtain the predictor, without importing AutoGluon.

    ``ModelSpec.build(params, ctx)`` returns one of these instead of a darts
    model: a Chronos-2 "model" is an AutoGluon ``TimeSeriesPredictor`` on disk,
    which cannot be constructed in the main environment and must not be built
    per fold.  :meth:`predictor` is the only method that touches AutoGluon.
    """

    hyperparameters: dict[str, Any]
    random_seed: int = 42
    prediction_length: int = HORIZON
    num_val_windows: int = NUM_VAL_WINDOWS
    eval_metric: str = EVAL_METRIC
    freq: str = FREQ
    fine_tune: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)

    def predictor(
        self,
        path: str | Path,
        *,
        train_data: Any = None,
        known_covariates_names: Sequence[str] | None = None,
        target: str,
        force: bool = False,
        verbosity: int | None = None,
    ) -> Any:
        """Load the predictor at ``path``, or fit it there (``_chronos2.py:509``)."""
        return load_or_fit_predictor(
            path,
            builder=self,
            train_data=train_data,
            known_covariates_names=known_covariates_names,
            target=target,
            force=force,
            verbosity=verbosity,
        )


def _zero_shot_build(params: Mapping[str, Any], ctx: RunContext) -> Chronos2Builder:
    suffix = str(params.get("name_suffix", "ZeroShot"))
    return Chronos2Builder(
        hyperparameters=zero_shot_hyperparameters(suffix),
        random_seed=ctx.seed,
        fine_tune=False,
        params=dict(params),
    )


def _fine_tuned_build(params: Mapping[str, Any], ctx: RunContext) -> Chronos2Builder:
    resolved = {**DEFAULT_FINE_TUNE_PARAMS, **dict(params)}
    suffix = str(resolved.get("name_suffix", "FT_best"))
    return Chronos2Builder(
        hyperparameters=fine_tune_hyperparameters(
            resolved["fine_tune_lr"], resolved["fine_tune_steps"], suffix
        ),
        random_seed=ctx.seed,
        fine_tune=True,
        params=dict(resolved),
    )


# --------------------------------------------------------------------------- #
# fitting / loading (AutoGluon)
# --------------------------------------------------------------------------- #
def fit_predictor(
    train_data: Any,
    *,
    builder: Chronos2Builder,
    target: str,
    known_covariates_names: Sequence[str] | None,
    path: str | Path,
    verbosity: int | None = None,
) -> Any:
    """``TimeSeriesPredictor(...).fit(...)`` exactly as ``_chronos2.py`` calls it.

    Constructor: ``prediction_length``, ``target``, ``known_covariates_names``,
    ``eval_metric="MASE"``, ``freq="D"``, ``path``
    (``_chronos2.py:513-519 / 653-658``).
    ``fit``: ``train_data``, ``hyperparameters``, ``enable_ensemble=False``,
    ``num_val_windows=3``, ``random_seed`` (``:520-525 / 659-670``).

    The Optuna trial branch additionally passes ``verbosity=0`` to both
    (``:591, :604``); ``verbosity=`` reproduces that.
    """
    from autogluon.timeseries import TimeSeriesPredictor  # noqa: PLC0415

    ctor: dict[str, Any] = {
        "prediction_length": builder.prediction_length,
        "target": target,
        "known_covariates_names": list(known_covariates_names or []),
        "eval_metric": builder.eval_metric,
        "freq": builder.freq,
        "path": str(path),
    }
    fit_kwargs: dict[str, Any] = {
        "hyperparameters": builder.hyperparameters,
        "enable_ensemble": False,
        "num_val_windows": builder.num_val_windows,
        "random_seed": builder.random_seed,
    }
    if verbosity is not None:
        ctor["verbosity"] = verbosity
        fit_kwargs["verbosity"] = verbosity

    logger.info("fitting Chronos-2 into %s (%s)", path, builder.hyperparameters)
    return TimeSeriesPredictor(**ctor).fit(train_data=train_data, **fit_kwargs)


def load_or_fit_predictor(
    path: str | Path,
    *,
    builder: Chronos2Builder,
    train_data: Any = None,
    target: str,
    known_covariates_names: Sequence[str] | None = None,
    force: bool = False,
    verbosity: int | None = None,
) -> Any:
    """``if DIR.exists(): load  else: fit`` (``_chronos2.py:509-526``, ``:648-670``).

    The legacy branch is a plain directory-existence check, which is how a
    re-run of the script reuses ``checkpoints/chronos2_best`` instead of
    fine-tuning again.  The run store's equivalent is the stage identity
    (§5.3); this keeps the legacy semantics so that a stored predictor -- the
    thesis' own ``golden/checkpoints/chronos2_best`` -- can be replayed.
    """
    from autogluon.timeseries import TimeSeriesPredictor  # noqa: PLC0415

    path = Path(path)
    if path.exists() and not force:
        logger.info("loading cached Chronos-2 predictor from %s", path)
        return TimeSeriesPredictor.load(str(path))
    if train_data is None:
        raise ValueError(
            f"no predictor at {path} and no train_data to fit one; "
            "pass train_data= or point at a saved predictor"
        )
    return fit_predictor(
        train_data,
        builder=builder,
        target=target,
        known_covariates_names=known_covariates_names,
        path=path,
        verbosity=verbosity,
    )


def internal_validation_score(predictor: Any) -> float:
    """``mase = -float(pred.leaderboard().iloc[0]["score_val"])`` (``_chronos2.py:607-609``).

    AutoGluon reports MASE as a negated score (higher is better); the objective
    flips it back.  Flag F130: the script takes row 0, which is only the right
    row because ``enable_ensemble=False`` plus one hyperparameter entry means
    there is exactly one model.  That is asserted here instead of assumed.
    """
    board = predictor.leaderboard()
    if len(board) != 1:
        raise ValueError(
            f"leaderboard has {len(board)} rows; `.iloc[0]` is only the tuned model "
            "when enable_ensemble=False leaves exactly one (F130)"
        )
    return -float(board.iloc[0]["score_val"])


# --------------------------------------------------------------------------- #
# the forecaster
# --------------------------------------------------------------------------- #
class Chronos2Forecaster:
    """A fixed AutoGluon predictor driven by the shared backtest engine.

    Implements :class:`~strikecast.backtest.protocols.Forecaster` with
    ``retrains = False``, so :class:`~strikecast.backtest.engine.Expanding\
    WindowBacktest` never calls :meth:`fit` and the whole of
    ``chronos2_rolling_long`` collapses into :meth:`predict`::

        context      = full_tsdf.slice_by_timestep(None, t0)
        future_slice = full_tsdf.slice_by_timestep(t0, t0 + horizon)
        pred         = predictor.predict(context, known_covariates=future_slice[known])

    ``t0`` is recovered from the engine's ``cutoff``, which is
    ``time_index[t0]`` of the model-space reference series -- the same integer
    the legacy loop iterates, because both index the same daily calendar.

    The covariates do NOT come through :meth:`prepare`: AutoGluon reads them off
    the one ``TimeSeriesDataFrame`` this object holds, so ``prepare`` is a
    no-op and the engine's ``past_covs`` / ``future_covs`` are ignored (they are
    ``None`` for this family).

    ``use_cache=False`` is passed to every ``predict`` -- flag F124: it changes
    no number, removes an O(n²) pickle rewrite, and is what makes a predictor
    saved on another machine replayable at all.
    """

    channels: tuple[str, ...]
    retrains: bool = False

    def __init__(
        self,
        predictor: Any,
        data: Any,
        *,
        target: str,
        known_covariates: Sequence[str] = (),
        region_names: Sequence[str],
        horizon: int = HORIZON,
        use_cache: bool = False,
    ) -> None:
        from strikecast.backtest.protocols import SINGLE_CHANNEL  # noqa: PLC0415

        self.channels = (SINGLE_CHANNEL,)
        self.predictor = predictor
        self.data = data.panel() if isinstance(data, LongFrames) else data
        self.target = target
        self.known_covariates = list(known_covariates)
        self.region_names = list(region_names)
        self.horizon = horizon
        self.use_cache = use_cache
        self.n_predicts = 0

        timestamps = self.data.loc[self.region_names[0]].index
        self._positions = {ts: i for i, ts in enumerate(timestamps)}
        self._n_steps = len(timestamps)

    # -- Forecaster protocol ---------------------------------------------- #
    def prepare(
        self,
        past_covs: list[TimeSeries] | None,
        future_covs: list[TimeSeries] | None,
    ) -> None:
        """No-op: the covariates live in the ``TimeSeriesDataFrame``."""
        if past_covs or future_covs:
            logger.debug(
                "Chronos2Forecaster ignores the engine's covariate lists; "
                "AutoGluon reads them off the TimeSeriesDataFrame"
            )

    def fit(self, train_series: list[TimeSeries], *, cutoff: pd.Timestamp) -> None:
        """Never called: ``retrains`` is False (plan §5.2)."""
        raise RuntimeError(
            "Chronos2Forecaster is a fixed predictor (retrains=False); the engine "
            "must not call fit(). Check that the stage config has retrain_stride: null."
        )

    def predict(
        self,
        n: int,
        context_series: list[TimeSeries],
        *,
        cutoff: pd.Timestamp,
    ) -> dict[str, list[TimeSeries]]:
        """One fold of ``chronos2_rolling_long`` (``_chronos2.py:419-427``)."""
        t0 = self._position(cutoff)
        pred, _ = self.predict_frames(t0, n)
        return {self.channels[0]: predictions_to_series(pred, self.region_names)}

    # -- the AutoGluon call ------------------------------------------------ #
    def predict_frames(self, t0: int, n: int | None = None) -> tuple[Any, Any]:
        """``(prediction, future_slice)`` for the fold starting at ``t0``.

        Split out of :meth:`predict` so that the golden test can compare the
        raw AutoGluon output against ``predictions_long_*.parquet`` without
        going through darts.
        """
        horizon = self.horizon if n is None else n
        context = self.data.slice_by_timestep(None, t0)
        future_slice = self.data.slice_by_timestep(t0, t0 + horizon)
        known = future_slice[self.known_covariates] if self.known_covariates else None

        pred = self.predictor.predict(
            context, known_covariates=known, use_cache=self.use_cache
        )
        self.n_predicts += 1
        if self.n_predicts == 1 or self.n_predicts % 4 == 0:
            logger.info("   fold %d: forecasted from t0=%d", self.n_predicts, t0)
        return pred, future_slice

    def _position(self, cutoff: pd.Timestamp) -> int:
        try:
            return self._positions[cutoff]
        except KeyError:
            raise KeyError(
                f"cutoff {cutoff!r} is not a timestamp of the AutoGluon frame; the "
                "engine's model-space reference series and the frame must share "
                "one calendar"
            ) from None


def make_forecaster(
    spec: ModelSpec,
    params: Mapping[str, Any],
    ctx: RunContext,
    *,
    data: Any,
    target: str,
    known_covariates: Sequence[str],
    region_names: Sequence[str],
    predictor_dir: str | Path,
    train_data: Any = None,
    horizon: int = HORIZON,
    force_fit: bool = False,
) -> Chronos2Forecaster:
    """Spec + params + context -> a ready :class:`Chronos2Forecaster`.

    This is the function the pipeline's ``kind == "chronos"`` branch calls; it
    is the only place that turns a :class:`Chronos2Builder` into a live
    AutoGluon predictor.
    """
    builder = spec.build(params, ctx)
    if not isinstance(builder, Chronos2Builder):
        raise TypeError(f"{spec.name} did not build a Chronos2Builder, got {type(builder)}")
    predictor = builder.predictor(
        predictor_dir,
        train_data=train_data,
        known_covariates_names=known_covariates,
        target=target,
        force=force_fit,
    )
    return Chronos2Forecaster(
        predictor,
        data,
        target=target,
        known_covariates=known_covariates,
        region_names=region_names,
        horizon=horizon,
    )


# --------------------------------------------------------------------------- #
# the rolling backtest on the shared schedule
# --------------------------------------------------------------------------- #
def level_targets_from_frames(
    data: Any,
    target: str,
    region_names: Sequence[str],
) -> list[TimeSeries]:
    """The level-space darts target list the engine schedules and scores on.

    Built straight off the AutoGluon frame, so the schedule is computed on the
    same 847-step daily index the legacy loop iterates and ``y_true`` comes from
    the same numbers.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from darts import TimeSeries  # noqa: PLC0415

    panel = data.panel() if isinstance(data, LongFrames) else data
    out: list[TimeSeries] = []
    for region in region_names:
        sub = panel.loc[region]
        out.append(
            TimeSeries.from_times_and_values(
                pd.DatetimeIndex(sub.index),
                np.asarray(sub[target], dtype=float).reshape(-1, 1),
                columns=[target],
            )
        )
    return out


def run_backtest(
    forecaster: Chronos2Forecaster,
    level_targets: Sequence[TimeSeries],
    *,
    start_frac: float,
    horizon: int = HORIZON,
    predict_stride: int = 1,
    max_folds: int | None = None,
) -> PredictionSet:
    """``chronos2_rolling_long``, on the shared engine.

    ``retrain_stride=None`` (never retrain) plus ``forecaster.retrains=False``
    is the Chronos shape of plan §5.2.  ``start_frac`` is the experiment's
    ``train_val_end`` (``TRAIN_FRAC + VAL_FRAC``, i.e. ``0.7999999999999999``,
    flag F17); on the real panel it resolves to ``t0 = 677`` and 164 folds,
    which is what golden level C already pins.

    ``max_folds`` truncates the schedule.  It is a TEST affordance (one fold of
    the real predictor costs ~30 s on CPU), never part of a real run: a
    truncated set must not be compared with a full one.

    Returns a :class:`~strikecast.backtest.predictions.PredictionSet` with
    region-major row order (the darts collector's), not the legacy fold-major
    one -- flag F43; sort before comparing with
    ``golden/results/chronos2/predictions_long_*.parquet``.
    """
    from itertools import islice  # noqa: PLC0415

    from strikecast.backtest.engine import ExpandingWindowBacktest  # noqa: PLC0415
    from strikecast.backtest.predictions import PredictionSet  # noqa: PLC0415
    from strikecast.config.schema import BacktestConfig  # noqa: PLC0415
    from strikecast.transforms.identity import Identity  # noqa: PLC0415

    config = BacktestConfig(
        start_frac=start_frac,
        horizon=horizon,
        predict_stride=predict_stride,
        retrain_stride=None,
    )
    engine = ExpandingWindowBacktest(config, Identity())

    targets = list(level_targets)
    channel = forecaster.channels[0]
    collected: list[list[TimeSeries]] = [[] for _ in targets]
    folds = engine.iter_folds(forecaster, targets)
    for result in islice(folds, max_folds) if max_folds is not None else folds:
        for r_idx, pred in enumerate(result.predictions[channel]):
            collected[r_idx].append(pred)

    return PredictionSet.from_fold_preds(targets, collected, forecaster.region_names)


def write_best_params(
    path: str | Path,
    best_params: Mapping[str, Any],
    *,
    future_covariates: Sequence[str],
    n_past_covariates: int,
    best_val_mase: float | None = None,
    n_trials: int = N_TRIALS,
    target: str,
) -> Path:
    """The ``best_params.json`` sidecar of ``_chronos2.py:672-686``, same keys.

    Kept so a new run's artefact is diffable against
    ``golden/checkpoints/chronos2_best/best_params.json``.  Note the legacy
    ``best_val_mase`` is ``None`` whenever the Optuna checkpoint already
    existed, which is why the stored sidecar has ``null`` there.
    """
    path = Path(path)
    payload = {
        "best_params": dict(best_params),
        "best_val_mase": best_val_mase,
        "n_trials": n_trials,
        "num_val_windows": NUM_VAL_WINDOWS,
        "prediction_length": HORIZON,
        "target": target,
        "future_covariates": list(future_covariates),
        "n_past_covariates": n_past_covariates,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
CHRONOS2_ZERO_SHOT = register(
    ModelSpec(
        name=ZERO_SHOT,
        family="chronos",
        kind=CHRONOS_KIND,
        build=_zero_shot_build,
        experiments=(CHRONOS_EXPERIMENT,),
        defaults={"name_suffix": "ZeroShot"},
        search_space=None,
        stochastic=False,  # F127: nothing is trained
        is_neural=False,
        needs_raw_past_covs=False,
        device="auto",
        tags=("foundation", "fixed_predictor"),
    )
)

CHRONOS2_FINE_TUNED = register(
    ModelSpec(
        name=FINE_TUNED,
        family="chronos",
        kind=CHRONOS_KIND,
        build=_fine_tuned_build,
        experiments=(CHRONOS_EXPERIMENT,),
        defaults=dict(DEFAULT_FINE_TUNE_PARAMS),
        search_space=search_space,
        from_best_params=_from_best_params,
        stochastic=True,  # fine-tuning is stochastic (plan §5.4)
        is_neural=True,
        needs_raw_past_covs=False,
        n_trials=N_TRIALS,
        device="auto",
        tags=("foundation", "fixed_predictor", "fine_tuned"),
    )
)


def registered() -> Iterable[ModelSpec]:
    """The two specs, in the order ``_chronos2.py`` evaluates them."""
    return (CHRONOS2_ZERO_SHOT, CHRONOS2_FINE_TUNED)
