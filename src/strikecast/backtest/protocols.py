"""Shared contracts for the backtest engine (Phase 2).

This module is the ONE place where the engine, the model adapters, the target
transforms and the prediction collector agree on an interface. It contains no
logic, only ``Protocol`` classes and small frozen dataclasses. Every other
Phase 2 module imports from here; nothing here imports from the rest of
``strikecast`` except ``darts`` and the standard library.

Semantics are copied from the legacy runners and must not drift:

* ``src/evaluation_tools.py::run_expanding_cv / run_expanding_cv_iter /
  run_final_test`` (count family),
* ``_diff_regression.py`` lines 660-1005 (diff family, ``_diff_to_level`` and
  the ``NaiveMean`` fallback for local models),
* ``final_hurdle.ipynb`` cells 15, 24, 29, 34 (hurdle: two darts models, a
  sample-weight series sliced at the retrain cutoff, three output channels),
* ``damage_classifier.ipynb`` cells 17 and 26 (one classifier per damage key,
  one output channel per key),
* ``_chronos2.py::chronos2_rolling_long`` (fixed predictor, never refit).

The loop that every one of those runners implements is::

    n_total   = len(reference_model_space_series)
    start_idx = int(start_frac * n_total)
    for t0 in range(start_idx, n_total - horizon + 1, predict_stride):
        retrain = (t0 - start_idx) % retrain_stride == 0
        cutoff  = time_index[t0]                     # darts drop_after(cutoff)
        if retrain: fit on  [ts.drop_after(cutoff) for ts in model_targets]
        predict n=horizon from [ts.drop_after(cutoff) for ts in model_targets]
        inverse-transform to level space, append per region

``drop_after(cutoff)`` in darts 0.43 drops ``cutoff`` itself, so training and
context both end at ``time_index[t0 - 1]``. The schedule is computed on the
MODEL-space reference series, exactly as the legacy diff runners do. For the
diff branch the legacy script differences BEFORE splitting, so the diffed test
series is one step shorter than the level series (846 vs 847) while the diffed
CV view has the SAME length as the level CV view (676) shifted one day forward
(flags F51, F52 in ``docs/REFACTOR_PLAN.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd
    from darts import TimeSeries

__all__ = [
    "Fold",
    "FoldHook",
    "FoldResult",
    "Forecaster",
    "SINGLE_CHANNEL",
    "TargetTransform",
]

#: Channel name used by every single-output forecaster. Composite forecasters
#: (hurdle, damage) return their own channel names instead.
SINGLE_CHANNEL: str = "y_pred"


@dataclass(frozen=True, slots=True)
class Fold:
    """One position of the expanding window.

    ``t0`` is the integer position in the MODEL-space reference series,
    ``cutoff`` is ``time_index[t0]`` (the first timestamp NOT in the training
    or context window), ``retrain`` says whether the engine refits before
    predicting, and ``index`` is the 0-based fold counter that the legacy
    ``collect_predictions_long`` records as the ``fold`` column.
    """

    index: int
    t0: int
    cutoff: pd.Timestamp
    retrain: bool


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Level-space predictions for one fold.

    ``predictions`` maps ``channel -> list[TimeSeries]``, one entry per region
    in the SAME order as the target list handed to the engine. Single-output
    forecasters use the key :data:`SINGLE_CHANNEL`.
    """

    fold: Fold
    predictions: dict[str, list[TimeSeries]]


@runtime_checkable
class TargetTransform(Protocol):
    """Maps level-space targets to model space and back.

    ``forward`` is called ONCE per backtest on the full level-space list and
    returns the model-space list the engine schedules and slices on.
    ``inverse`` is called per fold per region on a model-space prediction and
    receives the FULL level-space series of that region as ``context`` (the
    diff transform looks up the anchor, the last actual before the first
    predicted timestamp, from it, see ``_diff_regression.py::_diff_to_level``).
    ``Identity.inverse`` returns ``pred`` unchanged.
    """

    name: str

    def forward(self, level_series: list[TimeSeries]) -> list[TimeSeries]: ...

    def inverse(self, pred: TimeSeries, context: TimeSeries) -> TimeSeries: ...


@runtime_checkable
class Forecaster(Protocol):
    """One fit/predict unit the engine drives. Works in MODEL space.

    Lifecycle, mirroring the legacy runners:

    1. ``prepare(past_covs, future_covs)`` once before the loop. This is where
       the neural adapters fit their covariate ``Scaler`` on the full-length
       covariates (flag F1) and where composite forecasters keep their own
       auxiliary series. Non-neural adapters store the covariates unchanged.
    2. ``fit(train_series, cutoff=...)`` at every fold where ``retrain`` is
       true. ``train_series`` is already ``drop_after(cutoff)``; the cutoff is
       passed so that composite forecasters can slice auxiliary series
       (hurdle sample weights, binary targets) the same way. Local models
       (ARIMA and other per-series models) do nothing here, because the
       legacy runners only capture the builder at retrain time.
    3. ``predict(n, context_series, cutoff=...)`` at EVERY fold, including
       retrain folds. ``context_series`` is ``drop_after(cutoff)`` of the model
       targets. Local models build, fit and predict one model per series here,
       with the ``NaiveMean`` fallback where the legacy runner has one (F5).
       Returned lists are in region order and in MODEL space; the engine
       applies ``TargetTransform.inverse``. Post-processing that the legacy
       runner does before un-differencing (median of 200 samples, log-link
       ``exp``, taking the last likelihood component for classifiers, the
       ``prob * count`` hurdle product) belongs in ``predict``.

    ``channels`` lists the keys ``predict`` returns; single-output forecasters
    return ``{SINGLE_CHANNEL: [...]}``. ``retrains`` is False for forecasters
    that are never refit (Chronos-2 with a fixed predictor); the engine then
    never calls ``fit``.
    """

    channels: tuple[str, ...]
    retrains: bool

    def prepare(
        self,
        past_covs: list[TimeSeries] | None,
        future_covs: list[TimeSeries] | None,
    ) -> None: ...

    def fit(self, train_series: list[TimeSeries], *, cutoff: pd.Timestamp) -> None: ...

    def predict(
        self,
        n: int,
        context_series: list[TimeSeries],
        *,
        cutoff: pd.Timestamp,
    ) -> dict[str, list[TimeSeries]]: ...


@runtime_checkable
class FoldHook(Protocol):
    """Callback invoked by the engine after every fold.

    ``cumulative`` maps ``channel -> list[list[TimeSeries]]`` (outer by region,
    inner by fold so far), which is what ``run_expanding_cv_iter`` yields and
    what the Optuna pruning objective scores. The engine hands the hook a
    per-fold shallow copy (the diff-family twin's behaviour, F61); the
    count-family twin yields the live list, which only differs for a hook that
    retains what it was given. Raising
    ``optuna.TrialPruned`` (or any exception) from a hook aborts the run.
    """

    def on_fold(
        self,
        result: FoldResult,
        cumulative: dict[str, list[list[TimeSeries]]],
    ) -> None: ...
