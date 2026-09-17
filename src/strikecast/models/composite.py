"""Composite forecasters: the hurdle pair and the multi-target damage loop.

These are the two legacy loop shapes that do not fit one darts model and one
output channel. Like the darts adapters, they hold NO schedule logic: the engine
owns ``t0``, the ``retrain`` decision and ``drop_after(cutoff)``. What lives
here is the body of the legacy ``if retrain: ...`` and ``preds = ...`` blocks.

Sources, at commit ``b1bfd79``:

* ``final_hurdle.ipynb`` cell 15 (``run_hurdle_cv``), cell 24
  (``run_final_test``), cell 29 (``run_hurdle_cv_per_activity``) and cell 34
  (``run_hurdle_cv_per_region``, the local paradigm with the dummy fallback)
  -> :class:`HurdleForecaster`;
* ``damage_classifier.ipynb`` cells 17 and 26 ->
  :class:`MultiTargetClassifierForecaster`.

Byte-for-byte copies of all six loops live in ``tests/legacy_ref/`` and are what
the equivalence tests compare against.

Why these two classes hold their own series
-------------------------------------------
The :class:`~strikecast.backtest.protocols.Forecaster` protocol hands a
forecaster ONE target list and ONE pair of covariate lists. The hurdle needs
two target lists (binary events and counts), a sample-weight list and TWO pairs
of covariate lists; the damage loop needs one target list and one covariate pair
PER damage key. Both therefore take their auxiliary series through the
constructor and slice them with ``drop_after(cutoff)`` themselves -- which is
exactly what the legacy loops do, since they read the same notebook globals at
every retrain. The engine's ``prepare`` arguments are ignored (see each class's
``prepare``), and the engine's ``train_series`` / ``context_series`` are used
only where they correspond to a series the legacy loop also sliced.

Both classes assume the engine runs with the ``identity`` transform. The
``prob`` channel is a probability, not a count, so a non-identity
``TargetTransform`` would be applied to it as well and would be meaningless
(flag F71).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from darts import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd

__all__ = ["HurdleForecaster", "MultiTargetClassifierForecaster"]

logger = logging.getLogger(__name__)

#: Hurdle output channels, in the order the legacy loops return them
#: (``fold_preds_c``, ``fold_preds_r``, ``fold_preds_h``).
HURDLE_CHANNELS: tuple[str, str, str] = ("prob", "count", "hurdle")

#: ``MIN_POSITIVE_SAMPLES`` of ``final_hurdle.ipynb`` cell 34. Hard-coded there,
#: and used ONLY by the per-region (local) paradigm. See flag F62.
LEGACY_MIN_POSITIVE_SAMPLES = 50


class HurdleForecaster:
    """Event classifier x count regressor, three output channels.

    One fold of the legacy hurdle loop is:

    * retrain: rebuild both models, fit the classifier on the binary targets
      sliced at the cutoff, fit the regressor on the count targets sliced at the
      cutoff with the positive-only sample weights sliced the same way;
    * predict: classifier with ``predict_likelihood_parameters=True``, take the
      LAST likelihood component per region as ``P(Y>0)`` -- correct only while
      the fitted slice has seen both classes (flag F64); regressor as-is as
      ``E[Y|Y>0]``; ``hurdle = prob * count`` on the REGRESSOR's time index.

    Parameters
    ----------
    classifier_builder, regressor_builder:
        Zero-argument builders, called afresh at every retrain -- the notebook's
        ``get_event_classifier`` / ``get_count_regressor``.
    binary_targets:
        Level-space binary event targets (``target_for_cv_c`` in CV,
        ``full_target_c`` in the test loop), FULL length. Sliced here with
        ``drop_after(cutoff)`` at fit and at predict, exactly as the notebook
        slices the global it reads.
    count_targets:
        Optional. The count targets (``target_for_cv_r`` / ``full_target_r``),
        FULL length. When given, the forecaster slices them itself and IGNORES
        the engine's ``train_series`` / ``context_series``. When ``None`` (the
        default), the engine's lists are used -- the caller then has to hand the
        engine the count targets as ``level_targets``, which is the intended
        wiring, since the engine's lists are already ``drop_after(cutoff)`` of
        exactly those series.
    weights:
        Positive-only sample weights (``full_weights`` /
        ``make_positive_only_weights(full_target_r)``), FULL length, sliced with
        ``drop_after(cutoff)`` at every retrain.
    clf_past, clf_future, reg_past, reg_future:
        The four covariate lists. FULL length, never sliced, never scaled --
        the hurdle family has no neural models, so the legacy loop passes the
        notebook globals straight through to both ``fit`` and ``predict``
        (flag F63).
    min_positive_samples:
        ``None`` reproduces the global and per-activity paradigms (cells 15, 24
        and 29): the regressor is always fit. ``50`` reproduces the per-region
        paradigm (cell 34): at every retrain, if the training slice holds fewer
        than ``min_positive_samples`` positive weights, the regressor is
        replaced by a constant "dummy" equal to the mean of the positive
        training values (``1.0`` when there are none), and that decision holds
        until the next retrain. The legacy per-region loop runs one region at a
        time, so this mode requires single-series lists.

    Notes
    -----
    The legacy loops take the schedule from ``target_for_cv_c[0]`` -- the BINARY
    list -- while the engine takes it from the list it is handed, the counts.
    The two lists come out of the same ``get_covs_and_encodings`` split and
    share a time index, so the schedules are identical (flag F60).
    """

    channels: tuple[str, ...] = HURDLE_CHANNELS
    retrains: bool = True

    def __init__(
        self,
        classifier_builder: Callable[[], Any],
        regressor_builder: Callable[[], Any],
        *,
        binary_targets: list[TimeSeries],
        count_targets: list[TimeSeries] | None = None,
        weights: list[TimeSeries],
        clf_past: list[TimeSeries] | None,
        clf_future: list[TimeSeries] | None,
        reg_past: list[TimeSeries] | None,
        reg_future: list[TimeSeries] | None,
        min_positive_samples: int | None = None,
    ) -> None:
        if not binary_targets:
            raise ValueError("binary_targets is empty")
        if len(weights) != len(binary_targets):
            raise ValueError(
                f"weights has {len(weights)} series, binary_targets has {len(binary_targets)}"
            )
        if count_targets is not None and len(count_targets) != len(binary_targets):
            raise ValueError(
                f"count_targets has {len(count_targets)} series, "
                f"binary_targets has {len(binary_targets)}"
            )
        if min_positive_samples is not None and len(binary_targets) != 1:
            raise ValueError(
                "min_positive_samples reproduces the per-region (local) paradigm, "
                f"which runs one region at a time; got {len(binary_targets)} series"
            )

        self.classifier_builder = classifier_builder
        self.regressor_builder = regressor_builder
        self.binary_targets = list(binary_targets)
        self.count_targets = None if count_targets is None else list(count_targets)
        self.weights = list(weights)
        self.clf_past = clf_past
        self.clf_future = clf_future
        self.reg_past = reg_past
        self.reg_future = reg_future
        self.min_positive_samples = min_positive_samples

        self._clf: Any = None
        self._reg: Any = None
        # `use_dummy = False` / `dummy_mean = 1.0` before the loop, cell 34.
        self._use_dummy: bool = False
        self._dummy_mean: Any = 1.0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        past_covs: list[TimeSeries] | None,
        future_covs: list[TimeSeries] | None,
    ) -> None:
        """No-op: the hurdle carries FOUR covariate lists, given to ``__init__``.

        The engine's single ``(past, future)`` pair cannot express the
        classifier/regressor split, so it is IGNORED rather than silently used
        for one of the two heads. Passing covariates to
        :meth:`ExpandingWindowBacktest.iter_folds` alongside a
        ``HurdleForecaster`` is therefore a wiring mistake; it is logged at
        DEBUG rather than raised, so an engine call that hands covariates to a
        mixed lineup of forecasters still works.
        """
        if past_covs is not None or future_covs is not None:
            logger.debug(
                "HurdleForecaster.prepare ignoring the engine's covariates "
                "(past=%s, future=%s series); the four hurdle covariate lists "
                "come from the constructor",
                None if past_covs is None else len(past_covs),
                None if future_covs is None else len(future_covs),
            )

    def fit(self, train_series: list[TimeSeries], *, cutoff: pd.Timestamp) -> None:
        """Rebuild and fit both heads on the slices ending before ``cutoff``."""
        train_c = [ts.drop_after(cutoff) for ts in self.binary_targets]
        train_r = self._count_slice(train_series, cutoff)

        self._clf = self.classifier_builder()
        self._clf.fit(
            series=train_c,
            past_covariates=self.clf_past,
            future_covariates=self.clf_future,
        )

        if self.min_positive_samples is None:
            train_weights = [w.drop_after(cutoff) for w in self.weights]
            self._use_dummy = False
            self._reg = self.regressor_builder()
            self._reg.fit(
                series=train_r,
                past_covariates=self.reg_past,
                future_covariates=self.reg_future,
                sample_weight=train_weights,
            )
            return

        # --- cell 34: adaptive dummy fallback, per-region paradigm only ---
        w_series = self.weights[0].drop_after(cutoff)
        n_positive = (w_series.values() > 0).sum()
        if n_positive < self.min_positive_samples:
            self._use_dummy = True
            target_vals = train_r[0].values()
            pos_vals = target_vals[w_series.values() > 0]
            # `pos_vals.mean()` is a numpy scalar (np.float64); the `else`
            # branch is the Python float 1.0. Both are kept as-is, because
            # `np.full((n, 1), dummy_mean)` below inherits the dtype.
            self._dummy_mean = pos_vals.mean() if len(pos_vals) > 0 else 1.0
            logger.info(
                "  [cutoff=%s] Only %s samples. Using Dummy Regressor (mean=%.2f).",
                cutoff.date(),
                n_positive,
                self._dummy_mean,
            )
        else:
            self._use_dummy = False
            self._reg = self.regressor_builder()
            self._reg.fit(
                series=train_r,
                past_covariates=self.reg_past,
                future_covariates=self.reg_future,
                sample_weight=[w_series],
            )

    def predict(
        self,
        n: int,
        context_series: list[TimeSeries],
        *,
        cutoff: pd.Timestamp,
    ) -> dict[str, list[TimeSeries]]:
        """One fold of predictions: ``prob``, ``count`` and ``prob * count``."""
        if self._clf is None:
            raise RuntimeError("HurdleForecaster.predict called before fit")

        ctx_c = [ts.drop_after(cutoff) for ts in self.binary_targets]
        ctx_r = self._count_slice(context_series, cutoff)

        preds_c = list(
            self._clf.predict(
                n=n,
                series=ctx_c,
                past_covariates=self.clf_past,
                future_covariates=self.clf_future,
                predict_likelihood_parameters=True,
                show_warnings=False,
            )
        )

        if self._use_dummy:
            # cell 34: the dummy regression prediction is built on the
            # CLASSIFIER's time index, not the regressor's. Preserved.
            preds_r = [
                TimeSeries.from_times_and_values(
                    preds_c[0].time_index,
                    np.full((n, 1), self._dummy_mean),
                )
            ]
        else:
            if self._reg is None:
                raise RuntimeError("HurdleForecaster.predict called before fit")
            preds_r = list(
                self._reg.predict(
                    n=n,
                    series=ctx_r,
                    past_covariates=self.reg_past,
                    future_covariates=self.reg_future,
                    show_warnings=False,
                )
            )

        probs_out: list[TimeSeries] = []
        counts_out: list[TimeSeries] = []
        hurdle_out: list[TimeSeries] = []
        for pred_c, pred_r in zip(preds_c, preds_r, strict=True):
            prob_ts = pred_c.univariate_component(pred_c.n_components - 1)
            probs = prob_ts.values().ravel()
            counts = pred_r.values().ravel()
            hurdle_ts = TimeSeries.from_times_and_values(
                pred_r.time_index, (probs * counts).reshape(-1, 1)
            )
            probs_out.append(prob_ts)
            counts_out.append(pred_r)
            hurdle_out.append(hurdle_ts)

        return {"prob": probs_out, "count": counts_out, "hurdle": hurdle_out}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _count_slice(
        self, engine_series: list[TimeSeries], cutoff: pd.Timestamp
    ) -> list[TimeSeries]:
        """The count-target slice: own series when given, else the engine's.

        The engine already hands ``[ts.drop_after(cutoff) for ts in targets]``,
        so when ``count_targets`` is ``None`` the two branches produce equal
        values; the branch exists so that a caller can drive a hurdle whose
        engine-level targets are something else.
        """
        if self.count_targets is None:
            return list(engine_series)
        return [ts.drop_after(cutoff) for ts in self.count_targets]


class MultiTargetClassifierForecaster:
    """One classifier per damage key, one output channel per key.

    Reproduces ``damage_classifier.ipynb`` cells 17 and 26: at every retrain a
    fresh classifier per key, fit on that key's targets sliced at the cutoff
    with that key's covariates; at every fold a prediction per key with
    ``predict_likelihood_parameters=True``, of which the LAST likelihood
    component per region is kept (``P(damage)``; see flag F64 for the
    ascending-class-order assumption that read makes).

    Parameters
    ----------
    builder:
        Zero-argument builder -- the notebook's ``get_damage_classifier`` --
        called once per key at every retrain.
    targets_by_key:
        ``key -> list[TimeSeries]``, FULL length, one list per damage key
        (``damage_classes[key]['get_covs_and_encodings']['TCV']`` in CV,
        ``Tr + V + Te`` appended in the test loop). Iteration order defines
        :attr:`channels` and the order the models are built in, which is
        ``damage_classes`` insertion order in the notebook.
    past_by_key, future_by_key:
        ``key -> list[TimeSeries] | None`` (``'FPC'`` / ``'FFC'``), FULL length,
        never sliced.

    Notes
    -----
    The engine's ``train_series`` and ``context_series`` are IGNORED: every key
    has its own target list and this forecaster slices all of them with
    ``drop_after(cutoff)`` itself. Only the cutoff is used. The legacy loops
    take the schedule (``n_total``, ``start_idx``, ``ref_ts``) from the FIRST
    key alone and apply it to every key, so the caller must hand the engine that
    first key's list as ``level_targets``; the other keys' series are assumed to
    share its time index (flag F68).
    """

    retrains: bool = True

    def __init__(
        self,
        builder: Callable[[], Any],
        targets_by_key: Mapping[str, list[TimeSeries]],
        past_by_key: Mapping[str, list[TimeSeries] | None],
        future_by_key: Mapping[str, list[TimeSeries] | None],
    ) -> None:
        if not targets_by_key:
            raise ValueError("targets_by_key is empty")
        missing = [k for k in targets_by_key if k not in past_by_key or k not in future_by_key]
        if missing:
            raise ValueError(f"no covariates for damage keys {missing}")

        self.builder = builder
        self.targets_by_key = dict(targets_by_key)
        self.past_by_key = dict(past_by_key)
        self.future_by_key = dict(future_by_key)
        self.channels: tuple[str, ...] = tuple(self.targets_by_key)
        self._models: dict[str, Any] = dict.fromkeys(self.targets_by_key)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        past_covs: list[TimeSeries] | None,
        future_covs: list[TimeSeries] | None,
    ) -> None:
        """No-op: covariates are per key and come from ``__init__``.

        Logged at DEBUG when the engine passes covariates anyway, for the same
        reason as :meth:`HurdleForecaster.prepare`.
        """
        if past_covs is not None or future_covs is not None:
            logger.debug(
                "MultiTargetClassifierForecaster.prepare ignoring the engine's "
                "covariates (past=%s, future=%s series); covariates are per "
                "damage key and come from the constructor",
                None if past_covs is None else len(past_covs),
                None if future_covs is None else len(future_covs),
            )

    def fit(self, train_series: list[TimeSeries], *, cutoff: pd.Timestamp) -> None:
        """Rebuild and fit one classifier per key. ``train_series`` is unused."""
        for key, targets in self.targets_by_key.items():
            train_d = [ts.drop_after(cutoff) for ts in targets]
            model = self.builder()
            model.fit(
                series=train_d,
                past_covariates=self.past_by_key[key],
                future_covariates=self.future_by_key[key],
            )
            self._models[key] = model

    def predict(
        self,
        n: int,
        context_series: list[TimeSeries],
        *,
        cutoff: pd.Timestamp,
    ) -> dict[str, list[TimeSeries]]:
        """``P(damage)`` per key per region. ``context_series`` is unused."""
        out: dict[str, list[TimeSeries]] = {}
        for key, targets in self.targets_by_key.items():
            model = self._models[key]
            if model is None:
                raise RuntimeError(
                    f"MultiTargetClassifierForecaster.predict called before fit (key {key!r})"
                )
            pred_series_d = [ts.drop_after(cutoff) for ts in targets]
            preds_d = model.predict(
                n=n,
                series=pred_series_d,
                past_covariates=self.past_by_key[key],
                future_covariates=self.future_by_key[key],
                predict_likelihood_parameters=True,
                show_warnings=False,
            )
            out[key] = [
                pred_d.univariate_component(pred_d.n_components - 1) for pred_d in preds_d
            ]
        return out
