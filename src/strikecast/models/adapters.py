"""Darts model adapters implementing ``backtest.protocols.Forecaster``.

These are the fit/predict units the expanding-window engine drives. They hold
NO schedule logic: the engine owns ``t0``, ``drop_after(cutoff)`` and the
``retrain`` decision. What lives here is exactly the body of the legacy
runners' ``if retrain: ...`` and ``preds = ...`` blocks:

* ``src/evaluation_tools.py::run_expanding_cv``      (lines 275-365)  -- CV
* ``src/evaluation_tools.py::run_final_test``        (lines 368-450)  -- TEST
* ``src/evaluation_tools.py::run_expanding_cv_iter`` (lines 453-540)  -- TUNING
* ``_diff_regression.py::run_expanding_cv / run_expanding_cv_iter /
  run_final_test_diff`` (lines 680-942)  -- same three, plus the ARIMA
  ``NaiveMean`` fallback
* ``_regression_GBDT.py::run_manual_expanding_cv`` (lines 781-857) -- the
  inlined ancestor of the CV runner

The three legacy runners do NOT agree on post-processing, and the difference is
NOT documented anywhere in the thesis. Reproduced faithfully (flags F55/F56):

    runner                  median of 200 samples      exp() log-link
    ------------------------------------------------  --------------
    run_expanding_cv        yes (neural + likelihood)  yes
    run_final_test          NO                         yes
    run_expanding_cv_iter   yes (neural + likelihood)  NO

:meth:`GlobalDartsForecaster.for_cv`, :meth:`~GlobalDartsForecaster.for_test`
and :meth:`~GlobalDartsForecaster.for_tuning` are the only three combinations
that ever ran. The constructor defaults reproduce ``for_cv``; a caller has to
ask explicitly for anything else, and the four-th combination (no median, no
exp) is reachable only by passing both switches by hand.

Not implemented here, by design: a ``FixedForecaster`` for Chronos-2
(``_chronos2.py::chronos2_rolling_long``), which is never refit and would set
``retrains = False`` so the engine skips ``fit`` entirely. That is Phase 5 of
the plan; composite (hurdle / damage) forecasters live in a separate module
owned by another work stream.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from darts.dataprocessing.transformers import Scaler

from strikecast.backtest.protocols import SINGLE_CHANNEL

if TYPE_CHECKING:
    import pandas as pd
    from darts import TimeSeries

__all__ = ["GlobalDartsForecaster", "LocalDartsForecaster"]

logger = logging.getLogger(__name__)

#: Number of Monte-Carlo samples the legacy CV/tuning runners draw before
#: taking the median. Hard-coded as ``200`` in every copy of the loop.
LEGACY_NUM_SAMPLES = 200


def _maybe_scale_covs(
    past_covs: list[TimeSeries] | None,
    future_covs: list[TimeSeries] | None,
    do_scale: bool,
) -> tuple[list[TimeSeries] | None, list[TimeSeries] | None]:
    """Verbatim ``src/evaluation_tools.py::_maybe_scale_covs`` (line 267).

    Two independent ``Scaler`` instances (darts default: ``MinMaxScaler``) fit
    on the FULL-LENGTH covariate lists, validation and test periods included.
    That is flag F1 and is preserved.

    One guard is added: legacy would raise if ``do_scale`` were true and a list
    were ``None`` (``Scaler().fit_transform(None)``), so no recorded run can
    have taken that path; we pass ``None`` through instead of crashing. See
    flag F53.
    """
    if not do_scale:
        return past_covs, future_covs
    ps = Scaler()
    fs = Scaler()
    scaled_past = (
        None if past_covs is None else cast(list["TimeSeries"], ps.fit_transform(past_covs))
    )
    scaled_future = (
        None if future_covs is None else cast(list["TimeSeries"], fs.fit_transform(future_covs))
    )
    return scaled_past, scaled_future


class GlobalDartsForecaster:
    """One darts model fit across all regions, rebuilt from scratch at retrain.

    Parameters
    ----------
    builder:
        Zero-argument callable returning a fresh, unfitted darts model. This is
        the legacy ``builder_fn``; it is called once per retrain, exactly like
        ``model = builder_fn()``.
    is_neural:
        Drives two separate legacy behaviours that happen to share the flag:
        covariate scaling in :meth:`prepare` (F1) and the ``is_probabilistic``
        guard in :meth:`predict`.
    sample_median:
        Draw ``num_samples`` Monte-Carlo paths and keep the 0.5 quantile, but
        only when ``is_neural`` AND the built model exposes a non-``None``
        ``likelihood``. ``True`` in the CV and tuning runners, ``False`` in the
        test runner.
    apply_log_link:
        Apply ``exp`` when the built model carries ``_count_log_link = True``
        (set by ``build_lstm_count`` for the Poisson/Tweedie log-link RNNs).
        ``True`` in the CV and test runners, ``False`` in the tuning runner.
    num_samples:
        Monte-Carlo sample count for ``sample_median``. Legacy hard-codes 200.
    """

    channels: tuple[str, ...] = (SINGLE_CHANNEL,)
    retrains: bool = True

    def __init__(
        self,
        builder: Callable[[], Any],
        *,
        is_neural: bool = False,
        sample_median: bool = True,
        apply_log_link: bool = True,
        num_samples: int = LEGACY_NUM_SAMPLES,
    ) -> None:
        self.builder = builder
        self.is_neural = is_neural
        self.sample_median = sample_median
        self.apply_log_link = apply_log_link
        self.num_samples = num_samples

        self.model: Any | None = None
        self.past_for_fit: list[TimeSeries] | None = None
        self.fut_for_fit: list[TimeSeries] | None = None

    # -- presets ------------------------------------------------------------
    @classmethod
    def for_cv(
        cls,
        builder: Callable[[], Any],
        *,
        is_neural: bool = False,
        num_samples: int = LEGACY_NUM_SAMPLES,
    ) -> GlobalDartsForecaster:
        """``run_expanding_cv``: median of 200 samples AND the ``exp`` log-link."""
        return cls(
            builder,
            is_neural=is_neural,
            sample_median=True,
            apply_log_link=True,
            num_samples=num_samples,
        )

    @classmethod
    def for_test(
        cls,
        builder: Callable[[], Any],
        *,
        is_neural: bool = False,
        num_samples: int = LEGACY_NUM_SAMPLES,
    ) -> GlobalDartsForecaster:
        """``run_final_test``: NO sampling step, ``exp`` log-link only (F55)."""
        return cls(
            builder,
            is_neural=is_neural,
            sample_median=False,
            apply_log_link=True,
            num_samples=num_samples,
        )

    @classmethod
    def for_tuning(
        cls,
        builder: Callable[[], Any],
        *,
        is_neural: bool = False,
        num_samples: int = LEGACY_NUM_SAMPLES,
    ) -> GlobalDartsForecaster:
        """``run_expanding_cv_iter``: median of 200 samples, NO log-link (F56)."""
        return cls(
            builder,
            is_neural=is_neural,
            sample_median=True,
            apply_log_link=False,
            num_samples=num_samples,
        )

    # -- lifecycle ----------------------------------------------------------
    def prepare(
        self,
        past_covs: list[TimeSeries] | None,
        future_covs: list[TimeSeries] | None,
    ) -> None:
        self.past_for_fit, self.fut_for_fit = _maybe_scale_covs(
            past_covs, future_covs, do_scale=self.is_neural
        )

    def fit(self, train_series: list[TimeSeries], *, cutoff: pd.Timestamp) -> None:
        """``model = builder_fn(); model.fit(**fit_kwargs)``.

        ``cutoff`` is unused: the engine has already applied
        ``drop_after(cutoff)`` to ``train_series``. It is part of the frozen
        protocol so composite forecasters can slice their auxiliary series.
        """
        del cutoff
        model = self.builder()
        fit_kwargs: dict[str, Any] = {"series": train_series}
        if self.past_for_fit is not None and model.supports_past_covariates:
            fit_kwargs["past_covariates"] = self.past_for_fit
        if self.fut_for_fit is not None and model.supports_future_covariates:
            fit_kwargs["future_covariates"] = self.fut_for_fit
        model.fit(**fit_kwargs)
        self.model = model

    def predict(
        self,
        n: int,
        context_series: list[TimeSeries],
        *,
        cutoff: pd.Timestamp,
    ) -> dict[str, list[TimeSeries]]:
        del cutoff
        model = self.model
        if model is None:
            raise RuntimeError(
                "GlobalDartsForecaster.predict called before fit; the engine must "
                "retrain on the first fold (legacy runners rely on the same)."
            )

        pred_kwargs: dict[str, Any] = {"n": n, "series": context_series}
        if self.past_for_fit is not None and model.supports_past_covariates:
            pred_kwargs["past_covariates"] = self.past_for_fit
        if self.fut_for_fit is not None and model.supports_future_covariates:
            pred_kwargs["future_covariates"] = self.fut_for_fit

        # Legacy: `is_neural and getattr(model, "likelihood", None) is not None`.
        # Every in-scope builder forces `likelihood = None`, so this branch never
        # fired in the recorded runs -- see flag F57.
        is_probabilistic = (
            self.sample_median
            and self.is_neural
            and getattr(model, "likelihood", None) is not None
        )
        if is_probabilistic:
            pred_kwargs["num_samples"] = self.num_samples

        preds = model.predict(show_warnings=False, **pred_kwargs)

        if is_probabilistic:
            preds = [p.quantile(0.5) for p in preds]  # median = point forecast
        if self.apply_log_link and getattr(model, "_count_log_link", False):
            preds = [p.map(np.exp) for p in preds]

        return {SINGLE_CHANNEL: list(preds)}


class LocalDartsForecaster:
    """One model per region, rebuilt and refit inside every prediction step.

    This is the ``is_local`` branch of the legacy runners. Note what it does
    NOT do: it never passes covariates (ARIMA is fit on the bare series), it
    never applies the median or the log-link post-processing, and it refits at
    EVERY fold, not only at retrain folds -- the retrain branch merely captures
    the builder (``_local_builder = builder_fn``). :meth:`fit` therefore only
    records the builder, and :meth:`predict` does the real work.

    ``fallback`` reproduces flag F5: the diff-family runners wrap the per-region
    build/fit/predict in ``try/except Exception`` and fall back to
    ``NaiveMean()``. The count-family runners have NO such guard, so the default
    is ``None`` and an exception propagates exactly as it would have.
    """

    channels: tuple[str, ...] = (SINGLE_CHANNEL,)
    retrains: bool = True

    def __init__(
        self,
        builder: Callable[[], Any],
        *,
        fallback: Callable[[], Any] | None = None,
    ) -> None:
        self.builder = builder
        self.fallback = fallback

        self.past_for_fit: list[TimeSeries] | None = None
        self.fut_for_fit: list[TimeSeries] | None = None
        self._active_builder: Callable[[], Any] | None = None

    def prepare(
        self,
        past_covs: list[TimeSeries] | None,
        future_covs: list[TimeSeries] | None,
    ) -> None:
        """Store the covariates unchanged; the local branch never uses them.

        Kept for protocol conformance and so a future local model that does
        take covariates has them available. Local models are never neural, so
        no ``Scaler`` is fit (legacy: ``_maybe_scale_covs(..., do_scale=False)``
        is an identity).
        """
        self.past_for_fit = past_covs
        self.fut_for_fit = future_covs

    def fit(self, train_series: list[TimeSeries], *, cutoff: pd.Timestamp) -> None:
        """No-op beyond capturing the builder (legacy ``_local_builder = builder_fn``)."""
        del train_series, cutoff
        self._active_builder = self.builder

    def predict(
        self,
        n: int,
        context_series: list[TimeSeries],
        *,
        cutoff: pd.Timestamp,
    ) -> dict[str, list[TimeSeries]]:
        del cutoff
        builder = self._active_builder or self.builder

        preds: list[TimeSeries] = []
        for r_idx, ts in enumerate(context_series):
            if self.fallback is None:
                m = builder()
                m.fit(ts)
                preds.append(m.predict(n=n))
                continue
            try:
                m = builder()
                m.fit(ts)
                pred = m.predict(n=n)
            except Exception:  # noqa: BLE001 - legacy catches bare Exception (F5)
                logger.warning(
                    "local model failed for region index %d; falling back to %s",
                    r_idx,
                    getattr(self.fallback, "__name__", repr(self.fallback)),
                    exc_info=True,
                )
                fallback = self.fallback()
                fallback.fit(ts)
                pred = fallback.predict(n=n)
            preds.append(pred)

        return {SINGLE_CHANNEL: preds}
