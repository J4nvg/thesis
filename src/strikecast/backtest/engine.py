"""The one expanding-window backtest loop.

``ExpandingWindowBacktest`` replaces, with identical semantics, every legacy
runner listed in :mod:`strikecast.backtest.protocols`. It knows nothing about
darts models, covariate scaling, likelihoods or un-differencing: it drives the
:class:`~strikecast.backtest.protocols.Forecaster` protocol and applies a
:class:`~strikecast.backtest.protocols.TargetTransform`.

Per backtest, in this order (the order the legacy runners use):

1. ``transform.forward(level_targets)`` ONCE -- the diff family's
   ``Diff(lags=1, dropna=True).fit_transform(...)`` step;
2. ``forecaster.prepare(past_covs, future_covs)`` ONCE -- the adapters' home for
   ``_maybe_scale_covs`` (flag F1: the neural ``Scaler`` is fit on the
   full-length covariates, including validation and test);
3. the schedule, computed on the MODEL-space reference ``model_targets[0]``;
4. per fold: retrain slice, context slice, predict, inverse-transform, collect.

Both slices are ``[ts.drop_after(cutoff) for ts in model_targets]`` and both are
recomputed independently, exactly as ``run_expanding_cv`` does. They are equal
in value; keeping two objects preserves the legacy object identity (nothing
downstream mutates them, but the adapters receive fresh series either way).

The engine works unchanged when ``past_covs``/``future_covs`` are ``None``:
they are handed to ``prepare`` verbatim and the adapter decides.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from .schedule import schedule_from_config

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from darts import TimeSeries

    from strikecast.config.schema import BacktestConfig

    from .protocols import FoldHook, Forecaster, TargetTransform

from .protocols import FoldResult

__all__ = ["ExpandingWindowBacktest"]

logger = logging.getLogger(__name__)


class ExpandingWindowBacktest:
    """Expanding-window backtest with decoupled predict and retrain strides.

    Parameters
    ----------
    config:
        Start fraction, horizon, predict stride and retrain stride.
    transform:
        ``identity`` for the count / hurdle / damage families, ``diff`` for the
        diff branch. ``forward`` runs once, ``inverse`` runs per fold, per
        region, per channel, and always receives the FULL level-space series of
        that region as context -- that is where ``_diff_to_level`` looks up its
        anchor.
    hooks:
        Called after every fold, in the order given, with the fold result and a
        snapshot of the cumulative predictions.
    """

    def __init__(
        self,
        config: BacktestConfig,
        transform: TargetTransform,
        hooks: Sequence[FoldHook] = (),
    ) -> None:
        self.config = config
        self.transform = transform
        self.hooks: tuple[FoldHook, ...] = tuple(hooks)

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def iter_folds(
        self,
        forecaster: Forecaster,
        level_targets: list[TimeSeries],
        past_covs: list[TimeSeries] | None = None,
        future_covs: list[TimeSeries] | None = None,
        *,
        model_targets: list[TimeSeries] | None = None,
    ) -> Iterator[FoldResult]:
        """Yield one :class:`FoldResult` per fold, in schedule order.

        The generator twin of :meth:`run`, and the exact shape
        ``run_expanding_cv_iter`` has: the Optuna objective consumes it fold by
        fold, scores the cumulative predictions and prunes.

        ``model_targets`` overrides ``transform.forward(level_targets)``. The
        legacy diff family needs it for the VALIDATION stage (flag F80): the
        script differences the full series and only then takes the CV view, so
        its model-space CV list (``target_for_cv_diff``, 676 steps on the real
        panel) is one step longer than differencing the level CV view (675).
        Passing the legacy pair ``level_targets=target_for_cv`` and
        ``model_targets=target_for_cv_diff`` reproduces the legacy runner
        exactly; ``forward`` of the full series equals the legacy test-stage
        list, so the test stage needs no override.

        The cumulative snapshot passed to the hooks is a fresh outer/middle
        list-of-lists (``[list(region) for region in ...]``), like the diff
        family's iterator yields. The count family's ``run_expanding_cv_iter``
        yields the live object instead; copying is safe for both, because the
        legacy consumers only read it (flag F31).
        """
        if not level_targets:
            raise ValueError("level_targets is empty")

        cfg = self.config

        if model_targets is None:
            model_targets = self.transform.forward(list(level_targets))
        else:
            model_targets = list(model_targets)
        if len(model_targets) != len(level_targets):
            raise ValueError(
                f"model-space list has {len(model_targets)} series for "
                f"{len(level_targets)} targets"
            )

        forecaster.prepare(past_covs, future_covs)

        reference = model_targets[0]
        folds = schedule_from_config(reference, cfg)

        channels = tuple(forecaster.channels)
        cumulative: dict[str, list[list[TimeSeries]]] = {
            channel: [[] for _ in level_targets] for channel in channels
        }

        n_retrains = 0
        for fold in folds:
            cutoff = fold.cutoff

            if fold.retrain and forecaster.retrains:
                train_series = [ts.drop_after(cutoff) for ts in model_targets]
                forecaster.fit(train_series, cutoff=cutoff)
                n_retrains += 1
                # legacy prints `split_time.date()`; keep that wording, but do
                # not crash on an integer-indexed series the way legacy would.
                label = cutoff.date() if isinstance(cutoff, pd.Timestamp) else cutoff
                logger.info("   retrain %d  (data up to %s)", n_retrains, label)

            context = [ts.drop_after(cutoff) for ts in model_targets]
            model_preds = forecaster.predict(cfg.horizon, context, cutoff=cutoff)

            if set(model_preds) != set(channels):
                raise ValueError(
                    f"forecaster declared channels {sorted(channels)} but predict "
                    f"returned {sorted(model_preds)}"
                )

            predictions: dict[str, list[TimeSeries]] = {}
            for channel in channels:
                region_preds = model_preds[channel]
                if len(region_preds) != len(level_targets):
                    raise ValueError(
                        f"channel {channel!r} returned {len(region_preds)} series for "
                        f"{len(level_targets)} regions"
                    )
                predictions[channel] = [
                    self.transform.inverse(pred, level_targets[r_idx])
                    for r_idx, pred in enumerate(region_preds)
                ]

            for channel, region_preds in predictions.items():
                for r_idx, pred in enumerate(region_preds):
                    cumulative[channel][r_idx].append(pred)

            result = FoldResult(fold=fold, predictions=predictions)

            snapshot = {
                channel: [list(region_preds) for region_preds in regions]
                for channel, regions in cumulative.items()
            }
            for hook in self.hooks:
                hook.on_fold(result, snapshot)

            yield result

        logger.info("   %d predictions, %d retrains complete", len(folds), n_retrains)

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def run(
        self,
        forecaster: Forecaster,
        level_targets: list[TimeSeries],
        past_covs: list[TimeSeries] | None = None,
        future_covs: list[TimeSeries] | None = None,
        *,
        model_targets: list[TimeSeries] | None = None,
    ) -> dict[str, list[list[TimeSeries]]]:
        """Exhaust :meth:`iter_folds` and return the cumulative predictions.

        ``model_targets`` is forwarded to :meth:`iter_folds` (flag F80).

        The value is ``channel -> all_fold_preds``, where ``all_fold_preds`` is
        the legacy shape: outer list by region (in the order of
        ``level_targets``), inner list by fold. Single-output forecasters put
        everything under :data:`~strikecast.backtest.protocols.SINGLE_CHANNEL`,
        so ``run(...)[SINGLE_CHANNEL]`` is byte-for-byte what
        ``run_expanding_cv`` / ``run_final_test`` returns and what
        ``collect_predictions_long`` consumes.
        """
        collected: dict[str, list[list[TimeSeries]]] = {
            channel: [[] for _ in level_targets] for channel in forecaster.channels
        }
        for result in self.iter_folds(
            forecaster, level_targets, past_covs, future_covs, model_targets=model_targets
        ):
            for channel, region_preds in result.predictions.items():
                for r_idx, pred in enumerate(region_preds):
                    collected[channel][r_idx].append(pred)
        return collected
