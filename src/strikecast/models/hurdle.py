"""Assembly helpers for the two composite families (plan §5.2, phase P5).

:mod:`strikecast.models.composite` holds the two forecasters; this module holds
the *wiring*: which series list of which :class:`~strikecast.data.series.SeriesBundle`
becomes which constructor argument, per stage and per paradigm. It is the one
place that knows, for example, that the hurdle's CV stage takes the classifier
bundle's ``target_cv_view`` as ``binary_targets`` and the regressor bundle's
un-encoded full target list as the source of the positive-only sample weights.

Why a separate module
---------------------
``pipeline/composite_stage.py`` would otherwise carry ~150 lines of series
plumbing next to its store, calibration and metric logic, and the equivalence
tests would have no importable name for "the legacy wiring". Everything here is
pure: it reads bundles and returns forecasters, touches no store and no config
beyond the fractions the bundles already carry.

Provenance
----------

======================================  ===========================================
here                                    legacy
======================================  ===========================================
:func:`hurdle_series` ``stage="cv"``    ``final_hurdle.ipynb`` cells 10-11 + 15
:func:`hurdle_series` ``stage="test"``  cell 24's ``full_target_c`` / ``full_target_r``
                                        / ``full_weights_final`` preamble
:func:`damage_series`                   ``damage_classifier.ipynb`` cells 13, 17, 26
:data:`LOCAL_MIN_POSITIVE_SAMPLES`      cell 34's ``MIN_POSITIVE_SAMPLES`` (F62)
======================================  ===========================================

Flags this module preserves
---------------------------
* **F60.** The legacy loops take the schedule from the BINARY list while the
  engine takes it from whatever it is handed. :attr:`HurdleSeries.level_targets`
  is the COUNT list, as in ``tests/equivalence/test_hurdle_runners.py``; the two
  lists come out of the same split and share a time index, so the schedules are
  identical.
* **F62.** ``min_positive_samples`` is ``50`` for the local paradigm and
  ``None`` everywhere else -- cell 34 is the only loop that has the branch.
* **F63.** The four covariate lists are full length and never sliced or scaled:
  the hurdle family has no neural models, so the legacy loop passes the notebook
  globals straight through to ``fit`` and ``predict``.
* **F68.** Both damage loops take the schedule from the FIRST key only and apply
  it to every key, which is why :attr:`DamageSeries.level_targets` is that first
  key's list.
* The CV weights are the FULL-length ``make_positive_only_weights(
  target_series_list_r)`` of cell 10 -- computed on the *un-encoded* target list,
  before cell 11 re-runs the encoding -- while the test weights are recomputed
  from the encoded full targets (cell 24). Both are reproduced as written; the
  values are identical and only the static covariates of the weight series
  differ, which no darts code path reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from strikecast.data.series import positive_only_weights
from strikecast.models.composite import (
    LEGACY_MIN_POSITIVE_SAMPLES,
    HurdleForecaster,
    MultiTargetClassifierForecaster,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping, Sequence

    from darts import TimeSeries

    from strikecast.data.series import SeriesBundle

__all__ = [
    "DAMAGE_LABEL_PREFIX",
    "DAMAGE_LABEL_SUFFIX",
    "LOCAL_MIN_POSITIVE_SAMPLES",
    "DamageSeries",
    "HurdleSeries",
    "damage_label",
    "damage_series",
    "hurdle_series",
    "make_damage_forecaster",
    "make_hurdle_forecaster",
    "min_positive_samples_for",
]

#: ``MIN_POSITIVE_SAMPLES`` of ``final_hurdle.ipynb`` cell 34 (F62). Re-exported
#: from :mod:`strikecast.models.composite` so callers need one import.
LOCAL_MIN_POSITIVE_SAMPLES = LEGACY_MIN_POSITIVE_SAMPLES

#: What the damage notebook strips to make a printable label
#: (``key.replace(...)`` in cells 19, 24, 28 and 29).
DAMAGE_LABEL_PREFIX = "act_drone_infra_ua_"
DAMAGE_LABEL_SUFFIX = "_intent"


def _stage_targets(bundle: SeriesBundle, stage: str) -> list[TimeSeries]:
    """``target_cv_view`` for ``cv``, ``target_full`` for ``test``.

    ``target_full`` is the encoded full series; the legacy test loops build
    ``[tr.append(vl).append(te)]`` from the same split, which has the same index
    and the same values. Using ``target_full`` keeps one spelling shared with
    :func:`strikecast.pipeline.run_stage.stage_targets`.
    """
    if stage == "cv":
        return list(bundle.target_cv_view)
    if stage == "test":
        return list(bundle.target_full)
    raise KeyError(f"unknown stage {stage!r}; expected 'cv' or 'test'")


def min_positive_samples_for(paradigm: str) -> int | None:
    """``50`` for the local paradigm, ``None`` for global and activity (F62)."""
    return LOCAL_MIN_POSITIVE_SAMPLES if str(paradigm) == "local" else None


def damage_label(key: str) -> str:
    """``act_drone_infra_ua_health_intent`` -> ``health`` (notebook cell 19)."""
    return key.replace(DAMAGE_LABEL_PREFIX, "").replace(DAMAGE_LABEL_SUFFIX, "")


# --------------------------------------------------------------------------- #
# hurdle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HurdleSeries:
    """Every series list one hurdle stage needs, in legacy spelling.

    ``binary_targets`` / ``count_targets`` are the stage's two target lists;
    ``weights`` is full length in both stages, because
    :class:`HurdleForecaster` slices it with ``drop_after(cutoff)`` itself.

    ``actuals`` is what the ``y_true`` column of each channel is taken from:
    the BINARY list for ``prob`` and the COUNT list for ``count`` and
    ``hurdle``, exactly as ``final_hurdle.ipynb`` cell 16 calls
    ``collect_predictions_long``.
    """

    region_names: list[str]
    binary_targets: list[TimeSeries]
    count_targets: list[TimeSeries]
    weights: list[TimeSeries]
    clf_past: list[TimeSeries]
    clf_future: list[TimeSeries]
    reg_past: list[TimeSeries]
    reg_future: list[TimeSeries]

    @property
    def level_targets(self) -> list[TimeSeries]:
        """What the engine schedules and inverse-transforms on (F60)."""
        return self.count_targets

    @property
    def actuals(self) -> dict[str, list[TimeSeries]]:
        """``channel -> the actuals its ``y_true`` comes from``."""
        return {
            "prob": self.binary_targets,
            "count": self.count_targets,
            "hurdle": self.count_targets,
        }


def hurdle_series(
    classifier: SeriesBundle,
    regressor: SeriesBundle,
    stage: str,
) -> HurdleSeries:
    """Assemble one hurdle stage from the two post-selection bundles.

    Parameters
    ----------
    classifier:
        The bundle built on the BINARISED panel (``<target>_binary``), subset
        with the event head's own top-100 (``final_hurdle.ipynb`` cell 11).
    regressor:
        The bundle built on the unbinarised count panel, subset with the count
        head's top-100.
    stage:
        ``"cv"`` or ``"test"``.

    Notes
    -----
    The CV weights come from ``regressor.raw.target``, the **un-encoded** full
    target list -- that is literally ``make_positive_only_weights(
    target_series_list_r)`` of cell 10, which runs before cell 11 re-encodes.
    The test weights come from the encoded full targets, which is cell 24's
    ``make_positive_only_weights(full_target_r)``. Values are identical either
    way; the spelling is kept so the provenance stays readable.
    """
    if classifier.region_names != regressor.region_names:
        raise ValueError(
            "the hurdle's two bundles disagree on the region order: "
            f"{classifier.region_names} vs {regressor.region_names}"
        )

    binary_targets = _stage_targets(classifier, stage)
    count_targets = _stage_targets(regressor, stage)

    if stage == "cv":
        if regressor.raw is None:
            raise ValueError(
                "the hurdle CV stage needs the count bundle's un-encoded target list "
                "(cell 10's `make_positive_only_weights(target_series_list_r)`); "
                "this bundle has raw=None"
            )
        weights = positive_only_weights(regressor.raw.target)
    else:
        weights = positive_only_weights(count_targets)

    return HurdleSeries(
        region_names=list(regressor.region_names),
        binary_targets=binary_targets,
        count_targets=count_targets,
        weights=weights,
        clf_past=list(classifier.past_covs),
        clf_future=list(classifier.future_covs),
        reg_past=list(regressor.past_covs),
        reg_future=list(regressor.future_covs),
    )


def make_hurdle_forecaster(
    builders: Any,
    series: HurdleSeries,
    *,
    indices: Sequence[int] | None = None,
    min_positive_samples: int | None = None,
) -> HurdleForecaster:
    """Build the forecaster for one group of regions.

    ``builders`` is what ``get_spec("hurdle").build(params, ctx)`` returns -- a
    :class:`~strikecast.models.classifiers.HurdleBuilders` pair, or any
    two-element iterable of zero-argument builders.

    ``indices`` selects the regions of one group (``None`` means all of them),
    which is what the ``*_per_activity`` and ``*_per_region`` wrappers do by
    slicing every list with the same index list.

    ``count_targets`` is always passed explicitly rather than relying on the
    engine's list. The two are equal for every wiring this pipeline produces
    (the engine is handed :attr:`HurdleSeries.level_targets`), and being
    explicit means a future caller cannot silently drive the count head off a
    different list from the one the ``count`` channel is scored against.
    """
    classifier_builder, regressor_builder = tuple(builders)

    def pick(values: Sequence[TimeSeries]) -> list[TimeSeries]:
        return list(values) if indices is None else [values[i] for i in indices]

    return HurdleForecaster(
        classifier_builder,
        regressor_builder,
        binary_targets=pick(series.binary_targets),
        count_targets=pick(series.count_targets),
        weights=pick(series.weights),
        clf_past=pick(series.clf_past),
        clf_future=pick(series.clf_future),
        reg_past=pick(series.reg_past),
        reg_future=pick(series.reg_future),
        min_positive_samples=min_positive_samples,
    )


# --------------------------------------------------------------------------- #
# damage
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DamageSeries:
    """Every series list one damage stage needs, one entry per damage key.

    Iteration order of :attr:`targets_by_key` is the ``damage_classes``
    insertion order of the notebook, which is also the channel order of
    :class:`MultiTargetClassifierForecaster` and therefore the order the
    per-key metric components are written in.
    """

    region_names: list[str]
    keys: list[str]
    targets_by_key: dict[str, list[TimeSeries]]
    past_by_key: dict[str, list[TimeSeries]]
    future_by_key: dict[str, list[TimeSeries]]

    @property
    def level_targets(self) -> list[TimeSeries]:
        """The FIRST key's list: the schedule every key is run on (F68)."""
        return self.targets_by_key[self.keys[0]]

    @property
    def actuals(self) -> dict[str, list[TimeSeries]]:
        """``channel -> actuals``; each key is scored against its own target."""
        return dict(self.targets_by_key)


def damage_series(bundles: Mapping[str, SeriesBundle], stage: str) -> DamageSeries:
    """Assemble one damage stage from the per-key post-selection bundles.

    ``bundles`` is ``damage key -> SeriesBundle``; its iteration order is kept.
    """
    if not bundles:
        raise ValueError("damage_series needs at least one damage key")

    keys = list(bundles)
    first = bundles[keys[0]]
    for key in keys[1:]:
        if bundles[key].region_names != first.region_names:
            raise ValueError(
                f"damage key {key!r} disagrees with {keys[0]!r} on the region order"
            )

    return DamageSeries(
        region_names=list(first.region_names),
        keys=keys,
        targets_by_key={k: _stage_targets(bundles[k], stage) for k in keys},
        past_by_key={k: list(bundles[k].past_covs) for k in keys},
        future_by_key={k: list(bundles[k].future_covs) for k in keys},
    )


def make_damage_forecaster(
    builders: Any,
    series: DamageSeries,
    *,
    indices: Sequence[int] | None = None,
) -> MultiTargetClassifierForecaster:
    """Build the per-key forecaster for one group of regions.

    ``builders`` is what ``get_spec("damage").build(params, ctx)`` returns -- a
    :class:`~strikecast.models.classifiers.DamageBuilders` wrapper, or any
    one-element iterable holding the zero-argument builder. A bare callable is
    also accepted.
    """
    builder: Callable[[], Any]
    if callable(builders) and not hasattr(builders, "classifier_builder"):
        builder = builders
    else:
        (builder,) = tuple(builders)

    def pick(values: Sequence[TimeSeries]) -> list[TimeSeries]:
        return list(values) if indices is None else [values[i] for i in indices]

    return MultiTargetClassifierForecaster(
        builder,
        {k: pick(v) for k, v in series.targets_by_key.items()},
        {k: pick(v) for k, v in series.past_by_key.items()},
        {k: pick(v) for k, v in series.future_by_key.items()},
    )
