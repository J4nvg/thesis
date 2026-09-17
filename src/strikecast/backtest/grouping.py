"""Paradigm partitioning: Global, Activity and Local, in one place.

Replaces four near-identical wrapper pairs:

* ``src/evaluation_tools.py::run_expanding_cv_per_activity`` /
  ``::run_expanding_cv_per_region`` (lines 593-694),
* ``_diff_regression.py::run_final_test_diff_per_activity`` /
  ``::run_final_test_diff_per_region`` (lines 951-1005),
* ``final_hurdle.ipynb`` cells 29 and 34
  (``run_hurdle_cv_per_activity`` / ``_per_region`` and the two
  ``run_final_test_per_*`` twins),
* the damage classifier notebook, which only ever runs Global.

Every one of them does the same three things and differs only in what it calls
in the middle:

1. **partition** the region indices — one group for Global, one per activity
   level for Activity, one per region for Local;
2. **slice** the target list and each covariate list down to a group, with
   ``None`` covariates passed straight through;
3. **scatter** the per-group ``fold_preds`` back to the original region
   positions (``all_fold_preds[orig_idx] = group_preds[group_idx]``), so the
   caller's ``collect_predictions_long`` sees the regions in bundle order.

:func:`run_grouped` takes the middle step as a callable, so the engine
(``ExpandingWindowBacktest.run``) plugs in without this module importing it.

Behaviour-preservation notes (plan §6):

* ``Q1`` the Activity grouping is ``defaultdict(list)`` filled in region order
  and then iterated as ``for level in sorted(groups)``. Both are reproduced:
  within a group the region indices stay in bundle order, and the groups come
  out sorted by activity level, not by first appearance. Group order has no
  effect on the scattered result, but it does decide fit order, which decides
  the RNG draw sequence of a stochastic model, so it is preserved.
* ``Q2`` the legacy lookup is ``regions_activity[region]`` on a plain dict, so
  a region absent from the activity map raises ``KeyError``. That is preserved
  verbatim (it is a hard failure, not a silent drop). Note this is a different
  code path from flag F24, where the *activity tier filter* keeps NaN-tiered
  regions; by the time a runner partitions, every region in the list must have
  a tier.
* ``Q3`` ``take(None, indices)`` returns ``None``, mirroring
  ``[past_covs[i] for i in indices] if past_covs is not None else None``.
* ``Q4`` the Local wrappers call the runner with a ONE-element list per region
  and keep ``group_preds[0]``. That is exactly :func:`partition` with
  ``paradigm="local"`` plus :func:`restore_order`; the wrappers' ``append``
  and the Activity wrappers' indexed assignment produce the same list because
  Local groups are visited in region order.
* ``Q5`` the hurdle Local wrappers additionally carry a
  ``MIN_POSITIVE_SAMPLES = 50`` dummy-regressor fallback INSIDE the retrain
  branch (``final_hurdle.ipynb`` cells 29/34: when the sliced weight series has
  fewer than 50 positive samples the count head is replaced by a constant equal
  to the mean of the positive targets). That is a property of the hurdle
  forecaster, not of the partitioning, so it lives in the composite forecaster
  and deliberately has no counterpart here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from darts import TimeSeries

__all__ = [
    "Group",
    "Paradigm",
    "RunOne",
    "partition",
    "restore_order",
    "run_grouped",
    "take",
]

#: The three training paradigms of the thesis.
Paradigm = Literal["global", "activity", "local"]


@dataclass(frozen=True, slots=True)
class Group:
    """One set of regions that share a fitted model.

    ``label`` names the group for logging and for the run store: ``"global"``,
    the activity level as a string, or the region name. ``indices`` are
    positions into the caller's region-ordered lists, ascending.
    """

    label: str
    indices: list[int]


#: What :func:`run_grouped` calls per group: ``(group, targets, past, future)``
#: to ``{channel: [[pred per fold] per group member]}``.
RunOne = Callable[
    [
        Group,
        list["TimeSeries"],
        list["TimeSeries"] | None,
        list["TimeSeries"] | None,
    ],
    dict[str, list[list["TimeSeries"]]],
]


def partition(
    region_names: Sequence[str],
    paradigm: Paradigm,
    activity_by_region: Mapping[str, Any] | None = None,
) -> list[Group]:
    """Split region positions into the groups one paradigm trains separately.

    ``global`` gives a single group holding every index in order; ``activity``
    gives one group per distinct activity level in ``sorted()`` order (Q1),
    each holding its regions in bundle order; ``local`` gives one group per
    region, in bundle order.

    Raises ``KeyError`` for a region missing from ``activity_by_region`` (Q2)
    and ``ValueError`` when ``activity_by_region`` is omitted entirely or the
    paradigm is unknown.
    """
    if paradigm == "global":
        return [Group("global", list(range(len(region_names))))]

    if paradigm == "local":
        return [Group(str(name), [i]) for i, name in enumerate(region_names)]

    if paradigm == "activity":
        if activity_by_region is None:
            raise ValueError("paradigm='activity' needs activity_by_region")
        groups: defaultdict[Any, list[int]] = defaultdict(list)
        for i, region in enumerate(region_names):
            # Legacy: `groups[regions_activity[region]].append(i)` -- a missing
            # region raises KeyError here, and that is preserved (Q2).
            groups[activity_by_region[region]].append(i)
        return [Group(str(level), groups[level]) for level in sorted(groups)]

    raise ValueError(f"unknown paradigm {paradigm!r}")


def take[T](items: Sequence[T] | None, indices: Sequence[int]) -> list[T] | None:
    """Slice a region-ordered list down to a group; ``None`` passes through (Q3)."""
    if items is None:
        return None
    return [items[i] for i in indices]


def restore_order(
    group_results: Sequence[tuple[Group, dict[str, list[list[TimeSeries]]]]],
    n_regions: int,
) -> dict[str, list[list[TimeSeries]]]:
    """Scatter per-group fold predictions back to original region positions.

    Reproduces ``all_fold_preds[orig_idx] = group_preds[group_idx]`` for every
    channel at once. Channels are taken in order of first appearance across
    groups; every group must produce the same channel set, and every region
    position must be filled exactly once, otherwise this raises. (The legacy
    wrappers leave an unfilled slot as ``None`` and fail later inside
    ``collect_predictions_long``; every reachable partition covers all regions,
    so the explicit error only ever fires on a caller bug.)
    """
    channels: list[str] = []
    for _group, preds in group_results:
        for channel in preds:
            if channel not in channels:
                channels.append(channel)

    out: dict[str, list[list[TimeSeries] | None]] = {
        channel: [None] * n_regions for channel in channels
    }
    for group, preds in group_results:
        if set(preds) != set(channels):
            raise ValueError(
                f"group {group.label!r} returned channels {list(preds)}, expected {channels}"
            )
        for channel, group_preds in preds.items():
            if len(group_preds) != len(group.indices):
                raise ValueError(
                    f"group {group.label!r} channel {channel!r}: "
                    f"{len(group_preds)} series for {len(group.indices)} regions"
                )
            for group_idx, orig_idx in enumerate(group.indices):
                if out[channel][orig_idx] is not None:
                    raise ValueError(f"region index {orig_idx} filled twice for channel {channel!r}")
                out[channel][orig_idx] = group_preds[group_idx]

    for channel, per_region in out.items():
        missing = [i for i, v in enumerate(per_region) if v is None]
        if missing:
            raise ValueError(f"channel {channel!r}: no predictions for region indices {missing}")

    return {channel: [p for p in per_region if p is not None] for channel, per_region in out.items()}


def run_grouped(
    run_one: RunOne,
    *,
    region_names: Sequence[str],
    paradigm: Paradigm,
    targets: Sequence[TimeSeries],
    past_covs: Sequence[TimeSeries] | None = None,
    future_covs: Sequence[TimeSeries] | None = None,
    activity_by_region: Mapping[str, Any] | None = None,
) -> dict[str, list[list[TimeSeries]]]:
    """Partition, run ``run_one`` per group on the sliced lists, restore order.

    The whole body of every legacy ``*_per_activity`` / ``*_per_region``
    wrapper, with the runner left as a parameter. ``paradigm="global"`` calls
    ``run_one`` once on the full lists, which is what the notebooks do inline.
    """
    groups = partition(region_names, paradigm, activity_by_region)
    results: list[tuple[Group, dict[str, list[list[TimeSeries]]]]] = []
    for group in groups:
        results.append(
            (
                group,
                run_one(
                    group,
                    [targets[i] for i in group.indices],
                    take(past_covs, group.indices),
                    take(future_covs, group.indices),
                ),
            )
        )
    return restore_order(results, len(targets))
