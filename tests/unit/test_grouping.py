"""Unit tests for the paradigm partitioning that replaces the legacy wrappers.

Everything here is pure bookkeeping, so the "series" are plain sentinel strings
except in the round-trip test, which drives `run_grouped` with a fake `run_one`
and checks that region order survives the partition/scatter cycle exactly as
``all_fold_preds[orig_idx] = group_preds[group_idx]`` does.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from strikecast.backtest.grouping import Group, partition, restore_order, run_grouped, take

REGIONS = ["kyiv", "odesa", "lviv", "kharkiv", "sumy"]
# deliberately unsorted values and unsorted first appearance: kyiv=3, odesa=1, ...
ACTIVITY = {"kyiv": 3, "odesa": 1, "lviv": 1, "kharkiv": 3, "sumy": 2}


# --------------------------------------------------------------------------
# partition
# --------------------------------------------------------------------------


def test_global_is_one_group_in_order() -> None:
    groups = partition(REGIONS, "global")
    assert groups == [Group("global", [0, 1, 2, 3, 4])]


def test_local_is_one_group_per_region_in_bundle_order() -> None:
    groups = partition(REGIONS, "local")
    assert [g.label for g in groups] == REGIONS
    assert [g.indices for g in groups] == [[0], [1], [2], [3], [4]]


def test_activity_groups_are_sorted_by_level_members_in_bundle_order() -> None:
    """Legacy Q1: `defaultdict` insertion order inside, `sorted(groups)` outside."""
    groups = partition(REGIONS, "activity", ACTIVITY)
    assert [g.label for g in groups] == ["1", "2", "3"]
    assert [g.indices for g in groups] == [[1, 2], [4], [0, 3]]
    # every index exactly once
    assert sorted(i for g in groups for i in g.indices) == list(range(len(REGIONS)))


def test_activity_missing_region_raises_keyerror() -> None:
    """Legacy Q2: `regions_activity[region]` on a plain dict raises, it does not skip."""
    incomplete = {k: v for k, v in ACTIVITY.items() if k != "sumy"}
    with pytest.raises(KeyError, match="sumy"):
        partition(REGIONS, "activity", incomplete)


def test_activity_without_map_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="activity_by_region"):
        partition(REGIONS, "activity", None)


def test_unknown_paradigm_raises() -> None:
    with pytest.raises(ValueError, match="unknown paradigm"):
        partition(REGIONS, cast(Any, "per_oblast"))


def test_empty_region_list() -> None:
    assert partition([], "global") == [Group("global", [])]
    assert partition([], "local") == []
    assert partition([], "activity", {}) == []


# --------------------------------------------------------------------------
# take
# --------------------------------------------------------------------------


def test_take_slices_and_passes_none_through() -> None:
    assert take(["a", "b", "c", "d"], [3, 1]) == ["d", "b"]
    assert take(None, [0, 1]) is None
    assert take([], []) == []


# --------------------------------------------------------------------------
# restore_order
# --------------------------------------------------------------------------


def _fake_preds(labels: list[str]) -> list[list[Any]]:
    """One 'fold list' per group member, tagged with the member's own label."""
    return [[f"{label}-f0", f"{label}-f1"] for label in labels]


def test_restore_order_scatters_back_to_bundle_order() -> None:
    groups = partition(REGIONS, "activity", ACTIVITY)
    results = [
        (g, {"y_pred": _fake_preds([REGIONS[i] for i in g.indices])}) for g in groups
    ]
    restored = restore_order(cast(Any, results), len(REGIONS))
    assert list(restored) == ["y_pred"]
    assert restored["y_pred"] == _fake_preds(REGIONS)


def test_restore_order_multi_channel() -> None:
    groups = partition(REGIONS, "local")
    results = [
        (
            g,
            {
                "prob": _fake_preds([f"p:{REGIONS[i]}" for i in g.indices]),
                "count": _fake_preds([f"c:{REGIONS[i]}" for i in g.indices]),
            },
        )
        for g in groups
    ]
    restored = restore_order(cast(Any, results), len(REGIONS))
    assert list(restored) == ["prob", "count"]
    assert restored["prob"] == _fake_preds([f"p:{r}" for r in REGIONS])
    assert restored["count"] == _fake_preds([f"c:{r}" for r in REGIONS])


def test_restore_order_rejects_inconsistent_channels() -> None:
    groups = partition(REGIONS, "activity", ACTIVITY)
    results: list[Any] = [(g, {"y_pred": _fake_preds([str(i) for i in g.indices])}) for g in groups]
    results[1] = (results[1][0], {"other": results[1][1]["y_pred"]})
    with pytest.raises(ValueError, match="expected"):
        restore_order(results, len(REGIONS))


def test_restore_order_rejects_wrong_group_size() -> None:
    group = Group("global", [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="for 5 regions"):
        restore_order(cast(Any, [(group, {"y_pred": _fake_preds(["a", "b"])})]), len(REGIONS))


def test_restore_order_rejects_uncovered_region() -> None:
    group = Group("partial", [0, 1])
    with pytest.raises(ValueError, match="no predictions for region indices"):
        restore_order(cast(Any, [(group, {"y_pred": _fake_preds(["a", "b"])})]), len(REGIONS))


# --------------------------------------------------------------------------
# run_grouped
# --------------------------------------------------------------------------


def _recording_run_one(seen: list[tuple[str, list[Any], Any, Any]]):
    def run_one(group, targets, past, future):
        seen.append((group.label, list(targets), past, future))
        return {"y_pred": _fake_preds([str(t) for t in targets])}

    return run_one


@pytest.mark.parametrize(
    ("paradigm", "activity", "labels"),
    [
        ("global", None, ["global"]),
        ("local", None, REGIONS),
        ("activity", ACTIVITY, ["1", "2", "3"]),
    ],
)
def test_run_grouped_round_trip(paradigm, activity, labels) -> None:
    targets = [f"t:{r}" for r in REGIONS]
    past = [f"p:{r}" for r in REGIONS]
    seen: list[tuple[str, list[Any], Any, Any]] = []

    restored = run_grouped(
        _recording_run_one(seen),
        region_names=REGIONS,
        paradigm=paradigm,
        targets=cast(Any, targets),
        past_covs=cast(Any, past),
        future_covs=None,
        activity_by_region=activity,
    )

    assert [label for label, *_ in seen] == labels
    # covariates are sliced with the same indices, `None` passes through (Q3)
    for (label, group_targets, group_past, group_future) in seen:
        assert group_future is None
        assert group_past == [f"p:{t.removeprefix('t:')}" for t in group_targets]
        assert label  # non-empty
    # and the scatter puts every region back where it started
    assert restored["y_pred"] == _fake_preds(targets)


def test_run_grouped_global_sees_the_full_lists() -> None:
    targets = [f"t:{r}" for r in REGIONS]
    seen: list[tuple[str, list[Any], Any, Any]] = []
    run_grouped(
        _recording_run_one(seen),
        region_names=REGIONS,
        paradigm="global",
        targets=cast(Any, targets),
    )
    assert len(seen) == 1
    assert seen[0][1] == targets
    assert seen[0][2] is None and seen[0][3] is None


def test_run_grouped_matches_legacy_scatter_loop() -> None:
    """Explicitly reproduce `all_fold_preds[orig_idx] = group_preds[group_idx]`."""
    targets = [f"t:{r}" for r in REGIONS]
    groups = partition(REGIONS, "activity", ACTIVITY)

    legacy: list[Any] = [None] * len(REGIONS)
    for group in groups:
        group_preds = _fake_preds([targets[i] for i in group.indices])
        for group_idx, orig_idx in enumerate(group.indices):
            legacy[orig_idx] = group_preds[group_idx]

    new = run_grouped(
        lambda g, t, p, f: {"y_pred": _fake_preds([str(x) for x in t])},
        region_names=REGIONS,
        paradigm="activity",
        targets=cast(Any, targets),
        activity_by_region=ACTIVITY,
    )
    assert new["y_pred"] == legacy
