"""Feature-ablation feature spaces: ``sensitivity.feature_space`` (plan 2026-10-01).

Pure functions behind the opt-in feature-ablation runs
(``docs/feature_ablation/README.md``). Nothing here is thesis or publication
behaviour: with ``sensitivity.feature_space`` unset no function of this module
is ever called.

Groups
------
The nine groups follow the grouped importance figure
(:data:`strikecast.evaluation.importance.CATEGORY_RULES`, matched on the base
variable name, first match wins). The CORE -- the target's own 7 lags and the
two static covariates (region, activity tier) -- is always on and belongs to no
group. Every other input does:

* seven PAST groups, the ``(feature, lag)`` pairs of the selection pool
  (every past-covariate component at ``common_kwargs.lags_past_covariates``);
* ``weather`` -- the weather / geomagnetic FUTURE covariates;
* ``calendar`` -- the holiday FUTURE covariates plus darts' cyclic calendar
  encoders (``darts_enc_fc_cyc_*``; "Other" in the importance figure, folded in
  here because they are calendar features). The encoders are not series
  components, so they are switched by :attr:`FeatureSpaceConfig.calendar_encoders`
  rather than by a component list.

A pool component that falls in "Other" (or a future component outside
``weather``/``calendar``) is an error, so the groups always partition the pool.

Modes (:class:`strikecast.config.schema.FeatureSpaceConfig`)
-----------------------------------------------------------
``selected``        the cached selection, unchanged (the control);
``all``             core + every group: every pool pair, every future input;
``random``          ``k`` pool pairs drawn without replacement, uniformly or
                    stratified to the selected set's per-group counts; future
                    inputs as in publication, so only the ranking changes;
``groups``          core + the listed groups, every pair of a listed past group;
``selected_minus``  the selected pairs minus the listed groups' (holes stay);
``reselect_only``   the selector re-run on the listed groups' pool only;
``reselect_drop``   the selector re-run on the pool minus the listed groups.

The two ``reselect_*`` modes are a different SELECTION (the data stage hashes
them into the feature-selection identity and lets ``select_top_k`` rank the
restricted pool, see :func:`restricted_components`); every other mode keeps the
publication selection's hash, so its tuned parameters still resolve.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "FUTURE_GROUPS",
    "GROUP_CATEGORY",
    "GROUP_ORDER",
    "PAST_GROUPS",
    "PastLags",
    "canonical_groups",
    "describe",
    "group_of",
    "kept_groups",
    "pairs_by_group",
    "pool_pairs",
    "random_pairs",
    "resolve_space",
    "restricted_components",
]

#: ``((component, (lag, ...)), ...)`` -- :data:`strikecast.models.spec.PastLags`.
PastLags = tuple[tuple[str, tuple[int, ...]], ...]

#: The nine groups, in display order; slug -> importance-figure category.
GROUP_CATEGORY: dict[str, str] = {
    "strikes": "Autoregressive strikes",
    "spatial": "Spatial / static",
    "conflict": "Conflict & damage",
    "comms": "Comms / diplo / aid",
    "macro": "Macroeconomic",
    "missile": "Missile / launch",
    "cyber": "Cyber",
    "weather": "Weather / geomag.",
    "calendar": "Calendar",
}
GROUP_ORDER: tuple[str, ...] = tuple(GROUP_CATEGORY)
FUTURE_GROUPS: tuple[str, ...] = ("weather", "calendar")
PAST_GROUPS: tuple[str, ...] = tuple(g for g in GROUP_ORDER if g not in FUTURE_GROUPS)
_SLUG_OF_CATEGORY = {category: slug for slug, category in GROUP_CATEGORY.items()}

#: Modes whose result keeps every future input (weather AND calendar).
_ALL_FUTURE_MODES = ("selected", "all", "random")
#: Modes whose ``groups`` lists what is KEPT (else: what is dropped).
_KEEP_MODES = ("groups", "reselect_only")
_DROP_MODES = ("selected_minus", "reselect_drop")


def canonical_groups(groups: Iterable[str]) -> tuple[str, ...]:
    """Validated, de-duplicated slugs in :data:`GROUP_ORDER` (a stable identity)."""
    given = [str(g) for g in groups]
    unknown = sorted(set(given) - set(GROUP_ORDER))
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown}; known: {list(GROUP_ORDER)}")
    return tuple(g for g in GROUP_ORDER if g in set(given))


def group_of(component: str, *, future: bool = False) -> str:
    """The group slug of one covariate component (base name, no lag suffix).

    Raises on a component the importance rules put in "Other", and on a future
    component outside ``weather``/``calendar``: the groups must partition every
    input, or a "groups" run would silently drop columns.
    """
    from strikecast.evaluation.importance import classify_feature  # noqa: PLC0415

    category = classify_feature(component)
    slug = _SLUG_OF_CATEGORY.get(category)
    if slug is None:
        raise ValueError(
            f"covariate {component!r} classifies as {category!r}, which is no feature group; "
            "extend evaluation.importance.CATEGORY_RULES before ablating it"
        )
    if future and slug not in FUTURE_GROUPS:
        raise ValueError(
            f"future covariate {component!r} classifies as {category!r}; only "
            f"{list(FUTURE_GROUPS)} may hold future covariates"
        )
    if not future and slug in FUTURE_GROUPS:
        raise ValueError(
            f"past covariate {component!r} classifies as {category!r}; past covariates must "
            f"fall in one of {list(PAST_GROUPS)}"
        )
    return slug


def kept_groups(mode: str, groups: Sequence[str] = ()) -> tuple[str, ...]:
    """The groups a mode's feature space contains (core aside), in display order.

    ``selected``/``all``/``random`` keep every group (the selection, or the
    random draw, decides which past pairs); ``groups``/``reselect_only`` keep
    the listed ones; ``selected_minus``/``reselect_drop`` keep the rest.
    """
    listed = set(canonical_groups(groups))
    if mode in _ALL_FUTURE_MODES:
        return GROUP_ORDER
    if mode in _KEEP_MODES:
        return tuple(g for g in GROUP_ORDER if g in listed)
    if mode in _DROP_MODES:
        return tuple(g for g in GROUP_ORDER if g not in listed)
    raise ValueError(f"unknown feature-space mode {mode!r}")


def pool_pairs(past_components: Sequence[str], lags: Iterable[int]) -> list[tuple[str, int]]:
    """Every ``(component, lag)`` pair of the selection pool, in canonical order:
    components in series order, lags ascending."""
    grid = sorted(int(lag) for lag in lags)
    return [(str(c), lag) for c in past_components for lag in grid]


def _group_pairs(pairs: Iterable[tuple[str, int]]) -> PastLags:
    """Pairs -> :data:`PastLags`: components in order of first appearance, lags
    ascending (the canonical form ``feature_selection._group_past_lags`` uses)."""
    grouped: dict[str, list[int]] = {}
    for component, lag in pairs:
        grouped.setdefault(component, []).append(int(lag))
    return tuple((c, tuple(sorted(lags))) for c, lags in grouped.items())


def _flatten(past_lags: PastLags | None) -> list[tuple[str, int]]:
    return [(c, int(lag)) for c, lags in (past_lags or ()) for lag in lags]


def pairs_by_group(past_lags: PastLags | None) -> dict[str, int]:
    """``{slug: number of pairs}`` of a past-lag set, every past group present."""
    counts = Counter(group_of(c) for c, _ in _flatten(past_lags))
    return {g: int(counts.get(g, 0)) for g in PAST_GROUPS}


def random_pairs(
    pool: Sequence[tuple[str, int]],
    k: int,
    draw: int,
    strata: Mapping[str, int] | None = None,
) -> list[tuple[str, int]]:
    """``k`` pool pairs drawn without replacement with ``default_rng(draw)``.

    ``strata`` (``{group: count}``, summing to ``k``) draws that many pairs
    uniformly WITHIN each group instead, groups in :data:`GROUP_ORDER`, one
    generator for the whole draw. The result is in pool order either way, so it
    is a pure function of ``(pool, k, draw, strata)``.
    """
    if k <= 0 or k > len(pool):
        raise ValueError(f"cannot draw k={k} pairs from a pool of {len(pool)}")
    rng = np.random.default_rng(int(draw))
    if strata is None:
        picked = sorted(rng.choice(len(pool), size=int(k), replace=False).tolist())
        return [pool[i] for i in picked]
    if sum(int(n) for n in strata.values()) != k:
        raise ValueError(f"strata {dict(strata)} do not sum to k={k}")
    by_group: dict[str, list[int]] = {}
    for i, (component, _) in enumerate(pool):
        by_group.setdefault(group_of(component), []).append(i)
    chosen: list[int] = []
    for g in GROUP_ORDER:
        n = int(strata.get(g, 0))
        if n == 0:
            continue
        members = by_group.get(g, [])
        if n > len(members):
            raise ValueError(f"stratum {g!r} wants {n} pairs but the pool has {len(members)}")
        chosen.extend(members[j] for j in rng.choice(len(members), size=n, replace=False))
    return [pool[i] for i in sorted(chosen)]


def restricted_components(
    mode: str,
    groups: Sequence[str],
    past_components: Sequence[str],
    future_components: Sequence[str],
) -> tuple[list[str], list[str]]:
    """``(past, future)`` components a feature space may draw from, in series order.

    For the ``reselect_*`` modes this IS the selector's pool: the data stage
    subsets the bundle to it before ``select_top_k``. For the others it is the
    union the mode chooses pairs from.
    """
    keep = set(kept_groups(mode, groups))
    past = [c for c in past_components if group_of(c) in keep]
    future = [c for c in future_components if group_of(c, future=True) in keep]
    return past, future


def resolve_space(
    mode: str,
    *,
    groups: Sequence[str] = (),
    k: int = 100,
    draw: int = 0,
    stratified: bool = False,
    selected_past_lags: PastLags | None,
    past_components: Sequence[str],
    future_components: Sequence[str],
    lag_grid: Iterable[int],
) -> tuple[PastLags, list[str]]:
    """``(past_lags, future_keep)`` of a non-``reselect`` mode.

    ``selected_past_lags`` is the cached selection (needed by ``selected``,
    ``selected_minus`` and the stratified ``random``); ``past_components`` and
    ``future_components`` are the PRE-selection series' components; ``lag_grid``
    is ``common_kwargs.lags_past_covariates``. ``past_lags == ()`` means "no
    past covariates at all"; an empty ``future_keep`` means "no future
    covariate components" (the encoders are switched separately).
    """
    if mode.startswith("reselect"):
        raise ValueError(f"{mode!r} is a re-selection; the data stage runs the selector for it")
    past_pool, future_keep = restricted_components(mode, groups, past_components, future_components)
    if mode == "selected":
        if selected_past_lags is None:
            raise ValueError("mode 'selected' needs a figure-protocol selection (past_lags)")
        return tuple(selected_past_lags), list(future_components)
    if mode in ("all", "groups"):
        return _group_pairs(pool_pairs(past_pool, lag_grid)), future_keep
    if mode == "random":
        pool = pool_pairs(past_components, lag_grid)
        strata = pairs_by_group(selected_past_lags) if stratified else None
        if stratified and selected_past_lags is None:
            raise ValueError("a stratified draw needs the selected set's per-group counts")
        return _group_pairs(random_pairs(pool, k, draw, strata)), future_keep
    if mode == "selected_minus":
        if selected_past_lags is None:
            raise ValueError("mode 'selected_minus' needs a figure-protocol selection (past_lags)")
        keep = set(kept_groups(mode, groups))
        pairs = [(c, lag) for c, lag in _flatten(selected_past_lags) if group_of(c) in keep]
        return _group_pairs(pairs), future_keep
    raise ValueError(f"unknown feature-space mode {mode!r}")


def describe(
    past_lags: PastLags | None,
    future_keep: Sequence[str],
    **extra: Any,
) -> dict[str, Any]:
    """The ``feature_space.json`` payload (counts first, then the full space)."""
    return {
        **extra,
        "n_past_pairs": len(_flatten(past_lags)),
        "pairs_by_group": pairs_by_group(past_lags),
        "past_lags": [[c, list(lags)] for c, lags in (past_lags or ())],
        "future_keep": list(future_keep),
    }
