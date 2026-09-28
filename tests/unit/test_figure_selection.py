"""The figure protocol of the feature selection (plan 2026-09-28).

Five selectors (``diff_l2``, ``count_poisson``, ``count_tweedie``,
``hurdle_binary``, ``hurdle_tweedie_pos``) on one shared skeleton, each under
its branch's matched objective, target and weights; only the ``_pastcov`` lag
columns compete; exactly ``top_k`` ``(feature, lag)`` pairs survive, ties by
name; every future covariate is kept; the selection persists as JSON schema 2
with ``past_lags`` and is refused from cache without it; the count family
routes its GBDTs to the selection of their objective.

Synthetic data only: ``select_top_k`` runs on hand-built darts series with a
planted signal column, the data-stage tests on a small panel through
``build_bundle``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")
pytest.importorskip("lightgbm")

from darts import TimeSeries  # noqa: E402

from strikecast.config.loader import load_experiment  # noqa: E402
from strikecast.config.schema import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    FeatureSelectionStageConfig,
    ModelEntry,
    SeriesConfig,
    StoreConfig,
    TrackingConfig,
    TransformConfig,
    WindowTransformConfig,
)
from strikecast.data import feature_selection as fsel  # noqa: E402
from strikecast.data.feature_selection import (  # noqa: E402
    FIGURE_SELECTORS,
    LEGACY_SELECTORS,
    FeatureSelection,
    parse_pastcov_lag,
    select_top_k,
    shared_selector_kwargs,
)
from strikecast.data.series import build_bundle, model_space_parts  # noqa: E402
from strikecast.pipeline import data_stage  # noqa: E402
from strikecast.pipeline.data_stage import FeatureCacheMismatch  # noqa: E402
from strikecast.store import RunStore  # noqa: E402
from strikecast.transforms.diff import Diff  # noqa: E402

N_DAYS = 220
N_NOISE = 40  # 41 past components x 3 lags = 123 pool columns > 100
FUTURE = ["w_temp", "w_rain", "holiday"]
SIGNAL = "zz_signal"  # sorts LAST by name, so ranking first is the gain's doing


# --------------------------------------------------------------------------- #
# hand-built series with a planted signal
# --------------------------------------------------------------------------- #
def _signal(rng: np.random.Generator, n: int) -> np.ndarray:
    """A slow, blocky driver: persistent enough that its lag -1 explains y."""
    blocks = rng.normal(size=n // 14 + 1).repeat(14)[:n]
    return blocks + 0.1 * rng.normal(size=n)


def _series(kind: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=N_DAYS, freq="D")
    fidx = pd.date_range("2022-01-01", periods=N_DAYS + 10, freq="D")
    targets, pasts, futures = [], [], []
    noise_names = [f"noise_{i:02d}" for i in range(N_NOISE)]
    # one noise component whose own name contains `_target` (Q2)
    noise_names[0] = "act_target_mix"
    for region in range(3):
        sig = _signal(rng, N_DAYS)
        rate = np.exp(0.3 + 1.2 * sig)
        counts = rng.poisson(rate).astype(float)
        y = (counts > 1).astype(float) if kind == "binary" else counts
        target = TimeSeries.from_times_and_values(idx, y, columns=["y"])
        targets.append(
            target.with_static_covariates(pd.DataFrame({"region_code": [float(region)]}))
        )
        past = np.column_stack([rng.normal(size=(N_DAYS, N_NOISE)), sig])
        pasts.append(
            TimeSeries.from_times_and_values(idx, past, columns=[*noise_names, SIGNAL])
        )
        futures.append(
            TimeSeries.from_times_and_values(
                fidx, rng.normal(size=(len(fidx), len(FUTURE))), columns=FUTURE
            )
        )
    return targets, pasts, futures


def _positive_weights(targets):
    return [t.map(lambda v: (v > 0).astype(float)) for t in targets]


def _figure_selection(selector: str, *, seed: int = 0, top_k: int = 100):
    kind = "binary" if selector == "hurdle_binary" else "count"
    targets, pasts, futures = _series(kind, seed)
    config = FeatureSelectionStageConfig(
        selector=selector, device="cpu", num_threads=2, top_k=top_k, deterministic=True
    ).build()
    weights = _positive_weights(targets) if selector == "hurdle_tweedie_pos" else None
    return select_top_k(
        targets, pasts, futures, config, sample_weight=weights, protocol="figure"
    ), (targets, pasts, futures)


@pytest.fixture(autouse=True)
def _hash_seed_zero(monkeypatch) -> None:
    """Deterministic selections refuse to compute without it (A12/D2)."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")


@pytest.fixture(scope="module")
def selections() -> dict[str, FeatureSelection]:
    return {selector: _figure_selection(selector)[0] for selector in FIGURE_SELECTORS}


# --------------------------------------------------------------------------- #
# the five builders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("selector", "objective", "kind", "power", "unbalance"),
    [
        ("diff_l2", "regression", "regressor", None, None),
        ("count_poisson", "poisson", "regressor", None, None),
        ("count_tweedie", "tweedie", "regressor", 1.5, None),
        ("hurdle_binary", "binary", "classifier", None, True),
        ("hurdle_tweedie_pos", "tweedie", "regressor", 1.5, None),
    ],
)
def test_each_figure_selector_has_its_matched_objective(
    selector, objective, kind, power, unbalance
) -> None:
    stage = FeatureSelectionStageConfig(selector=selector, device="cpu", num_threads=3)
    assert stage.protocol == "figure"
    cfg = stage.build()
    assert (cfg.objective, cfg.kind, cfg.top_k) == (objective, kind, 100)
    kwargs = cfg.model_kwargs
    assert "objective" not in kwargs  # the field is the objective, not a kwarg override
    assert kwargs.get("tweedie_variance_power") == power
    assert kwargs.get("is_unbalance") == unbalance
    assert (kwargs["device_type"], kwargs["num_threads"]) == ("cpu", 3)
    # ONE shared skeleton: everything else is shared_selector_kwargs() verbatim
    shared = shared_selector_kwargs()
    assert {k: kwargs[k] for k in shared} == shared
    assert set(kwargs) - set(shared) <= {
        "device_type", "num_threads", "tweedie_variance_power", "is_unbalance"
    }
    # the objective reaches LightGBM
    model = cfg.build_model()
    assert model.kwargs["objective"] == objective


def test_the_shared_skeleton_is_the_count_diff_one() -> None:
    shared = shared_selector_kwargs()
    diffreg = fsel.diffreg_config().model_kwargs
    for key, value in shared.items():
        assert diffreg[key] == value, key
    assert shared["lags_past_covariates"] == [-1, -7, -14]
    assert shared["n_estimators"] == 500 and shared["random_state"] == 42


def test_protocol_follows_the_selector_name() -> None:
    for name in LEGACY_SELECTORS:
        assert FeatureSelectionStageConfig(selector=name).protocol == "legacy"
    for name in FIGURE_SELECTORS:
        assert FeatureSelectionStageConfig(selector=name).protocol == "figure"


def test_only_the_positive_only_selectors_get_weights(tmp_path) -> None:
    bundle = _bundle(transform="identity")
    for selector in (*FIGURE_SELECTORS, *LEGACY_SELECTORS):
        cfg = _stage_cfg(tmp_path, selector)
        weights = data_stage._selection_sample_weight(cfg, bundle)
        if selector in ("hurdle_tweedie_pos", "zipoisson_regressor"):
            assert weights is not None and len(weights) == len(bundle.target_train)
            for w, ts in zip(weights, bundle.target_train, strict=True):
                assert w.time_index.equals(ts.time_index)
                assert set(np.unique(w.values())) <= {0.0, 1.0}
        else:
            assert weights is None, selector


# --------------------------------------------------------------------------- #
# ranking: pool, exact top-k, determinism, signal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selector", FIGURE_SELECTORS)
def test_the_pool_is_past_covariate_lags_only_and_exactly_top_k(selections, selector) -> None:
    sel = selections[selector]
    assert sel.protocol == "figure"
    assert len(sel.top_features) == 100
    assert all(parse_pastcov_lag(n) is not None for n in sel.top_features)
    assert not any(
        tag in n for n in sel.top_features for tag in ("_target_lag", "_futcov", "_statcov")
    )
    # the ranking table IS the pool: nothing else competed
    assert all(parse_pastcov_lag(n) is not None for n in sel.gain_table["Feature"])
    assert len(sel.gain_table) == (N_NOISE + 1) * 3
    # past_lags holds exactly the 100 pairs, components in rank order
    pairs = {(c, lag) for c, lags in sel.past_lags for lag in lags}
    assert pairs == {parse_pastcov_lag(n) for n in sel.top_features}
    assert sum(len(lags) for _, lags in sel.past_lags) == 100
    firsts = list(dict.fromkeys(parse_pastcov_lag(n)[0] for n in sel.top_features))
    assert sel.past_keep == firsts == [c for c, _ in sel.past_lags]
    assert all(isinstance(lag, int) for _, lags in sel.past_lags for lag in lags)
    assert sel.n_nonzero_gain == int((sel.gain_table["mean_gain"] > 0).sum())


@pytest.mark.parametrize("selector", FIGURE_SELECTORS)
def test_the_planted_signal_ranks_first(selections, selector) -> None:
    sel = selections[selector]
    assert parse_pastcov_lag(sel.top_features[0])[0] == SIGNAL
    assert sel.past_keep[0] == SIGNAL


def test_the_cut_is_a_stable_sort_by_gain_then_name(selections) -> None:
    table = selections["count_tweedie"].gain_table
    expected = sorted(
        zip(table["mean_gain"], table["Feature"], strict=True), key=lambda r: (-r[0], r[1])
    )
    assert list(table["Feature"]) == [name for _, name in expected]


def test_the_figure_selection_is_deterministic() -> None:
    first, _ = _figure_selection("count_poisson", seed=1)
    again, _ = _figure_selection("count_poisson", seed=1)
    assert first.top_features == again.top_features
    assert first.past_lags == again.past_lags
    pd.testing.assert_frame_equal(first.gain_table, again.gain_table)


def test_future_keep_is_everything_under_figure_and_filtered_under_legacy() -> None:
    targets, pasts, futures = _series("count", 2)
    stage = FeatureSelectionStageConfig(
        selector="countreg", device="cpu", num_threads=2, top_k=5, deterministic=True
    )
    legacy = select_top_k(targets, pasts, futures, stage.build())
    assert legacy.protocol == "legacy" and legacy.past_lags is None
    assert set(legacy.future_keep) < set(FUTURE)  # filtered (5 slots, signal/target win)
    figure, _ = _figure_selection("count_tweedie", seed=2, top_k=5)
    assert figure.future_keep == FUTURE  # all, original order
    assert len(figure.top_features) == 5


# --------------------------------------------------------------------------- #
# JSON schema 2 and the ranking CSV
# --------------------------------------------------------------------------- #
def test_schema_2_round_trips_past_lags(selections, tmp_path) -> None:
    sel = selections["hurdle_tweedie_pos"]
    path = sel.to_json(tmp_path / "sel.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == 2 and payload["protocol"] == "figure"
    assert payload["past_lags"][0][0] == SIGNAL
    back = FeatureSelection.from_json(path)
    assert back.past_lags == sel.past_lags
    assert back.protocol == "figure"
    assert back.n_nonzero_gain == sel.n_nonzero_gain
    assert back.past_keep == sel.past_keep and back.future_keep == sel.future_keep
    assert back.config == sel.config

    csv = sel.write_ranking_csv(tmp_path / "sel.ranking.csv")
    frame = pd.read_csv(csv)
    assert list(frame.columns) == [
        "Feature", *[f"h{h}_gain" for h in range(1, 8)], "mean_gain", "selected"
    ]
    assert int(frame["selected"].sum()) == 100
    assert frame["selected"].head(100).all()


def test_a_legacy_payload_reads_back_without_past_lags(tmp_path) -> None:
    targets, pasts, futures = _series("count", 3)
    stage = FeatureSelectionStageConfig(selector="diffreg", device="cpu", top_k=10)
    sel = select_top_k(targets, pasts, futures, stage.build())
    payload = json.loads(sel.to_json(tmp_path / "legacy.json").read_text(encoding="utf-8"))
    assert payload["past_lags"] is None and payload["protocol"] == "legacy"
    # a schema-1 payload (no schema/protocol/past_lags keys) is a legacy selection
    for key in ("schema", "protocol", "past_lags", "n_nonzero_gain"):
        payload.pop(key)
    (tmp_path / "v1.json").write_text(json.dumps(payload), encoding="utf-8")
    back = FeatureSelection.from_json(tmp_path / "v1.json")
    assert back.protocol == "legacy" and back.past_lags is None
    assert back.past_keep == sel.past_keep


# --------------------------------------------------------------------------- #
# the data stage: differenced target, cache, featsel groups
# --------------------------------------------------------------------------- #
TARGET = "act_drone_strike_on_ua"
REGIONS = ("alpha", "beta", "gamma")
ACTIVITY = {"alpha": 1, "beta": 2, "gamma": 3}
PAST = ["past_a", "past_b", "past_c"]
PANEL_FUTURE = ["fut_x", "fut_y"]


def _panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=140, freq="D")
    frames = []
    for i, region in enumerate(REGIONS):
        driver = rng.poisson(1.0 + i, len(dates)).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "event_date": dates,
                    "Activity_Level": ACTIVITY[region],
                    TARGET: rng.poisson(0.5 + 0.3 * driver).astype(float),
                    "past_a": np.roll(driver, 1),
                    "past_b": rng.normal(size=len(dates)),
                    "past_c": np.arange(len(dates), dtype=float) % 7,
                    "fut_x": (dates.dayofweek >= 5).astype(float),
                    "fut_y": rng.normal(size=len(dates)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _window() -> SeriesConfig:
    return SeriesConfig(window=WindowTransformConfig(expdecay="leaky"))


def _bundle(transform: str = "identity"):  # noqa: ARG001 - the bundle is transform-free
    return build_bundle(_panel(), TARGET, PAST, PANEL_FUTURE, _window(), ACTIVITY)


def _fs(selector: str, top_k: int = 12) -> FeatureSelectionStageConfig:
    return FeatureSelectionStageConfig(
        selector=selector, device="cpu", num_threads=2, top_k=top_k, cache=True,
        deterministic=True, require_cached=True,
    )


def _stage_cfg(tmp_path: Path, selector: str, transform: str = "identity", **extra):
    return ExperimentConfig(
        name="diff" if transform == "diff" else "count",
        data=DataConfig(target=TARGET),
        series=_window(),
        transform=TransformConfig(kind=transform),  # type: ignore[arg-type]
        feature_selection=_fs(selector),
        tracking=TrackingConfig(backend="noop"),
        store=StoreConfig(root=str(tmp_path / "runs")),
        **extra,
    )


def _spy_select(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    real = fsel.select_top_k

    def spy(train_target, past, future, config, **kw):
        calls.append({"target": train_target, "config": config, **kw})
        return real(train_target, past, future, config, **kw)

    monkeypatch.setattr(fsel, "select_top_k", spy)
    return calls


def test_the_diff_selector_is_fitted_on_the_differenced_target(tmp_path, monkeypatch) -> None:
    bundle = _bundle()
    calls = _spy_select(monkeypatch)
    cfg = _stage_cfg(tmp_path, "diff_l2", "diff")
    data_stage.build_or_load_features(cfg, RunStore(cfg.store.root), bundle, compute=True)
    (call,) = calls
    assert call["protocol"] == "figure"
    expected = model_space_parts(bundle)["target_train"]
    for got, want, raw in zip(call["target"], expected, bundle.raw.target, strict=True):
        np.testing.assert_array_equal(got.values(), want.values())
        # ... which is Diff().forward of the full level series, on the train part
        diffed = Diff().forward([raw])[0]
        np.testing.assert_array_equal(got.values(), diffed.values()[: len(got)])
        assert got.start_time() == raw.time_index[1]  # one step lost to differencing

    # the legacy diffreg keeps the LEVEL target (Q8, reproduced)
    calls.clear()
    legacy = _stage_cfg(tmp_path / "legacy", "diffreg", "diff")
    data_stage.build_or_load_features(legacy, RunStore(legacy.store.root), bundle, compute=True)
    (call,) = calls
    assert call["protocol"] == "legacy"
    assert call["target"] is bundle.target_train


def test_a_figure_selection_caches_past_lags_and_refuses_a_cache_without(tmp_path) -> None:
    bundle = _bundle()
    cfg = _stage_cfg(tmp_path, "count_poisson")
    store = RunStore(cfg.store.root)
    made = data_stage.build_or_load_features(cfg, store, bundle, compute=True)
    assert made.source == "computed" and made.past_lags is not None
    assert sum(len(lags) for _, lags in made.past_lags) == 12
    assert made.past_keep == [c for c, _ in made.past_lags]
    assert made.future_keep == list(bundle.future_covs[0].components)  # all of them

    csv = data_stage.ranking_csv_path(made.path)
    assert csv.name == made.path.name.replace(".json", ".ranking.csv") and csv.is_file()
    provenance = json.loads(made.path.read_text(encoding="utf-8"))["provenance"]
    assert provenance["protocol"] == "figure"

    hit = data_stage.build_or_load_features(cfg, store, bundle)
    assert hit.source == "store"
    assert (hit.past_keep, hit.future_keep, hit.past_lags) == (
        made.past_keep, made.future_keep, made.past_lags
    )

    payload = json.loads(made.path.read_text(encoding="utf-8"))
    payload.pop("past_lags")
    made.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FeatureCacheMismatch, match="past_lags"):
        data_stage.build_or_load_features(cfg, store, bundle)


def test_featsel_caches_every_distinct_selection_and_prepare_routes(
    tmp_path, monkeypatch
) -> None:
    cfg = _stage_cfg(
        tmp_path,
        "count_tweedie",
        feature_selections={"poisson": _fs("count_poisson"), "tweedie": _fs("count_tweedie")},
        models=[
            ModelEntry(name="lightgbm_poisson", selection="poisson"),
            ModelEntry(name="lightgbm_tweedie", selection="tweedie"),
            ModelEntry(name="lstm_w7"),
        ],
    )
    assert cfg.selection_groups() == {
        "poisson": ["lightgbm_poisson"],
        None: ["lightgbm_tweedie", "lstm_w7"],
    }
    assert data_stage.is_composite_family(cfg) is False

    panel = _panel()
    monkeypatch.setattr(
        data_stage,
        "build_or_load_panel",
        lambda head_cfg, store=None, **_: (panel, [], data_stage._panel_hash(head_cfg)),
    )
    import strikecast.data.covariates as covariates  # noqa: PLC0415

    monkeypatch.setattr(
        covariates,
        "split_covariates",
        lambda panel, weather, target: SimpleNamespace(
            past_covariates=list(PAST), future_covariates=list(PANEL_FUTURE)
        ),
    )
    inputs = SimpleNamespace(activity_by_region=dict(ACTIVITY))
    store = RunStore(cfg.store.root)

    made = data_stage.select_features(cfg, store, inputs=inputs)  # type: ignore[arg-type]
    assert set(made) == {"", "poisson"}
    assert made[""].hash != made["poisson"].hash
    selectors = {
        key: json.loads(s.path.read_text(encoding="utf-8"))["config"]["objective"]  # type: ignore[union-attr]
        for key, s in made.items()
    }
    assert selectors == {"": "tweedie", "poisson": "poisson"}

    prepared = data_stage.prepare_by_selection(cfg, store, inputs=inputs)
    assert list(prepared) == ["poisson", None]
    for key, (sub, art, names) in prepared.items():
        assert sub.feature_selection.selector == ("count_poisson" if key else "count_tweedie")
        assert art.features.source == "store"
        assert art.features.hash == made[key or ""].hash
        assert art.features.past_lags == made[key or ""].past_lags
        assert list(art.bundle.future_covs[0].components) == PANEL_FUTURE
        assert names == cfg.selection_groups()[key]
    only = data_stage.prepare_by_selection(cfg, store, models=["lstm_w7"], inputs=inputs)
    assert list(only) == [None]


# --------------------------------------------------------------------------- #
# schema: routing keys, for_selection, configs
# --------------------------------------------------------------------------- #
def test_a_model_selection_must_name_a_feature_selections_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="not a feature_selections key"):
        _stage_cfg(tmp_path, "count_tweedie", models=[ModelEntry(name="m", selection="nope")])
    cfg = _stage_cfg(tmp_path, "count_tweedie")
    assert cfg.for_selection(None) is cfg
    with pytest.raises(KeyError, match="nope"):
        cfg.for_selection("nope")


def test_selection_groups_follow_the_requested_models(tmp_path) -> None:
    cfg = _stage_cfg(
        tmp_path,
        "count_tweedie",
        feature_selections={"poisson": _fs("count_poisson"), "other": _fs("count_poisson")},
        models=[
            ModelEntry(name="a"),
            ModelEntry(name="b", selection="poisson"),
            ModelEntry(name="c", selection="other"),  # equal config -> same group
        ],
    )
    assert cfg.selection_groups() == {None: ["a"], "poisson": ["b", "c"]}
    assert cfg.selection_groups(["c", "a"]) == {"other": ["c"], None: ["a"]}
    narrowed = cfg.for_selection("poisson")
    assert narrowed.feature_selection.selector == "count_poisson"
    assert narrowed.models == cfg.models and narrowed.name == cfg.name


@pytest.mark.parametrize("name", ["count", "diff", "hurdle", "damage", "chronos2"])
@pytest.mark.parametrize("legacy", [False, True])
def test_every_experiment_and_legacy_overlay_validates(name: str, legacy: bool) -> None:
    cfg = load_experiment(name, [f"legacy={name}"] if legacy else [])
    for key in (None, *cfg.feature_selections):
        fs = cfg.feature_selection_for(key)
        if name == "chronos2":
            continue
        want = "legacy" if (legacy or name == "damage") else "figure"
        assert fs.protocol == want, (name, key)
        fs.build()  # every configured selector has a builder
    for key in cfg.selection_groups():
        cfg.for_selection(key)
