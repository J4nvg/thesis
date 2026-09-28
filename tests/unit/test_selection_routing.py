"""Per-model selection routing and the tuning-staleness guard.

Plan "figure feature selection" (2026-09-28) §5 and §7, with its amendments:

* ``run_experiment`` / ``tune_experiment`` prepare ONE data stage per
  selection group (``cfg.selection_groups``) under the narrowed config
  (``cfg.for_selection(key)``); an explicit ``data=`` is accepted only when all
  requested models share one group;
* ``make_run_context(..., data)`` carries ``features.past_lags`` (and, for a
  composite, every head's) into the builder's context;
* a study tuned on another selection is never resumed or reused
  (:class:`TunedParamsStale`); ``--force`` archives it, never deletes it;
  provenance-less (imported thesis) params are valid only under ``legacy``.

Same synthetic 4-region panel and ``diff`` experiment as
``test_pipeline_stages``; no real data, no GPU.
"""
# The fixtures are imported from test_pipeline_stages and requested by name.
# ruff: noqa: F811

from __future__ import annotations

import json
from dataclasses import replace

import pytest

pytest.importorskip("darts")

from test_pipeline_stages import (  # noqa: E402
    _tunable_linear_spec,
    bundle,  # noqa: F401  (fixture)
    cfg,  # noqa: F401  (fixture)
    data,  # noqa: F401  (fixture)
    panel,  # noqa: F401  (fixture)
    store,  # noqa: F401  (fixture)
)

from strikecast.config.schema import FeatureSelectionStageConfig, ModelEntry  # noqa: E402
from strikecast.pipeline import data_stage, run_stage, tune_stage  # noqa: E402
from strikecast.pipeline.context import get_spec, make_run_context  # noqa: E402
from strikecast.pipeline.data_stage import CompositeData  # noqa: E402
from strikecast.pipeline.run_stage import TunedParamsStale  # noqa: E402
from strikecast.store import RunKey  # noqa: E402

LAGS_A = (("past_a", (-1, -7)),)
LAGS_B = (("past_b", (-14,)),)


def _routed(cfg):
    """``naive_last`` on the top-level selection, ``naive_weekly`` on ``alt``."""
    return cfg.model_copy(
        update={
            "feature_selections": {"alt": FeatureSelectionStageConfig(selector="diffreg")},
            "models": [
                ModelEntry(name="linear"),
                ModelEntry(name="naive_last"),
                ModelEntry(name="naive_weekly", selection="alt"),
            ],
        }
    )


def _figure(cfg):
    return cfg.model_copy(
        update={"feature_selection": FeatureSelectionStageConfig(selector="diff_l2")}
    )


def _write_best(store, cfg, model, provenance=None):
    directory = store.tuning_dir(cfg.name, model)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"variant": model, "best_params": {}}
    if provenance is not None:
        payload["provenance"] = provenance
    (directory / "best_params.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def test_run_experiment_prepares_one_data_stage_per_selection_group(
    cfg, data, store, monkeypatch
) -> None:
    routed = _routed(cfg)
    prepared: list[str] = []

    def _prepare(group_cfg, store_, **kw):
        prepared.append(group_cfg.feature_selection.selector)
        return data

    seen: list[tuple[str, str]] = []
    real = run_stage.run_stage

    def _spy(cfg_, model, paradigm, seed, stage, data_, **kw):
        seen.append((model, cfg_.feature_selection.selector))
        return real(cfg_, model, paradigm, seed, stage, data_, **kw)

    monkeypatch.setattr(data_stage, "prepare_data", _prepare)
    monkeypatch.setattr(run_stage, "run_stage", _spy)
    run_stage.run_experiment(
        routed,
        models=["naive_last", "naive_weekly"],
        paradigms=["global"],
        seeds=[42],
        stages=("test",),
        store=store,
    )
    assert prepared == ["countreg", "diffreg"]
    # each model runs under ITS narrowed config
    assert seen == [("naive_last", "countreg"), ("naive_weekly", "diffreg")]


def test_tune_experiment_routes_like_run_experiment(cfg, data, store, monkeypatch) -> None:
    routed = _routed(cfg)
    prepared: list[str] = []

    def _prepare(group_cfg, store_, **kw):
        prepared.append(group_cfg.feature_selection.selector)
        return data

    monkeypatch.setattr(data_stage, "prepare_data", _prepare)
    outcomes = tune_stage.tune_experiment(
        routed, models=["naive_last", "naive_weekly"], store=store
    )
    assert prepared == ["countreg", "diffreg"]
    assert [o.model for o in outcomes] == ["naive_last", "naive_weekly"]


def test_explicit_data_with_mixed_selections_is_refused(cfg, data, store) -> None:
    routed = _routed(cfg)
    with pytest.raises(ValueError, match="ONE feature selection"):
        run_stage.run_experiment(
            routed, data=data, models=["naive_last", "naive_weekly"], stages=("test",),
            store=store,
        )
    with pytest.raises(ValueError, match="ONE feature selection"):
        tune_stage.tune_experiment(routed, data=data, store=store)


def test_explicit_data_for_one_group_runs_under_the_narrowed_config(
    cfg, data, store, monkeypatch
) -> None:
    routed = _routed(cfg)
    monkeypatch.setattr(
        data_stage, "prepare_data", lambda *a, **k: pytest.fail("data was given")
    )
    seen: list[str] = []
    real = run_stage.run_stage

    def _spy(cfg_, model, paradigm, seed, stage, data_, **kw):
        seen.append(cfg_.feature_selection.selector)
        assert data_ is data
        return real(cfg_, model, paradigm, seed, stage, data_, **kw)

    monkeypatch.setattr(run_stage, "run_stage", _spy)
    run_stage.run_experiment(
        routed, data=data, models=["naive_weekly"], paradigms=["global"], seeds=[42],
        stages=("test",), store=store,
    )
    assert seen == ["diffreg"]


# --------------------------------------------------------------------------- #
# the run context
# --------------------------------------------------------------------------- #
def test_make_run_context_carries_the_selected_lags(cfg, data) -> None:
    plain = make_run_context(cfg, "linear", 7)
    assert plain.past_lags is None and plain.head_past_lags == ()
    # a legacy selection (past_lags=None) leaves the context untouched
    assert make_run_context(cfg, "linear", 7, data) == plain

    figure = replace(data, features=replace(data.features, past_lags=LAGS_A))
    ctx = make_run_context(cfg, "linear", 7, figure)
    assert ctx.past_lags == LAGS_A
    assert ctx.seed == 7 and ctx.head_past_lags == ()


def test_make_run_context_carries_every_heads_lags_for_a_composite(cfg, data) -> None:
    regressor = replace(data, features=replace(data.features, past_lags=LAGS_A))
    classifier = replace(data, features=replace(data.features, past_lags=LAGS_B))
    composite = CompositeData(
        bundle=regressor.bundle,
        features=regressor.features,
        panel_hash=regressor.panel_hash,
        series_hash=regressor.series_hash,
        heads={"classifier": classifier, "regressor": regressor},
        primary="regressor",
        family="hurdle",
    )
    ctx = make_run_context(cfg, "linear", 42, composite)
    assert ctx.head_past_lags == (("classifier", LAGS_B), ("regressor", LAGS_A))
    assert ctx.past_lags == LAGS_A  # the primary head's
    assert ctx.for_head("classifier").past_lags == LAGS_B


# --------------------------------------------------------------------------- #
# resolve_params
# --------------------------------------------------------------------------- #
def test_resolve_params_refuses_a_study_tuned_on_another_selection(cfg, store) -> None:
    _write_best(store, cfg, "linear", {"features_hash": "old"})
    spec = get_spec("linear", "diff")
    with pytest.raises(TunedParamsStale, match="--force"):
        run_stage.resolve_params(cfg, spec, "linear", store, features_hash="new")
    params, source = run_stage.resolve_params(cfg, spec, "linear", store, features_hash="old")
    assert source == "tuned"
    # no features_hash given (Chronos-2): no check
    assert run_stage.resolve_params(cfg, spec, "linear", store)[1] == "tuned"


def test_provenance_less_params_are_legacy_only(cfg, store) -> None:
    _write_best(store, cfg, "linear", {"audit": "2026-09-26 B1"})  # the import script's
    spec = get_spec("linear", "diff")
    assert cfg.feature_selection.protocol == "legacy"
    assert run_stage.resolve_params(cfg, spec, "linear", store, features_hash="x")[1] == "tuned"
    with pytest.raises(TunedParamsStale, match="thesis"):
        run_stage.resolve_params(_figure(cfg), spec, "linear", store, features_hash="x")


def test_run_stage_records_a_stale_study_as_a_failed_stage(cfg, data, store) -> None:
    _write_best(store, cfg, "linear", {"features_hash": "old"})
    with pytest.raises(TunedParamsStale):
        run_stage.run_stage(cfg, "linear", "global", 42, "cv", data, store=store)
    state = store.read_state(RunKey(cfg.name, "linear", "global", 42), "cv")
    assert state.status == "failed" and "TunedParamsStale" in (state.error or "")


# --------------------------------------------------------------------------- #
# tune_model
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    """Raised by the fake `tune` so no study actually runs."""


@pytest.fixture
def tunable(monkeypatch):
    spec = _tunable_linear_spec()
    monkeypatch.setattr(tune_stage, "get_spec", lambda name, exp=None: spec)
    return spec


def _fake_tune(monkeypatch, seen):
    import strikecast.tuning as tuning

    def _tune(spec, run_trial, settings, storage, *, out_dir, **kw):
        seen.append(sorted(p.name for p in out_dir.iterdir()))
        seen.append(json.loads((out_dir / tune_stage.STUDY_PROVENANCE).read_text()))
        raise _Stop

    monkeypatch.setattr(tuning, "tune", _tune)


def test_tune_model_refuses_best_params_of_another_selection(
    cfg, data, store, tunable
) -> None:
    _write_best(store, cfg, "linear", {"features_hash": "old"})
    with pytest.raises(TunedParamsStale, match="--force"):
        tune_stage.tune_model(cfg, "linear", data, store=store)
    # the same selection still short-circuits (Appendix C)
    _write_best(store, cfg, "linear", {"features_hash": data.features.hash})
    assert tune_stage.tune_model(cfg, "linear", data, store=store).skipped


def test_study_provenance_is_written_before_tuning_and_checked_on_reentry(
    cfg, data, store, tunable, monkeypatch
) -> None:
    seen: list = []
    _fake_tune(monkeypatch, seen)
    with pytest.raises(_Stop):  # "interrupted": no best_params.json is written
        tune_stage.tune_model(cfg, "linear", data, store=store)
    assert tune_stage.STUDY_PROVENANCE in seen[0]
    assert seen[1]["features_hash"] == data.features.hash
    assert seen[1]["features_source"] == data.features.source

    other = replace(data, features=replace(data.features, hash="reselected"))
    with pytest.raises(TunedParamsStale, match=tune_stage.STUDY_PROVENANCE):
        tune_stage.tune_model(cfg, "linear", other, store=store)


def test_force_archives_a_stale_study_instead_of_deleting_it(
    cfg, data, store, tunable, monkeypatch
) -> None:
    directory = _write_best(store, cfg, "linear", {"features_hash": "0123456789abcdef"})
    (directory / "optuna.sqlite3").write_bytes(b"old study")
    seen: list = []
    _fake_tune(monkeypatch, seen)
    with pytest.raises(_Stop):
        tune_stage.tune_model(cfg, "linear", data, store=store, force=True)

    archive = directory.with_name("linear.stale-01234567")
    assert (archive / "optuna.sqlite3").read_bytes() == b"old study"
    assert (archive / "best_params.json").is_file()
    # the new study started in a fresh directory: only the provenance marker
    assert seen[0] == [tune_stage.STUDY_PROVENANCE]

    # a second archive of the same hash never overwrites the first
    (directory / "best_params.json").write_text(
        json.dumps({"best_params": {}, "provenance": {"features_hash": "0123456789abcdef"}})
    )
    with pytest.raises(_Stop):
        tune_stage.tune_model(cfg, "linear", data, store=store, force=True)
    assert archive.is_dir() and directory.with_name("linear.stale-01234567.1").is_dir()


def test_imported_thesis_params_are_stale_for_a_figure_study(
    cfg, data, store, tunable
) -> None:
    _write_best(store, cfg, "linear", {"audit": "2026-09-26 B1"})
    # legacy: the imported params are the thesis' and are reused
    assert tune_stage.tune_model(cfg, "linear", data, store=store).skipped
    with pytest.raises(TunedParamsStale, match="imported thesis params"):
        tune_stage.tune_model(_figure(cfg), "linear", data, store=store)
