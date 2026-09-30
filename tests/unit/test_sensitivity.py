"""``sensitivity.future_covariate_lags``: the opt-in future-window switch (audit 2026-09-30).

The switch must (1) reach the darts model, (2) leave every default run bit-for-bit
unchanged (no key in the config dump, legacy ``(2, 7)`` skeleton), and (3) stay
out of the feature-selection hash, so a sensitivity run reuses the publication
selection and its tuned parameters.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import strikecast.models.gbm  # noqa: F401  (populates the registry)
from strikecast.config.loader import load_experiment
from strikecast.config.schema import ExperimentConfig, SensitivityConfig
from strikecast.data.feature_selection import legacy_common_kwargs
from strikecast.models.spec import RunContext, darts_common_kwargs, get_spec
from strikecast.pipeline import data_stage

OVERRIDE = "+sensitivity.future_covariate_lags=[8,7]"


def _count(*extra: str) -> ExperimentConfig:
    return load_experiment("count", ["paradigm=global", "hydra.job.chdir=false", *extra])


def test_span_is_a_tuple_and_rejects_other_lengths():
    assert SensitivityConfig(future_covariate_lags=[8, 7]).future_covariate_lags == (8, 7)
    assert SensitivityConfig().future_covariate_lags is None
    with pytest.raises(ValidationError, match="2-element span"):
        SensitivityConfig(future_covariate_lags=[-8, -1, 0])


def test_default_config_has_no_sensitivity_and_dumps_without_it():
    cfg = _count()
    assert cfg.sensitivity is None
    assert "sensitivity" not in cfg.model_dump(mode="json")
    assert cfg.to_run_context("lightgbm_poisson", 42).future_lags is None


def test_override_reaches_config_dump_and_context():
    cfg = _count(OVERRIDE)
    assert cfg.sensitivity is not None and cfg.sensitivity.future_covariate_lags == (8, 7)
    assert cfg.model_dump(mode="json")["sensitivity"] == {"future_covariate_lags": [8, 7]}
    assert ExperimentConfig.model_validate(cfg.model_dump(mode="json")) == cfg
    assert cfg.to_run_context("lightgbm_poisson", 42).future_lags == (8, 7)


def test_darts_kwargs_default_is_legacy_and_override_replaces_only_the_future_span():
    assert darts_common_kwargs(RunContext()) == legacy_common_kwargs()
    kw = darts_common_kwargs(RunContext(future_lags=(8, 7)))
    assert kw["lags_future_covariates"] == (8, 7)
    assert isinstance(kw["lags_future_covariates"], tuple)
    rest = {k: v for k, v in kw.items() if k != "lags_future_covariates"}
    assert rest == {k: v for k, v in legacy_common_kwargs().items() if k != "lags_future_covariates"}


def test_built_model_sees_the_wider_window():
    spec = get_spec("lightgbm_poisson", "count")
    wide = spec.build(dict(spec.defaults), RunContext(future_lags=(8, 7)))
    legacy = spec.build(dict(spec.defaults), RunContext())
    assert sorted(wide.lags["future"]) == list(range(-8, 7))
    assert sorted(legacy.lags["future"]) == list(range(-2, 7))


def test_composite_heads_keep_the_future_window():
    ctx = RunContext(future_lags=(8, 7), head_past_lags=(("regressor", None),))
    assert ctx.for_head("regressor").future_lags == (8, 7)


@pytest.mark.parametrize("selection", ["poisson", "tweedie"])
def test_feature_selection_hash_ignores_the_switch(selection):
    base = _count().for_selection(selection)
    sens = _count(OVERRIDE).for_selection(selection)
    ph, sh = data_stage._panel_hash(base), data_stage._series_hash(base, data_stage._panel_hash(base))
    assert data_stage._features_hash(base, ph, sh) == data_stage._features_hash(sens, ph, sh)
