"""The populated registry must hold every in-scope variant of plan §2.1.

``strikecast.models.spec`` only defines the contract; the specs register
themselves when their module is imported. ``strikecast.models.registry``
imports all four spec modules, so this is the one place where the whole lineup
is visible and can be checked against the plan's inventory:

===========  ==========================================================
experiment   variants (plan §2.1, §2.3)
===========  ==========================================================
``count``    6 GBDT (lightgbm/xgboost/catboost × poisson/tweedie)
             + 15 RNN ({lstm, gru} × {poisson, tweedie, mse} × {7, 14, 28},
             minus the three gru_mse variants the script never ran)
``diff``     3 GBDT (MSE) + 6 RNN + ``linear`` + ``arima`` + 2 naives
``hurdle``   the two stage models + the ``hurdle`` composite
``damage``   the classifier + the ``damage`` composite
===========  ==========================================================

The names are asserted literally rather than derived from the modules under
test, so a rename or a dropped variant fails here instead of silently changing
the lineup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import shim for `tests.`
    sys.path.insert(0, str(REPO_ROOT))

from strikecast.models import registry  # noqa: E402
from strikecast.models.registry import (  # noqa: E402
    EXPECTED_COUNTS,
    all_specs,
    get_spec,
    registered_experiments,
    registered_names,
    specs_by_experiment,
)

# --------------------------------------------------------------------------- #
# the lineup of plan §2.1, spelled out                                         #
# --------------------------------------------------------------------------- #
COUNT_GBDT = (
    "lightgbm_poisson",
    "lightgbm_tweedie",
    "xgboost_poisson",
    "xgboost_tweedie",
    "catboost_poisson",
    "catboost_tweedie",
)
COUNT_RNN = (
    "lstm_poisson_w7",
    "lstm_poisson_w14",
    "lstm_poisson_w28",
    "lstm_tweedie_w7",
    "lstm_tweedie_w14",
    "lstm_tweedie_w28",
    "lstm_w7",
    "lstm_w14",
    "lstm_w28",
    "gru_poisson_w7",
    "gru_poisson_w14",
    "gru_poisson_w28",
    "gru_tweedie_w7",
    "gru_tweedie_w14",
    "gru_tweedie_w28",
)
DIFF_GBDT = ("lightgbm", "xgboost", "catboost")
DIFF_RNN = ("lstm_w7", "lstm_w14", "lstm_w28", "gru_w7", "gru_w14", "gru_w28")
DIFF_BASELINES = ("linear", "arima", "naive_last", "naive_weekly")
HURDLE = ("spe_event_classifier", "catboost_tweedie_count_head", "hurdle")
DAMAGE = ("spe_damage_classifier", "damage")

EXPECTED: dict[str, tuple[str, ...]] = {
    "count": COUNT_GBDT + COUNT_RNN,
    "diff": DIFF_GBDT + DIFF_RNN + DIFF_BASELINES,
    "hurdle": HURDLE,
    "damage": DAMAGE,
}

#: Names the legacy scripts reused across families for DIFFERENT models.
SHARED_NAMES = ("lstm_w7", "lstm_w14", "lstm_w28")


# --------------------------------------------------------------------------- #
# completeness                                                                 #
# --------------------------------------------------------------------------- #
#: `chronos2` (Stream H, `models/chronos.py`) registers its two specs on import,
#: but `models/registry.py` does NOT import that module yet -- the one-line
#: `from . import chronos` is the pending hook (plan sec. 8 P5). So the family
#: appears in the process-global registry only when another test module has
#: imported it, which makes an `==` assertion order-dependent. Drop this set
#: and restore the `==` once the hook lands.
PENDING_EXPERIMENTS = {"chronos2"}


def test_the_registry_holds_exactly_the_in_scope_experiments():
    got = set(registered_experiments())
    core = {"count", "damage", "diff", "hurdle"}
    assert core <= got
    assert got <= core | PENDING_EXPERIMENTS


@pytest.mark.parametrize("experiment", sorted(EXPECTED))
def test_every_in_scope_variant_is_registered(experiment: str):
    assert set(registered_names(experiment)) == set(EXPECTED[experiment])


@pytest.mark.parametrize("experiment", sorted(EXPECTED))
def test_counts_match_the_plan(experiment: str):
    """6 + 15 for count, 3 + 6 + 4 for diff, 3 for hurdle, 2 for damage."""
    assert len(EXPECTED[experiment]) == EXPECTED_COUNTS[experiment]
    assert len(all_specs(experiment)) == EXPECTED_COUNTS[experiment]
    assert len(specs_by_experiment()[experiment]) == EXPECTED_COUNTS[experiment]

    assert EXPECTED_COUNTS["count"] == 6 + 15
    assert EXPECTED_COUNTS["diff"] == 3 + 6 + 4


@pytest.mark.parametrize(
    ("experiment", "name"),
    [(experiment, name) for experiment, names in EXPECTED.items() for name in names],
)
def test_each_variant_resolves_for_its_experiment(experiment: str, name: str):
    spec = get_spec(name, experiment)
    assert spec.name == name
    assert experiment in spec.experiments
    assert spec.kind in {"global", "local", "naive", "composite"}


def test_no_variant_leaks_into_another_experiment():
    """Membership is exactly the plan's; a name in two families is deliberate.

    ``lstm_w7/14/28`` are the only names the legacy scripts reused, for an
    MSE-on-levels RNN in the count family and an MSE-on-differences one in the
    diff family. Everything else belongs to exactly one experiment.
    """
    for experiment, names in EXPECTED.items():
        others = {n for e, ns in EXPECTED.items() if e != experiment for n in ns}
        assert set(names) - others == set(names) - set(SHARED_NAMES)

    for name in SHARED_NAMES:
        assert get_spec(name, "count") is not get_spec(name, "diff")
        with pytest.raises(KeyError, match="pass experiment="):
            get_spec(name)


def test_registry_names_are_unique_per_experiment():
    for experiment in EXPECTED:
        names = [spec.name for spec in all_specs(experiment)]
        assert len(names) == len(set(names))


# --------------------------------------------------------------------------- #
# what the pipeline reads off a spec                                           #
# --------------------------------------------------------------------------- #
def test_tunable_specs_are_exactly_the_tuned_lineup():
    """Optuna ran on the GBDTs and the RNNs; no baseline was ever tuned."""
    tuned = {(e, s.name) for e in ("count", "diff") for s in all_specs(e) if s.tunable}
    assert tuned == {
        *[("count", n) for n in COUNT_GBDT + COUNT_RNN],
        *[("diff", n) for n in DIFF_GBDT + DIFF_RNN],
    }
    for name in DIFF_BASELINES:
        spec = get_spec(name, "diff")
        assert not spec.tunable
        assert spec.n_trials is None
        assert spec.stochastic is False  # deterministic: run once, broadcast (§5.4)


def test_neural_specs_are_exactly_the_rnns():
    neural = {(e, s.name) for e in EXPECTED for s in all_specs(e) if s.is_neural}
    assert neural == {
        *[("count", n) for n in COUNT_RNN],
        *[("diff", n) for n in DIFF_RNN],
    }
    # F28: the RNNs are the only models fed the raw (un-windowed) past covs.
    raw = {(e, s.name) for e in EXPECTED for s in all_specs(e) if s.needs_raw_past_covs}
    assert raw == neural


def test_every_spec_is_complete_enough_to_run():
    for spec in all_specs():
        assert spec.experiments, spec.name
        assert callable(spec.build), spec.name
        assert spec.family, spec.name
        if spec.tunable:
            assert spec.n_trials is not None, spec.name
        if spec.kind == "naive":
            assert spec.build(spec.defaults, registry.RunContext()) is None, spec.name


# --------------------------------------------------------------------------- #
# import cost                                                                  #
# --------------------------------------------------------------------------- #
HEAVY = ("torch", "darts", "lightgbm", "xgboost", "catboost", "optuna")


def test_importing_the_package_does_not_import_the_model_libraries():
    """`import strikecast.models` must stay cheap; the registry is opt-in."""
    code = (
        "import sys; import strikecast.models; "
        f"print([m for m in {HEAVY!r} if m in sys.modules])"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout


def test_attribute_access_resolves_lazily_through_the_registry():
    """The package re-exports the registry lookups, not the empty contract."""
    import strikecast.models as models  # noqa: PLC0415

    assert models.get_spec is get_spec
    assert models.all_specs is all_specs
    assert models.get_spec("catboost_poisson", "count").family == "catboost"
    assert "get_spec" in dir(models)
    with pytest.raises(AttributeError):
        models.not_a_thing  # noqa: B018
