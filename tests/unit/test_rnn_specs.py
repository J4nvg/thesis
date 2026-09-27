"""The RNN specs must build exactly what the legacy scripts built.

Every assertion compares ``strikecast.models.rnn`` against the verbatim copies
in ``tests.legacy_ref.rnn_builders`` on identical inputs: the dead default
branch, a fixed params dict, and the stored ``best_params.json`` of the run
that is in ``golden/``. Nothing here trains: no ``.fit`` is ever called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna
import pytest
from pytorch_lightning.callbacks import EarlyStopping
from tests.legacy_ref.rnn_builders import COUNT, DIFF

from strikecast.models import rnn
from strikecast.models.spec import RunContext, get_spec, registered_names

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_TUNING = REPO_ROOT / "golden" / "converted" / "tuning"

CTX = RunContext()  # seed=42, the value both scripts pinned as RANDOM_STATE

# A params dict that is NOT any variant's defaults and NOT any stored best, so
# the comparison cannot pass by accident.
FIXED_PARAMS: dict[str, Any] = dict(
    hidden_dim=64,
    n_rnn_layers=2,
    hidden_fc_sizes=[64, 32],
    dropout=0.25,
    batch_size=128,
    n_epochs=100,
    optimizer_kwargs={"lr": 3e-4, "weight_decay": 5e-5},
    add_encoders={"cyclic": {"past": ["month", "week", "dayofyear", "dayofweek", "day"]}},
)


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------
def _describe_callback(cb: Any) -> tuple[Any, ...]:
    """Callbacks are fresh objects on both sides, so compare what they do."""
    if isinstance(cb, EarlyStopping):
        return ("EarlyStopping", cb.monitor, cb.patience, cb.min_delta, cb.mode)
    return ("pruning", type(cb).__mro__[-2].__name__, getattr(cb, "monitor", None))


def _normalise(model_params: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(model_params).items():
        if key == "loss_fn" and value is not None:
            out[key] = (type(value).__name__, getattr(value, "power", None))
        elif key == "pl_trainer_kwargs" and value is not None:
            tk = dict(value)
            tk["callbacks"] = [_describe_callback(c) for c in tk.get("callbacks", [])]
            out[key] = tk
        else:
            out[key] = value
    return out


def assert_same_model(built: Any, legacy: Any) -> None:
    """The two darts models are the same construction."""
    assert type(built) is type(legacy)
    assert _normalise(built.model_params) == _normalise(legacy.model_params)
    # F57: the guard that gates sampling must keep seeing likelihood=None.
    assert built.likelihood is None
    assert legacy.likelihood is None
    # F108: the diff family sets no marker at all; the count family always does.
    assert hasattr(built, "_count_log_link") == hasattr(legacy, "_count_log_link")
    assert getattr(built, "_count_log_link", None) == getattr(legacy, "_count_log_link", None)
    # Explicit re-checks of the fields the plan calls out, so a regression in
    # the dict comparison above cannot hide them.
    bp, lp = built.model_params, legacy.model_params
    assert bp.get("optimizer_kwargs") == lp.get("optimizer_kwargs")
    assert bp.get("add_encoders") == lp.get("add_encoders")
    assert bp.get("random_state") == lp.get("random_state")
    if "loss_fn" in lp:
        assert type(bp["loss_fn"]).__name__ == type(lp["loss_fn"]).__name__
        assert getattr(bp["loss_fn"], "power", None) == getattr(lp["loss_fn"], "power", None)
    else:
        assert "loss_fn" not in bp  # F107: darts' default MSE, never passed


def _golden_best(spec_name: str, experiment: str) -> dict[str, Any] | None:
    """The stored ``best_params`` of ``spec_name`` in ``experiment``.

    The variant directory is the spec name in both families; only the tuning
    checkpoint directory differs (``checkpoints_tune`` for count,
    ``checkpoints_tune_diff`` for diff), which is what
    ``rnn.GOLDEN_TUNING_DIR`` maps. The experiment has to be passed because
    ``lstm_w7/w14/w28`` exist in both families with different stored studies.
    """
    path = GOLDEN_TUNING / rnn.GOLDEN_TUNING_DIR[experiment] / spec_name / "best_params.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["best_params"]


#: ``(name, experiment)`` for every RNN spec, the shape `get_spec` needs.
ALL_VARIANTS = [
    *[(n, "count") for n in rnn.COUNT_VARIANTS],
    *[(n, "diff") for n in rnn.DIFF_VARIANTS],
]


# ---------------------------------------------------------------------------
# registry shape
# ---------------------------------------------------------------------------
def test_registered_lineups_match_the_legacy_scripts() -> None:
    # The registry is process-global and other spec modules register into the
    # same two families, so compare the RNN slice of it, in order.
    count = registered_names("count")
    assert [n for n in count if n in rnn.COUNT_VARIANTS] == list(rnn.COUNT_VARIANTS)
    diff = registered_names("diff")
    assert [n for n in diff if n in rnn.DIFF_VARIANTS] == list(rnn.DIFF_VARIANTS)
    assert len(rnn.COUNT_VARIANTS) == 15
    assert len(rnn.DIFF_VARIANTS) == 6
    # every RNN spec is reachable for its own family, and only for it
    for name, experiment in ALL_VARIANTS:
        assert get_spec(name, experiment).experiments == (experiment,)


@pytest.mark.parametrize(("name", "experiment"), ALL_VARIANTS)
def test_spec_flags(name: str, experiment: str) -> None:
    spec = get_spec(name, experiment)
    assert spec.kind == "global"
    assert spec.is_neural is True
    assert spec.stochastic is True
    assert spec.needs_raw_past_covs is True  # F28: raw past covs, MinMax scaled
    assert spec.tunable
    assert spec.n_trials == 50  # OPTUNA_N_TRIALS in both scripts
    assert spec.device == "auto"  # the tuned presets' accelerator literal
    assert spec.threads_from_context is False
    assert spec.fallback is None


@pytest.mark.parametrize(("name", "experiment"), ALL_VARIANTS)
def test_every_spec_has_a_golden_tuning_artefact(name: str, experiment: str) -> None:
    path = GOLDEN_TUNING / rnn.GOLDEN_TUNING_DIR[experiment] / name / "best_params.json"
    if not path.exists():
        pytest.skip(f"golden artefact not available: {path}")
    payload = json.loads(path.read_text())
    assert payload["variant"] == name
    assert payload["source_dir"] == rnn.GOLDEN_TUNING_DIR[experiment]
    assert payload["n_trials"] == get_spec(name, experiment).n_trials


def test_variant_parsers_match_legacy() -> None:
    from src.general_tools import parse_lstm_variant  # noqa: PLC0415

    for variant in rnn.COUNT_VARIANTS:
        assert rnn.parse_count_variant(variant) == COUNT._parse_lstm_variant(variant)
    for name in rnn.DIFF_VARIANTS:
        # `_diff_regression.py` gets its parser from `src.general_tools`
        # through `from src import *`, and the spec name IS the legacy variant.
        assert rnn.parse_diff_variant(name) == parse_lstm_variant(name)


# ---------------------------------------------------------------------------
# trainer kwargs (F10, F100-F105)
# ---------------------------------------------------------------------------
def test_default_trainer_kwargs_match_legacy() -> None:
    for ours, theirs in (
        (rnn.count_default_trainer_kwargs(), COUNT.NN_TRAINER_KWARGS),
        (rnn.diff_default_trainer_kwargs(), DIFF.NN_TRAINER_KWARGS),
    ):
        a, b = dict(ours), dict(theirs)
        assert [_describe_callback(c) for c in a.pop("callbacks")] == [
            _describe_callback(c) for c in b.pop("callbacks")
        ]
        assert a == b
    # F10 / F100: the literals, spelled out.
    assert rnn.COUNT_NN_TRAINER_KWARGS["accelerator"] == "gpu"
    assert rnn.COUNT_NN_TRAINER_KWARGS["devices"] == 1
    assert rnn.COUNT_NN_TRAINER_KWARGS["precision"] == "32-true"
    assert rnn.DIFF_NN_TRAINER_KWARGS["accelerator"] == "auto"
    assert "devices" not in rnn.DIFF_NN_TRAINER_KWARGS
    for kwargs in (rnn.COUNT_NN_TRAINER_KWARGS, rnn.DIFF_NN_TRAINER_KWARGS):
        assert kwargs["enable_progress_bar"] is True
        assert kwargs["gradient_clip_val"] == 1.0
        (es,) = kwargs["callbacks"]
        assert (es.monitor, es.patience) == ("train_loss", 10)


def test_tuned_trainer_kwargs_match_legacy() -> None:
    for ours, theirs in (
        (rnn.count_tuned_trainer_kwargs(variant="lstm_poisson_w7"), COUNT._nn_trainer_kwargs()),
        (rnn.diff_tuned_trainer_kwargs(), DIFF._nn_trainer_kwargs()),
    ):
        a, b = dict(ours), dict(theirs)
        assert [_describe_callback(c) for c in a.pop("callbacks")] == [
            _describe_callback(c) for c in b.pop("callbacks")
        ]
        assert a == b
        assert a["accelerator"] == "auto"
        assert a["enable_progress_bar"] is False
        assert a["enable_model_summary"] is False
        assert a["gradient_clip_val"] == 1.0


def test_pruning_callback_only_at_tuning_time() -> None:
    """F103: the trial's model carries the pruning callback, the rebuild does not."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    trial = study.ask()
    for ours, theirs in (
        (rnn.count_tuned_trainer_kwargs(trial, variant="lstm_w7"), COUNT._nn_trainer_kwargs(trial)),
        (rnn.diff_tuned_trainer_kwargs(trial), DIFF._nn_trainer_kwargs(trial)),
    ):
        assert [_describe_callback(c) for c in ours["callbacks"]] == [
            _describe_callback(c) for c in theirs["callbacks"]
        ]
        assert len(ours["callbacks"]) == 2
        assert ours["callbacks"][1].monitor == "train_loss"
    assert len(rnn.count_tuned_trainer_kwargs()["callbacks"]) == 1
    assert len(rnn.diff_tuned_trainer_kwargs()["callbacks"]) == 1


# ---------------------------------------------------------------------------
# (a) defaults
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", rnn.COUNT_VARIANTS)
def test_count_defaults_are_the_legacy_default_branch(name: str) -> None:
    """``build_regressor("lstm")`` verbatim, bar the two variant-fixed fields."""
    model_type, _objective, icl = rnn.parse_count_variant(name)
    legacy = COUNT.build_regressor("lstm")
    ours = dict(get_spec(name, "count").defaults)
    assert ours.pop("model") == model_type
    assert ours.pop("input_chunk_length") == icl
    theirs = dict(legacy.model_params)
    for key in ("model", "input_chunk_length", "hidden_fc_sizes", "activation",
                "output_chunk_shift", "use_static_covariates"):
        theirs.pop(key, None)
    # `_normalise` compares the callbacks by what they do: both sides hold
    # fresh `EarlyStopping` objects, which are never `==` to each other.
    assert _normalise(ours) == _normalise(theirs)
    # F106: the dead default branch encodes on `future`, the tuned path on `past`.
    assert "future" in ours["add_encoders"]["cyclic"]
    assert ours["pl_trainer_kwargs"] is COUNT.NN_TRAINER_KWARGS or (
        ours["pl_trainer_kwargs"]["accelerator"] == "gpu"
    )


@pytest.mark.parametrize("name", rnn.DIFF_VARIANTS)
def test_diff_defaults_are_the_legacy_default_branch(name: str) -> None:
    model_type, icl = rnn.parse_diff_variant(name)
    legacy = DIFF.build_regressor("lstm")
    ours = dict(get_spec(name, "diff").defaults)
    assert ours.pop("model") == model_type
    assert ours.pop("input_chunk_length") == icl
    theirs = dict(legacy.model_params)
    for key in ("model", "input_chunk_length", "hidden_fc_sizes", "activation",
                "output_chunk_shift", "use_static_covariates"):
        theirs.pop(key, None)
    assert _normalise(ours) == _normalise(theirs)
    assert "past" in ours["add_encoders"]["cyclic"]


@pytest.mark.parametrize("name", rnn.COUNT_VARIANTS)
def test_count_build_from_defaults_matches_legacy(name: str) -> None:
    _model_type, objective_kind, _icl = rnn.parse_count_variant(name)
    spec = get_spec(name, "count")
    params = dict(spec.defaults)
    built = spec.build(params, CTX)
    legacy = COUNT.build_lstm_count(objective_kind, dict(params))
    assert_same_model(built, legacy)


@pytest.mark.parametrize("name", rnn.DIFF_VARIANTS)
def test_diff_build_from_defaults_matches_legacy(name: str) -> None:
    spec = get_spec(name, "diff")
    params = dict(spec.defaults)
    built = spec.build(params, CTX)
    legacy = DIFF.build_lstm_from_params(dict(params))
    assert_same_model(built, legacy)


# ---------------------------------------------------------------------------
# (b) a fixed params dict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", rnn.COUNT_VARIANTS)
def test_count_build_from_fixed_params_matches_legacy(name: str) -> None:
    model_type, objective_kind, icl = rnn.parse_count_variant(name)
    params = dict(
        FIXED_PARAMS,
        model=model_type,
        input_chunk_length=icl,
        output_chunk_length=rnn.OUTPUT_CHUNK_LEN,
        random_state=rnn.RANDOM_STATE,
        pl_trainer_kwargs=rnn.count_tuned_trainer_kwargs(variant=name),
    )
    tweedie_power = 1.234
    built = get_spec(name, "count").build({**params, "tweedie_variance_power": tweedie_power}, CTX)
    legacy_params = dict(params, pl_trainer_kwargs=COUNT._nn_trainer_kwargs(variant=name))
    legacy = COUNT.build_lstm_count(objective_kind, legacy_params, tweedie_power=tweedie_power)
    assert_same_model(built, legacy)
    if objective_kind == "tweedie":
        assert built.model_params["loss_fn"].power == tweedie_power
    assert built._count_log_link is (objective_kind != "mse")


@pytest.mark.parametrize("name", rnn.DIFF_VARIANTS)
def test_diff_build_from_fixed_params_matches_legacy(name: str) -> None:
    model_type, icl = rnn.parse_diff_variant(name)
    params = dict(
        FIXED_PARAMS,
        model=model_type,
        input_chunk_length=icl,
        output_chunk_length=rnn.OUTPUT_CHUNK_LEN,
        random_state=rnn.RANDOM_STATE,
        pl_trainer_kwargs=rnn.diff_tuned_trainer_kwargs(),
    )
    built = get_spec(name, "diff").build(params, CTX)
    legacy = DIFF.build_lstm_from_params(
        dict(params, pl_trainer_kwargs=DIFF._nn_trainer_kwargs())
    )
    assert_same_model(built, legacy)
    assert not hasattr(built, "_count_log_link")  # F108


# ---------------------------------------------------------------------------
# (c) the stored best params
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", rnn.COUNT_VARIANTS)
def test_count_build_from_golden_best_matches_legacy(name: str) -> None:
    best = _golden_best(name, "count")
    if best is None:
        pytest.skip("golden best_params.json not available")
    assert isinstance(best["hidden_fc_sizes"], str)  # F109: stored as a string
    built = get_spec(name, "count").build(get_spec(name, "count").from_best_params(best), CTX)
    legacy = COUNT._build_lstm_from_best(name, dict(best))()
    assert_same_model(built, legacy)


@pytest.mark.parametrize("name", rnn.DIFF_VARIANTS)
def test_diff_build_from_golden_best_matches_legacy(name: str) -> None:
    best = _golden_best(name, "diff")
    if best is None:
        pytest.skip("golden best_params.json not available")
    # the spec name IS the legacy variant name in the diff script
    built = get_spec(name, "diff").build(get_spec(name, "diff").from_best_params(best), CTX)
    legacy = DIFF._build_lstm_from_best(name, dict(best))()
    assert_same_model(built, legacy)


def test_from_best_params_maps_the_stored_string_and_floats() -> None:
    best = _golden_best("lstm_tweedie_w14", "count")
    if best is None:
        pytest.skip("golden best_params.json not available")
    params = get_spec("lstm_tweedie_w14").from_best_params(best)
    assert params["hidden_fc_sizes"] == [32]  # "32" -> [32]
    assert params["optimizer_kwargs"] == {"lr": best["lr"], "weight_decay": best["weight_decay"]}
    assert params["input_chunk_length"] == 14  # from the variant, never from best
    assert params["n_epochs"] == 100  # F103: same as the trial
    assert params["tweedie_variance_power"] == best["tweedie_variance_power"]
    (es,) = [c for c in params["pl_trainer_kwargs"]["callbacks"] if isinstance(c, EarlyStopping)]
    assert es.patience == 5  # F10: the tuned preset
    assert len(params["pl_trainer_kwargs"]["callbacks"]) == 1  # no pruning on rebuild


def test_from_best_params_without_tweedie_power_falls_back_to_1_5() -> None:
    best = _golden_best("lstm_poisson_w7", "count")
    if best is None:
        pytest.skip("golden best_params.json not available")
    assert "tweedie_variance_power" not in best
    params = get_spec("lstm_poisson_w7").from_best_params(best)
    assert params["tweedie_variance_power"] == 1.5


# ---------------------------------------------------------------------------
# seeds (F110)
# ---------------------------------------------------------------------------
def test_random_state_comes_from_the_context() -> None:
    spec = get_spec("lstm_poisson_w7")
    params = spec.from_best_params(_golden_best("lstm_poisson_w7", "count") or {"hidden_dim": 16,
        "n_rnn_layers": 1, "hidden_fc_sizes": "none", "dropout": 0.1, "batch_size": 32,
        "lr": 1e-3, "weight_decay": 1e-5})
    assert params["random_state"] == rnn.RANDOM_STATE == 42
    assert spec.build(params, RunContext()).model_params["random_state"] == 42
    assert spec.build(params, RunContext(seed=7)).model_params["random_state"] == 7
    # Legacy pinned 42, so the default context reproduces it exactly.
    legacy = COUNT._build_lstm_from_best("lstm_poisson_w7", dict(params_only(params)))()
    assert legacy.model_params["random_state"] == 42


def params_only(params: dict[str, Any]) -> dict[str, Any]:
    """The flat ``best_params`` shape ``_build_lstm_from_best`` expects."""
    return {
        "hidden_dim": params["hidden_dim"],
        "n_rnn_layers": params["n_rnn_layers"],
        "hidden_fc_sizes": next(
            k for k, v in rnn.FC_MAP.items() if v == params["hidden_fc_sizes"]
        ),
        "dropout": params["dropout"],
        "batch_size": params["batch_size"],
        "lr": params["optimizer_kwargs"]["lr"],
        "weight_decay": params["optimizer_kwargs"]["weight_decay"],
    }


# ---------------------------------------------------------------------------
# search spaces
# ---------------------------------------------------------------------------
def _run_study(suggest: Any, n_trials: int = 3) -> list[dict[str, Any]]:
    """Three trials with a pinned TPE sampler; returns what each trial drew."""
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=rnn.RANDOM_STATE)
    )
    drawn: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        drawn.append(suggest(trial))
        return float(len(drawn))

    study.optimize(objective, n_trials=n_trials)
    return drawn


def _strip(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    tk = out.pop("pl_trainer_kwargs", None)
    if tk is not None:
        out["_trainer"] = {k: v for k, v in tk.items() if k != "callbacks"}
        out["_callbacks"] = [_describe_callback(c) for c in tk["callbacks"]]
    return out


@pytest.mark.parametrize("name", rnn.COUNT_VARIANTS)
def test_count_search_space_matches_legacy(name: str) -> None:
    model_type, objective_kind, icl = rnn.parse_count_variant(name)
    spec = get_spec(name, "count")
    assert spec.search_space is not None

    def legacy(trial: optuna.Trial) -> dict[str, Any]:
        params, extras = COUNT._suggest_lstm_params(
            trial, objective_kind, input_chunk_length=icl, model_type=model_type
        )
        params["pl_trainer_kwargs"] = COUNT._nn_trainer_kwargs(trial, variant=name)
        params.update(extras)
        return params

    ours = _run_study(spec.search_space)
    theirs = _run_study(legacy)
    assert [_strip(p) for p in ours] == [_strip(p) for p in theirs]
    if objective_kind == "tweedie":
        assert all("tweedie_variance_power" in p for p in ours)
    else:
        assert all("tweedie_variance_power" not in p for p in ours)


@pytest.mark.parametrize("name", rnn.DIFF_VARIANTS)
def test_diff_search_space_matches_legacy(name: str) -> None:
    model_type, icl = rnn.parse_diff_variant(name)
    spec = get_spec(name, "diff")
    assert spec.search_space is not None

    def legacy(trial: optuna.Trial) -> dict[str, Any]:
        return DIFF._suggest_lstm_params(trial, input_chunk_length=icl, model_type=model_type)

    assert [_strip(p) for p in _run_study(spec.search_space)] == [
        _strip(p) for p in _run_study(legacy)
    ]


def test_search_space_draws_the_same_names_in_the_same_order() -> None:
    """The TPE sequence depends on the suggestion order, so pin it."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=rnn.RANDOM_STATE))
    trial = study.ask()
    get_spec("lstm_tweedie_w7").search_space(trial)
    assert list(trial.params) == [
        "hidden_fc_sizes",
        "hidden_dim",
        "n_rnn_layers",
        "dropout",
        "batch_size",
        "lr",
        "weight_decay",
        "tweedie_variance_power",
    ]


@pytest.mark.parametrize("name", rnn.COUNT_VARIANTS[:3])
def test_a_trial_model_builds_from_the_search_space(name: str) -> None:
    """The params a trial draws go straight into ``build`` -- no ``.fit``."""
    spec = get_spec(name, "count")
    assert spec.search_space is not None
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=rnn.RANDOM_STATE))
    params = spec.search_space(study.ask())
    model = spec.build(params, CTX)
    assert model.likelihood is None  # F57
    assert model.model_params["n_epochs"] == 100
    cbs = model.model_params["pl_trainer_kwargs"]["callbacks"]
    assert len(cbs) == 2  # early stopping + pruning, at tuning time only
