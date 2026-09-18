"""RNN model specs: the count family and the diff family (Phase 3, §5.2).

Everything here is a re-expression of two legacy scripts, with the same darts
objects coming out the other end:

* ``_regression_LSTM.py`` -- the COUNT family. ``BlockRNNModel`` with a
  log-link Poisson or Tweedie NLL loss module, or plain MSE, and the
  ``_count_log_link`` marker attribute the adapters read (F55-F57). Fifteen
  tuned variants: ``{lstm,gru} x {poisson,tweedie} x {w7,w14,w28}`` plus
  ``lstm_{w7,w14,w28}`` on MSE.
* ``_diff_regression.py`` -- the DIFF family. ``BlockRNNModel`` on the
  differenced target with darts' default loss (MSE), no log link, no
  ``_count_log_link`` attribute at all. Six tuned variants:
  ``{lstm,gru} x {w7,w14,w28}``.

``lstm_w7``, ``lstm_w14`` and ``lstm_w28`` are names BOTH families use, for
different models. ``register`` takes one spec per name per experiment family,
so both keep their legacy names and callers disambiguate with
``get_spec(name, experiment="count" | "diff")``.

Methodology is frozen (``docs/REFACTOR_PLAN.md`` §8 P3). Nothing here is a fix:

* Every builder forces or inherits ``likelihood=None``, which is what keeps
  F55, F56 and F58 latent (F57). Do not put a likelihood in a search space
  without revisiting all four flags.
* Early-stopping patience is 10 in the default trainer kwargs and 5 in the
  tuned ones, both on ``train_loss`` (F10). Both presets are kept.
* The count family's RNNs take the RAW, un-windowed past covariates, MinMax
  scaled by the adapter (F28, F1), hence ``needs_raw_past_covs=True``.
* ``accelerator`` is a literal in both scripts -- ``"gpu"`` in the count
  family's default preset, ``"auto"`` everywhere else. The legacy code never
  made it configurable, so ``ctx.device`` is NOT mapped onto it; see
  :data:`RNN_FLAGS`.
* ``ctx.seed`` maps onto darts' ``random_state``, which the legacy code pinned
  to ``RANDOM_STATE = 42``. With the default context the value is identical;
  §5.4's seed sweep is what the parameter is for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna
import torch
from pytorch_lightning.callbacks import Callback as PLCallback
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.nn import functional as F

from .spec import ModelSpec, RunContext, register

__all__ = [
    "COUNT_NN_TRAINER_KWARGS",
    "COUNT_VARIANTS",
    "DIFF_NN_TRAINER_KWARGS",
    "DIFF_VARIANTS",
    "GOLDEN_TUNING_DIR",
    "INPUT_LAGS",
    "OUTPUT_CHUNK_LEN",
    "RANDOM_STATE",
    "RNN_FLAGS",
    "TWEEDIE_DEFAULT_POWER",
    "CountPruningCallback",
    "DiffPruningCallback",
    "PoissonNLLLogLink",
    "TweedieNLLLogLink",
    "build_lstm_count",
    "build_lstm_from_params",
    "count_default_trainer_kwargs",
    "count_tuned_trainer_kwargs",
    "diff_default_trainer_kwargs",
    "diff_tuned_trainer_kwargs",
    "parse_count_variant",
    "parse_diff_variant",
    "suggest_count_rnn_params",
    "suggest_diff_rnn_params",
]

# --- legacy module globals, identical in both scripts ------------------------
OUTPUT_CHUNK_LEN = 7  # _regression_LSTM.py:97, _diff_regression.py:106
INPUT_LAGS = 7  # _regression_LSTM.py:98, _diff_regression.py:107
RANDOM_STATE = 42  # _regression_LSTM.py:129, _diff_regression.py:133
TWEEDIE_DEFAULT_POWER = 1.5  # build_lstm_count / _build_lstm_from_best default

#: ``hidden_fc_sizes`` is tuned as a STRING categorical and mapped to a list.
#: Stored ``best_params.json`` payloads therefore carry e.g. ``"64_32"``.
FC_MAP: dict[str, list[int]] = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}

#: The cyclic encoders the TUNED path uses. ``past`` in both scripts; the dead
#: default branch of the count family uses ``future`` instead (see RNN_FLAGS).
TUNED_ADD_ENCODERS: dict[str, Any] = {
    "cyclic": {"past": ["month", "week", "dayofyear", "dayofweek", "day"]}
}
COUNT_DEFAULT_ADD_ENCODERS: dict[str, Any] = {
    "cyclic": {"future": ["month", "week", "dayofyear", "dayofweek", "day"]}
}
DIFF_DEFAULT_ADD_ENCODERS: dict[str, Any] = {
    "cyclic": {"past": ["month", "week", "dayofyear", "dayofweek", "day"]}
}


# ---------------------------------------------------------------------------
# Loss modules (verbatim, `_regression_LSTM.py` 255-273)
# ---------------------------------------------------------------------------
# https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/metrics/point.py
class PoissonNLLLogLink(nn.Module):
    """Poisson NLL on a log link. Darts expects a scalar, hence ``mean``."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.poisson_nll_loss(
            y_pred, y_true, log_input=True, full=False, reduction="mean"
        )


class TweedieNLLLogLink(nn.Module):
    """Tweedie NLL on a log link, for ``1 < power < 2``."""

    def __init__(self, power: float = 1.5) -> None:
        super().__init__()
        if not (1.0 < power < 2.0):
            raise ValueError("power must lie strictly in (1, 2)")
        self.power = power

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        p = self.power
        mu = torch.exp(y_pred)
        a = -y_true * torch.pow(mu, 1.0 - p) / (1.0 - p)
        b = torch.pow(mu, 2.0 - p) / (2.0 - p)
        return torch.mean(a + b)


# ---------------------------------------------------------------------------
# Trainer kwargs. Two presets per family (F10): patience 10 in the default
# builder, patience 5 in the tuned builder, both on `train_loss`.
# ---------------------------------------------------------------------------
# The legacy scripts build ONE EarlyStopping instance and ONE dict at module
# level and hand the same objects to every default build (`ES_NN`,
# `NN_TRAINER_KWARGS`). That shared mutable state is reproduced here.
COUNT_ES_NN = EarlyStopping(monitor="train_loss", patience=10, min_delta=1e-4, mode="min")
COUNT_NN_TRAINER_KWARGS: dict[str, Any] = dict(  # _regression_LSTM.py:324-333
    accelerator="gpu",
    devices=1,
    precision="32-true",
    enable_progress_bar=True,
    enable_model_summary=True,
    log_every_n_steps=10,
    callbacks=[COUNT_ES_NN],
    gradient_clip_val=1.0,
)

DIFF_ES_NN = EarlyStopping(monitor="train_loss", patience=10, min_delta=1e-4, mode="min")
DIFF_NN_TRAINER_KWARGS: dict[str, Any] = dict(  # _diff_regression.py:275-282
    accelerator="auto",
    enable_progress_bar=True,
    enable_model_summary=True,
    log_every_n_steps=10,
    callbacks=[DIFF_ES_NN],
    gradient_clip_val=1.0,
)


def count_default_trainer_kwargs() -> dict[str, Any]:
    """The count family's ``NN_TRAINER_KWARGS``, the very object legacy passed."""
    return COUNT_NN_TRAINER_KWARGS


def diff_default_trainer_kwargs() -> dict[str, Any]:
    """The diff family's ``NN_TRAINER_KWARGS``, the very object legacy passed."""
    return DIFF_NN_TRAINER_KWARGS


class _PruningCallbackBase(PLCallback):
    """Optuna pruning that survives PL >= 2.x.

    ``optuna.integration.PyTorchLightningPruningCallback`` raises "Expected a
    parent" under PL 2.x, so both scripts grew their own drop-in.
    """

    def __init__(self, trial: optuna.Trial, monitor: str) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def _maybe_prune(self, trainer: Any) -> None:
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        self.trial.report(float(score.detach().cpu()), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {trainer.current_epoch} "
                f"({self.monitor}={float(score):.4f})"
            )

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        # Using train_loss because no val_series is passed to .fit()
        self._maybe_prune(trainer)


class CountPruningCallback(_PruningCallbackBase):
    """``_regression_LSTM.py:941``. Also hooks ``on_validation_end``, which is
    dead in these runs because ``.fit()`` never gets a ``val_series``."""

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        # Harmless if val isn't run; helpful if you later add val_series
        self._maybe_prune(trainer)


class DiffPruningCallback(_PruningCallbackBase):
    """``_diff_regression.py:1057``. Train-epoch hook only."""


def count_tuned_trainer_kwargs(
    trial: optuna.Trial | None = None, variant: str | None = None
) -> dict[str, Any]:
    """``_regression_LSTM.py::_nn_trainer_kwargs``. ``variant`` is accepted and
    ignored, exactly as in the source."""
    callbacks: list[PLCallback] = [
        EarlyStopping(monitor="train_loss", patience=5, min_delta=1e-4, mode="min")
    ]
    if trial is not None:
        callbacks.append(CountPruningCallback(trial, monitor="train_loss"))

    return dict(
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )


def diff_tuned_trainer_kwargs(trial: optuna.Trial | None = None) -> dict[str, Any]:
    """``_diff_regression.py::_nn_trainer_kwargs``."""
    callbacks: list[PLCallback] = [
        EarlyStopping(monitor="train_loss", patience=5, min_delta=1e-4, mode="min")
    ]
    if trial is not None:
        callbacks.append(DiffPruningCallback(trial, monitor="train_loss"))

    return dict(
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_lstm_count(
    objective_kind: str, params: Mapping[str, Any], *, tweedie_power: float = TWEEDIE_DEFAULT_POWER
) -> Any:
    """``_regression_LSTM.py::build_lstm_count``.

    RNN in deterministic point-forecast mode. Poisson/Tweedie get a log-link
    NLL loss and ``_count_log_link = True``, so the adapter inverts with
    ``exp``; MSE stays in level space with ``_count_log_link = False``.
    ``likelihood`` is popped and forced to ``None`` either way (F57).
    """
    from darts.models import BlockRNNModel  # noqa: PLC0415  (legacy import site)

    if objective_kind == "mse":
        p = dict(params)
        p.pop("likelihood", None)
        m = BlockRNNModel(**p)
        m._count_log_link = False
        return m

    if objective_kind == "poisson":
        loss_fn: nn.Module = PoissonNLLLogLink()
    elif objective_kind == "tweedie":
        loss_fn = TweedieNLLLogLink(power=tweedie_power)
    else:
        raise ValueError(f"unknown objective_kind {objective_kind!r}")

    p = dict(params)
    p.pop("likelihood", None)  # force deterministic -- no sampling
    p["loss_fn"] = loss_fn
    p["likelihood"] = None

    m = BlockRNNModel(**p)
    m._count_log_link = True  # signal to the adapter to invert the link
    return m


def build_lstm_from_params(params: Mapping[str, Any]) -> Any:
    """``_diff_regression.py::build_lstm_from_params``.

    RNN with darts' default MSE loss -- no log link, no count likelihood, and
    NO ``_count_log_link`` attribute (the adapter's ``getattr`` default of
    ``False`` is what covers it).
    """
    from darts.models import BlockRNNModel  # noqa: PLC0415  (legacy import site)

    return BlockRNNModel(**params)


# ---------------------------------------------------------------------------
# Variant parsing
# ---------------------------------------------------------------------------
def parse_count_variant(variant: str) -> tuple[str, str, int]:
    """``_regression_LSTM.py::_parse_lstm_variant``.

    ``'lstm_poisson_w14' -> ('LSTM', 'poisson', 14)``,
    ``'lstm_w7' -> ('LSTM', 'mse', 7)``.
    """
    parts = variant.split("_")
    model_type = parts[0].upper()
    if len(parts) == 3:
        objective_kind = parts[1]
        icl = int(parts[2].lstrip("w"))
    else:
        objective_kind = "mse"
        icl = int(parts[1].lstrip("w"))
    return model_type, objective_kind, icl


def parse_diff_variant(variant: str) -> tuple[str, int]:
    """``src/general_tools.py::parse_lstm_variant``, which ``_diff_regression``
    imports through ``from src import *``. ``'gru_w7' -> ('GRU', 7)``. It takes
    ``parts[-1]``, so a three-part name would parse the same way."""
    parts = variant.split("_")
    model_type = parts[0].upper()
    icl = int(parts[-1].lstrip("w"))
    return model_type, icl


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def suggest_count_rnn_params(
    trial: optuna.Trial,
    objective_kind: str,
    input_chunk_length: int,
    model_type: str = "LSTM",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``_regression_LSTM.py::_suggest_lstm_params``. Suggestion ORDER is part
    of the contract: changing it changes the TPE sequence."""
    fc_choice = trial.suggest_categorical("hidden_fc_sizes", ["none", "32", "64", "64_32"])

    params = dict(
        model=model_type,
        input_chunk_length=input_chunk_length,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_dim=trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
        n_rnn_layers=trial.suggest_int("n_rnn_layers", 1, 3),
        hidden_fc_sizes=FC_MAP[fc_choice],
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        n_epochs=100,
        optimizer_kwargs={
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        },
        random_state=RANDOM_STATE,
        add_encoders=dict(TUNED_ADD_ENCODERS),
    )
    extras: dict[str, Any] = {}
    if objective_kind == "tweedie":
        extras["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.1, 1.9
        )
    return params, extras


def suggest_diff_rnn_params(
    trial: optuna.Trial, input_chunk_length: int, model_type: str = "LSTM"
) -> dict[str, Any]:
    """``_diff_regression.py::_suggest_lstm_params``. Unlike the count twin it
    injects ``pl_trainer_kwargs`` itself and returns a single dict."""
    fc_choice = trial.suggest_categorical("hidden_fc_sizes", ["none", "32", "64", "64_32"])
    return dict(
        model=model_type,
        input_chunk_length=input_chunk_length,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_dim=trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
        n_rnn_layers=trial.suggest_int("n_rnn_layers", 1, 3),
        hidden_fc_sizes=FC_MAP[fc_choice],
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        n_epochs=100,
        optimizer_kwargs={
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        },
        random_state=RANDOM_STATE,
        add_encoders=dict(TUNED_ADD_ENCODERS),
        pl_trainer_kwargs=diff_tuned_trainer_kwargs(trial),
    )


# ---------------------------------------------------------------------------
# Spec plumbing
# ---------------------------------------------------------------------------
def _count_defaults(model_type: str, icl: int) -> dict[str, Any]:
    """``build_regressor("lstm")`` of ``_regression_LSTM.py`` (475-493).

    That branch is dead code -- nothing in the script ever calls it -- and it
    only ever described ``model="LSTM"``, ``input_chunk_length=INPUT_LAGS``.
    Those two fields are taken from the variant; everything else, including
    the ``cyclic.future`` encoders and the shared ``NN_TRAINER_KWARGS``, is
    the branch verbatim. See RNN_FLAGS.
    """
    return dict(
        model=model_type,
        input_chunk_length=icl,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_dim=32,
        n_rnn_layers=1,
        dropout=0.1,
        batch_size=64,
        n_epochs=30,
        random_state=RANDOM_STATE,
        add_encoders=dict(COUNT_DEFAULT_ADD_ENCODERS),
        pl_trainer_kwargs=count_default_trainer_kwargs(),
    )


def _diff_defaults(model_type: str, icl: int) -> dict[str, Any]:
    """``build_regressor("lstm")`` of ``_diff_regression.py`` (368-386), with
    ``model`` and ``input_chunk_length`` from the variant. Also dead code."""
    return dict(
        model=model_type,
        input_chunk_length=icl,
        output_chunk_length=OUTPUT_CHUNK_LEN,
        hidden_dim=32,
        n_rnn_layers=1,
        dropout=0.1,
        batch_size=64,
        n_epochs=30,
        random_state=RANDOM_STATE,
        add_encoders=dict(DIFF_DEFAULT_ADD_ENCODERS),
        pl_trainer_kwargs=diff_default_trainer_kwargs(),
    )


def _make_count_build(objective_kind: str):
    def build(params: Mapping[str, Any], ctx: RunContext) -> Any:
        p = dict(params)
        p["random_state"] = ctx.seed  # legacy literal RANDOM_STATE, now per-run
        tweedie_power = p.pop("tweedie_variance_power", TWEEDIE_DEFAULT_POWER)
        return build_lstm_count(objective_kind, p, tweedie_power=tweedie_power)

    return build


def _make_diff_build():
    def build(params: Mapping[str, Any], ctx: RunContext) -> Any:
        p = dict(params)
        p["random_state"] = ctx.seed
        p.pop("tweedie_variance_power", None)  # never set in this family
        return build_lstm_from_params(p)

    return build


def _make_count_search_space(model_type: str, objective_kind: str, icl: int, variant: str):
    def search_space(trial: Any) -> dict[str, Any]:
        params, extras = suggest_count_rnn_params(
            trial, objective_kind, input_chunk_length=icl, model_type=model_type
        )
        # `make_nn_objective` injects the trainer kwargs AFTER the suggester,
        # so the pruning callback rides in the params dict (_regression_LSTM.py:1085).
        params["pl_trainer_kwargs"] = count_tuned_trainer_kwargs(trial, variant=variant)
        params.update(extras)
        return params

    return search_space


def _make_diff_search_space(model_type: str, icl: int):
    def search_space(trial: Any) -> dict[str, Any]:
        return suggest_diff_rnn_params(trial, input_chunk_length=icl, model_type=model_type)

    return search_space


def _make_count_from_best(model_type: str, objective_kind: str, icl: int, variant: str):
    """``_regression_LSTM.py::_build_lstm_from_best``: the flat
    ``study.best_params`` back into the shape the suggester produced. ``model``
    and ``input_chunk_length`` come from the VARIANT, not from ``best``."""

    def from_best_params(best: Mapping[str, Any]) -> dict[str, Any]:
        b = dict(best)
        tw_p = b.pop("tweedie_variance_power", TWEEDIE_DEFAULT_POWER)
        params = dict(
            model=model_type,
            input_chunk_length=icl,  # from variant, not best
            output_chunk_length=OUTPUT_CHUNK_LEN,
            hidden_dim=b["hidden_dim"],
            n_rnn_layers=b["n_rnn_layers"],
            hidden_fc_sizes=FC_MAP[b["hidden_fc_sizes"]],
            dropout=b["dropout"],
            batch_size=b["batch_size"],
            n_epochs=100,
            optimizer_kwargs={"lr": b["lr"], "weight_decay": b["weight_decay"]},
            random_state=RANDOM_STATE,
            add_encoders=dict(TUNED_ADD_ENCODERS),
            pl_trainer_kwargs=count_tuned_trainer_kwargs(variant=variant),
        )
        # Carried as a param so `build` can peel it off, exactly where the
        # legacy closure passed `tweedie_power=tw_p` to `build_lstm_count`.
        params["tweedie_variance_power"] = tw_p
        return params

    return from_best_params


def _make_diff_from_best(model_type: str, icl: int):
    """``_diff_regression.py::_build_lstm_from_best``. No tweedie power here."""

    def from_best_params(best: Mapping[str, Any]) -> dict[str, Any]:
        b = dict(best)
        return dict(
            model=model_type,
            input_chunk_length=icl,
            output_chunk_length=OUTPUT_CHUNK_LEN,
            hidden_dim=b["hidden_dim"],
            n_rnn_layers=b["n_rnn_layers"],
            hidden_fc_sizes=FC_MAP[b["hidden_fc_sizes"]],
            dropout=b["dropout"],
            batch_size=b["batch_size"],
            n_epochs=100,
            optimizer_kwargs={"lr": b["lr"], "weight_decay": b["weight_decay"]},
            random_state=RANDOM_STATE,
            add_encoders=dict(TUNED_ADD_ENCODERS),
            pl_trainer_kwargs=diff_tuned_trainer_kwargs(),
        )

    return from_best_params


#: ``OPTUNA_N_TRIALS`` in both scripts, and what every stored
#: ``best_params.json`` records under ``n_trials``.
N_TRIALS = 50

#: The literal the recorded (tuned) runs trained under. Not configurable in the
#: legacy code, so ``ctx.device`` is deliberately not mapped onto it.
TUNED_ACCELERATOR = "auto"

# ``_regression_LSTM.py:1126-1140`` -- the exact lineup that produced
# ``golden/converted/tuning/checkpoints_tune``.
COUNT_VARIANTS: tuple[str, ...] = (
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

# ``_diff_regression.py:1201``. `lstm_w7`, `lstm_w14` and `lstm_w28` are names
# the count family uses too, for a DIFFERENT model (MSE on levels, with the
# `_count_log_link = False` marker, versus MSE on differences with no marker).
# `register` allows one spec per name per experiment family, so both keep their
# legacy names and `get_spec(name, experiment=...)` disambiguates.
DIFF_VARIANTS: tuple[str, ...] = (
    "lstm_w7",
    "lstm_w14",
    "lstm_w28",
    "gru_w7",
    "gru_w14",
    "gru_w28",
)

#: experiment family -> the golden tuning directory its studies were written to.
GOLDEN_TUNING_DIR: dict[str, str] = {
    "count": "checkpoints_tune",
    "diff": "checkpoints_tune_diff",
}


def _register_count() -> list[ModelSpec]:
    specs = []
    for variant in COUNT_VARIANTS:
        model_type, objective_kind, icl = parse_count_variant(variant)
        specs.append(
            register(
                ModelSpec(
                    name=variant,
                    family=model_type.lower(),
                    kind="global",
                    build=_make_count_build(objective_kind),
                    experiments=("count",),
                    defaults=_count_defaults(model_type, icl),
                    search_space=_make_count_search_space(
                        model_type, objective_kind, icl, variant
                    ),
                    from_best_params=_make_count_from_best(
                        model_type, objective_kind, icl, variant
                    ),
                    is_neural=True,
                    stochastic=True,
                    needs_raw_past_covs=True,
                    n_trials=N_TRIALS,
                    device=TUNED_ACCELERATOR,
                    tags=("rnn", "count", objective_kind, f"w{icl}"),
                )
            )
        )
    return specs


def _register_diff() -> list[ModelSpec]:
    specs = []
    for name in DIFF_VARIANTS:
        model_type, icl = parse_diff_variant(name)
        specs.append(
            register(
                ModelSpec(
                    name=name,
                    family=model_type.lower(),
                    kind="global",
                    build=_make_diff_build(),
                    experiments=("diff",),
                    defaults=_diff_defaults(model_type, icl),
                    search_space=_make_diff_search_space(model_type, icl),
                    from_best_params=_make_diff_from_best(model_type, icl),
                    is_neural=True,
                    stochastic=True,
                    needs_raw_past_covs=True,
                    n_trials=N_TRIALS,
                    device=TUNED_ACCELERATOR,
                    tags=("rnn", "diff", "mse", f"w{icl}"),
                )
            )
        )
    return specs


COUNT_SPECS: tuple[ModelSpec, ...] = tuple(_register_count())
DIFF_SPECS: tuple[ModelSpec, ...] = tuple(_register_diff())


#: Preserved-not-fixed observations, numbered from F100 as §4 asks. These are
#: documentation; nothing reads them at run time.
RNN_FLAGS: dict[str, str] = {
    "F100": (
        "The count family's DEFAULT trainer kwargs pin accelerator='gpu', "
        "devices=1, precision='32-true'; the diff family's pin "
        "accelerator='auto' with no devices and no precision. Every TUNED "
        "preset in both families uses accelerator='auto' with no devices key, "
        "so the recorded runs are all 'auto'. None of it is configurable in "
        "the legacy code, so ctx.device is not mapped onto accelerator."
    ),
    "F101": (
        "enable_progress_bar / enable_model_summary are True in the default "
        "presets and False in the tuned presets of both families."
    ),
    "F102": (
        "EarlyStopping monitors 'train_loss' in all four presets, confirming "
        "F10; .fit() never receives a val_series, so there is no validation "
        "loop and 'val_loss' would never be logged. min_delta=1e-4, "
        "mode='min' throughout; patience 10 (default) vs 5 (tuned)."
    ),
    "F103": (
        "The tuning-time trainer kwargs append a pruning callback; the final "
        "rebuild from best params does not. Everything else about the two is "
        "identical (n_epochs=100, patience=5), so the rebuilt model differs "
        "from the trial's model only by that callback."
    ),
    "F104": (
        "gradient_clip_val=1.0 in all four presets. force_reset and "
        "save_checkpoints are never passed, so darts' defaults (False) apply "
        "and no checkpoint is written by the builders."
    ),
    "F105": (
        "Both scripts build ONE module-level EarlyStopping instance and ONE "
        "NN_TRAINER_KWARGS dict, shared by every default build. The tuned "
        "kwargs build a fresh EarlyStopping per call. Reproduced as-is: the "
        "shared instance carries wait_count/stopped_epoch across models."
    ),
    "F106": (
        "The count family's dead default branch uses cyclic FUTURE encoders "
        "while every tuned path (and the diff family's default branch) uses "
        "cyclic PAST encoders. build_regressor('lstm') is never called in "
        "either script, so the divergence never reached a recorded run."
    ),
    "F107": (
        "The diff family's RNN loss is darts' default, i.e. torch.nn.MSELoss "
        "is never passed explicitly: build_lstm_from_params does not set "
        "loss_fn, so 'loss_fn' is absent from model_params. The count "
        "family's MSE variants (lstm_w7/w14/w28) also pass no loss_fn; they "
        "differ from the diff family only by the _count_log_link=False marker."
    ),
    "F108": (
        "build_lstm_from_params sets no _count_log_link attribute at all, "
        "while build_lstm_count always sets it (True or False). The adapters "
        "read it with getattr(..., False), so the diff family behaves like "
        "_count_log_link=False, but the attribute's absence is real."
    ),
    "F109": (
        "hidden_fc_sizes is a STRING categorical ('none'|'32'|'64'|'64_32') "
        "mapped through the same fc_map in the suggester and in the rebuild, "
        "in both families and in _regression_GBDT.py. The stored "
        "best_params.json keeps the string. No drift, but the rebuild does "
        "b['hidden_fc_sizes'] with no default: a best dict that already held "
        "a list would raise KeyError on the lookup."
    ),
    "F110": (
        "random_state=RANDOM_STATE=42 is passed to darts in every RNN params "
        "dict (default branch, suggester and rebuild alike). Both scripts "
        "additionally call torch.manual_seed(42) at module level, so the "
        "recorded runs had a process-wide torch seed as well as darts' "
        "per-model one. build(params, ctx) maps ctx.seed onto random_state; "
        "the process-wide seeding belongs to the runner, not the builder."
    ),
    "F111": (
        "_regression_GBDT.py holds an older copy of the same functions. It "
        "tunes model type and input_chunk_length as Optuna parameters (so no "
        "w7/w14/w28 variants), uses optuna.integration's pruning callback "
        "inside a try/ImportError, adds logger=False to the tuned trainer "
        "kwargs, has no 'mse' branch in build_lstm_count, and runs 25 trials "
        "for NNs instead of 50. None of it produced a golden artefact: the "
        "recorded checkpoints_tune directory matches _regression_LSTM.py."
    ),
    "F112": (
        "The diff family's pruning callback has no on_validation_end hook; "
        "the count family's has one. Dead either way -- no val_series."
    ),
    "F113": (
        "Next to F10: _regression_GBDT.py:316 builds its DEFAULT EarlyStopping "
        "on monitor='val_loss', while _regression_LSTM.py:316 and "
        "_diff_regression.py:273 use 'train_loss'. Only the latter two ran. "
        "'val_loss' is never logged (no val_series), so that callback would "
        "have monitored a metric that does not exist. The 'train_loss' "
        "default is kept; the 'val_loss' copy is recorded, not reproduced."
    ),
    "F114": (
        "_regression_GBDT.py:1085 sets `n_trials = 25 if is_nn else "
        "OPTUNA_N_TRIALS`. Dead: that script's NN branch never ran. All 15 "
        "count studies and all 6 diff studies in golden/converted/tuning "
        "record n_trials = 50, so every spec carries n_trials=50."
    ),
    "F115": (
        "`lstm_w7`, `lstm_w14` and `lstm_w28` name two different models: MSE "
        "on levels with `_count_log_link = False` in the count family, and "
        "MSE on differences with no marker attribute in the diff family. "
        "Their tuned trainer presets agree (accelerator 'auto'); their dead "
        "default presets do not (count 'gpu' + devices=1 + precision, diff "
        "'auto'). Any store keyed on the bare name alone would confuse them."
    ),
}
