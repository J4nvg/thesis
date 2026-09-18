"""Verbatim extraction of the RNN builders, losses, trainer kwargs and search
spaces of the count and diff families.

Source
------
``_regression_LSTM.py`` (count family) and ``_diff_regression.py`` (diff
family), at git commit ``b1bfd795e70f004578ee760ffe7c5fd5ccfb3d79``
(branch ``refactor``).

===========================  =====================================
copied object                source lines
===========================  =====================================
``COUNT.PoissonNLLLogLink``  ``_regression_LSTM.py`` 254-306
``COUNT.TweedieNLLLogLink``  (same block)
``COUNT.build_lstm_count``   (same block)
``COUNT.ES_NN``              ``_regression_LSTM.py`` 316-333
``COUNT.NN_TRAINER_KWARGS``  (same block)
``COUNT.build_regressor``    ``_regression_LSTM.py`` 337-345, 475-493, 508
``COUNT.build_nn_from_params``           ``_regression_LSTM.py`` 582-588
``COUNT.LightningPruningCallback``       ``_regression_LSTM.py`` 941-1019
``COUNT._nn_trainer_kwargs`` (same block)
``COUNT._suggest_lstm_params``           (same block)
``COUNT._parse_lstm_variant``            ``_regression_LSTM.py`` 1062-1078
``COUNT._build_lstm_from_best``          ``_regression_LSTM.py`` 1198-1225
``DIFF.ES_NN``               ``_diff_regression.py`` 273-282
``DIFF.NN_TRAINER_KWARGS``   (same block)
``DIFF.build_regressor``     ``_diff_regression.py`` 285-292, 367-386, 402
``DIFF.build_lstm_from_params``          ``_diff_regression.py`` 464-467
``DIFF.LightningPruningCallback``        ``_diff_regression.py`` 1057-1117
``DIFF._nn_trainer_kwargs``  (same block)
``DIFF._suggest_lstm_params``            (same block)
``DIFF._build_lstm_from_best``           ``_diff_regression.py`` 1269-1291
===========================  =====================================

Deviations from the sources (all mechanical, all listed):

1. Imports are hoisted to the top of this module: ``torch``, ``torch.nn as
   nn``, ``torch.nn.functional as F`` (``_regression_LSTM.py`` 251-253),
   ``EarlyStopping`` / ``Callback as PLCallback`` and ``optuna``
   (``_regression_LSTM.py`` 935-936, ``_diff_regression.py`` 1050-1053), and
   ``parse_lstm_variant``, which ``_diff_regression.py`` gets from its
   ``from src import *`` on line 78 and which lives in
   ``src/general_tools.py:24``.
2. ``OUTPUT_CHUNK_LEN`` (7), ``INPUT_LAGS`` (7) and ``RANDOM_STATE`` (42) are
   module-level globals in BOTH scripts, with the same values in both. They
   are redefined here as module constants with the same names and values.
3. The two families define five colliding top-level names
   (``ES_NN``, ``NN_TRAINER_KWARGS``, ``build_regressor``,
   ``LightningPruningCallback``, ``_nn_trainer_kwargs``,
   ``_suggest_lstm_params``, ``_build_lstm_from_best``) with DIFFERENT bodies.
   To keep every identifier and every line byte-identical to its source, each
   family's definitions live inside a factory function that returns a
   ``SimpleNamespace``; the only textual change is four spaces of leading
   indentation on every copied line. Cross-references inside the copied bodies
   (``_nn_trainer_kwargs`` called by ``_suggest_lstm_params``, etc.) therefore
   resolve to the same family, exactly as they did in the source module.
4. ``build_regressor`` keeps only its ``lstm`` branch: the head (signature,
   docstring, ``name = name.lower()``), the ``if name == "lstm":`` block and
   the closing ``raise ValueError``. Every other branch is dropped, because
   the GBDT / ARIMA / naive branches are not this module's subject. Nothing
   inside the ``lstm`` block is touched, including the shared mutable
   ``NN_TRAINER_KWARGS`` dict it closes over.

NOT copied, and why: ``parse_lstm_variant`` (the diff family's variant parser)
is imported from ``src.general_tools`` because it IS importable -- the diff
script calls it through ``from src import *`` and never defines its own.

This module is TEST-ONLY. Nothing under ``src/`` may import it.
"""

# ruff: noqa: E731, F841, N801, N802, N803, N806, E402, B023, ANN001
# The bodies below are evidence, not code: alignment whitespace, lambdas,
# shadowed names and latent bugs are all intentional (tests/legacy_ref/README.md).

from __future__ import annotations

from types import SimpleNamespace

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback as PLCallback
from pytorch_lightning.callbacks import EarlyStopping

from src.general_tools import parse_lstm_variant

# --- module globals of both scripts, identical in both -----------------------
OUTPUT_CHUNK_LEN = 7  # _regression_LSTM.py:97, _diff_regression.py:106
INPUT_LAGS = 7  # _regression_LSTM.py:98, _diff_regression.py:107
RANDOM_STATE = 42  # _regression_LSTM.py:129, _diff_regression.py:133

__all__ = ["COUNT", "DIFF", "INPUT_LAGS", "OUTPUT_CHUNK_LEN", "RANDOM_STATE"]


def _count_family() -> SimpleNamespace:
    """``_regression_LSTM.py`` -- the count family (Poisson / Tweedie / MSE)."""
    # https://github.com/sktime/pytorch-forecasting/blob/main/pytorch_forecasting/metrics/point.py
    class PoissonNLLLogLink(nn.Module):
        def forward(self, y_pred, y_true):
            return F.poisson_nll_loss(
                y_pred, y_true, log_input=True, full=False, reduction="mean" # Darts expects a scalar loss back so reduction = mean
            )


    class TweedieNLLLogLink(nn.Module):
        def __init__(self, power: float = 1.5):
            super().__init__()
            if not (1.0 < power < 2.0):
                raise ValueError("power must lie strictly in (1, 2)")
            self.power = power

        def forward(self, y_pred, y_true):
            p  = self.power
            mu = torch.exp(y_pred)                              
            a  = -y_true * torch.pow(mu, 1.0 - p) / (1.0 - p)
            b  =           torch.pow(mu, 2.0 - p) / (2.0 - p)
            return torch.mean(a + b)

    def build_lstm_count(objective_kind: str, params: dict, *, tweedie_power: float = 1.5):
        """RNN in deterministic point-forecast mode.

        Poisson/Tweedie: log-link NLL loss, predictions inverted via exp() in
        run_expanding_cv (gated on _count_log_link).
        MSE: plain squared-error, predictions in level space — _count_log_link=False.
        """
        from darts.models import BlockRNNModel

        if objective_kind == "mse":
            p = dict(params)
            p.pop("likelihood", None)
            m = BlockRNNModel(**p)
            m._count_log_link = False
            return m

        if objective_kind == "poisson":
            loss_fn = PoissonNLLLogLink()
        elif objective_kind == "tweedie":
            loss_fn = TweedieNLLLogLink(power=tweedie_power)
        else:
            raise ValueError(f"unknown objective_kind {objective_kind!r}")

        p = dict(params)
        p.pop("likelihood", None)        # force deterministic — no sampling
        p["loss_fn"]    = loss_fn
        p["likelihood"] = None

        m = BlockRNNModel(**p)
        m._count_log_link = True         # signal to run_expanding_cv to invert the link
        return m

    ES_NN  = EarlyStopping(monitor="train_loss",   patience=10, min_delta=1e-4, mode="min")
    # ES = EarlyStopping(
    #     monitor="train_loss",   # use "val_loss" if you pass val_series to .fit()
    #     patience=5,
    #     min_delta=1e-4,
    #     mode="min",
    # )

    NN_TRAINER_KWARGS = dict(
        accelerator          = "gpu",
        devices = 1,
        precision = "32-true",
        enable_progress_bar  = True,
        enable_model_summary = True,
        log_every_n_steps    = 10,
        callbacks            = [ES_NN],
        gradient_clip_val    = 1.0,
    )

    def build_regressor(name: str):
        """Return a Darts forecasting model ready to ``.fit()``.

        Every branch uses the same ``INPUT_LAGS`` / ``OUTPUT_CHUNK_LEN`` so that
        a 7-day forecast from yesterday is directly comparable across models.
        These default-config branches are used for the feature-selection pass and
        as a fallback; the Optuna-tuned versions go through ``build_gbm_from_params``.
        """
        name = name.lower()
        if name == "lstm":
            from darts.models import BlockRNNModel
            return BlockRNNModel(
                model                = "LSTM",
                input_chunk_length   = INPUT_LAGS,
                output_chunk_length  = OUTPUT_CHUNK_LEN,
                hidden_dim           = 32,       # ★ 16–128
                n_rnn_layers         = 1,        # ★ 1–3
                dropout              = 0.1,      # ★ 0.0–0.3
                batch_size           = 64,       # · 32–256
                n_epochs             = 30,       # ★ add EarlyStopping
                random_state         = RANDOM_STATE,
                add_encoders = {
                    "cyclic": {
                        "future": ["month", "week", "dayofyear", "dayofweek", "day"]
                               },
                    },
                pl_trainer_kwargs    = NN_TRAINER_KWARGS,
            )
        raise ValueError(f"Unknown regressor name: {name!r}")

    def build_nn_from_params(family: str, params: dict):
        """LSTM only """
        p = dict(params)
        if family == "lstm":
            from darts.models import BlockRNNModel
            return BlockRNNModel(**p)
        raise ValueError(f"Unknown NN family: {family!r}")

    class LightningPruningCallback(PLCallback):
        """Drop-in replacement for optuna.integration.PyTorchLightningPruningCallback.

        Explicitly inherits from pytorch_lightning.callbacks.Callback so PL's
        is_overridden() check can resolve a parent class. Avoids the
        'Expected a parent' ValueError that bites optuna.integration in
        PL >= 2.x environments.
        """
        def __init__(self, trial: optuna.Trial, monitor: str):
            super().__init__()
            self.trial = trial
            self.monitor = monitor

        def _maybe_prune(self, trainer):
            score = trainer.callback_metrics.get(self.monitor)
            if score is None:
                return
            self.trial.report(float(score.detach().cpu()), step=trainer.current_epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial pruned at epoch {trainer.current_epoch} "
                    f"({self.monitor}={float(score):.4f})"
                )

        def on_train_epoch_end(self, trainer, pl_module):
            # Using train_loss because no val_series is passed to .fit()
            self._maybe_prune(trainer)

        def on_validation_end(self, trainer, pl_module):
            # Harmless if val isn't run; helpful if you later add val_series
            self._maybe_prune(trainer)


    def _nn_trainer_kwargs(trial=None, variant: str | None = None):
        callbacks = [EarlyStopping(monitor="train_loss", patience=5,
                                   min_delta=1e-4, mode="min")]

        if trial is not None:
            callbacks.append(LightningPruningCallback(trial, monitor="train_loss"))


        return dict(
            accelerator          = "auto",
            enable_progress_bar  = False,
            enable_model_summary = False,
            log_every_n_steps    = 10,
            gradient_clip_val    = 1.0,
            callbacks            = callbacks,
            # logger               = logger,
        )


    def _suggest_lstm_params(trial, objective_kind: str, input_chunk_length: int, model_type: str = "LSTM"):
        fc_choice = trial.suggest_categorical("hidden_fc_sizes", ["none", "32", "64", "64_32"])
        fc_map = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}

        params = dict(
            model               = model_type,
            input_chunk_length  = input_chunk_length,
            output_chunk_length = OUTPUT_CHUNK_LEN,
            hidden_dim          = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
            n_rnn_layers        = trial.suggest_int("n_rnn_layers", 1, 3),
            hidden_fc_sizes     = fc_map[fc_choice],
            dropout             = trial.suggest_float("dropout", 0.0, 0.4),
            batch_size          = trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            n_epochs            = 100,
            optimizer_kwargs    = {
                "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            },
            random_state        = RANDOM_STATE,
            add_encoders        = {                "cyclic": {
                "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                        }},
        )
        extras = {}
        if objective_kind == "tweedie":
            extras["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
        return params, extras

    def _parse_lstm_variant(variant: str):
        """Parse variant string into (model_type, objective_kind, icl).

        'lstm_poisson_w14' -> ('LSTM', 'poisson', 14)
        'gru_tweedie_w7'   -> ('GRU',  'tweedie', 7)
        'lstm_w7'          -> ('LSTM', 'mse', 7)
        'gru_w14'          -> ('GRU',  'mse', 14)
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

    def _build_lstm_from_best(variant: str, best: dict):
        """Reconstruct LSTM trial params from study.best_params for a re-run.
        `study.best_params` is flat; rebuild the dict shape _suggest_lstm_params
        produces, peel off tweedie_variance_power, and inject window length and
        objective from the variant string (they're not in best_params)."""
        fc_map = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}
        b      = dict(best)
        tw_p   = b.pop("tweedie_variance_power", 1.5)
        model_type, objective_kind, icl = _parse_lstm_variant(variant)

        params = dict(
            model               = model_type,
            input_chunk_length  = icl,                       # ← from variant, not best
            output_chunk_length = OUTPUT_CHUNK_LEN,
            hidden_dim          = b["hidden_dim"],
            n_rnn_layers        = b["n_rnn_layers"],
            hidden_fc_sizes     = fc_map[b["hidden_fc_sizes"]],
            dropout             = b["dropout"],
            batch_size          = b["batch_size"],
            n_epochs            = 100,
            optimizer_kwargs    = {"lr": b["lr"], "weight_decay": b["weight_decay"]},
            random_state        = RANDOM_STATE,
                    add_encoders        = {                "cyclic": {
                        "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                               }},
            pl_trainer_kwargs   = _nn_trainer_kwargs(variant=variant), 
        )
        return lambda: build_lstm_count(objective_kind, params, tweedie_power=tw_p)

    return SimpleNamespace(
        PoissonNLLLogLink=PoissonNLLLogLink,
        TweedieNLLLogLink=TweedieNLLLogLink,
        build_lstm_count=build_lstm_count,
        ES_NN=ES_NN,
        NN_TRAINER_KWARGS=NN_TRAINER_KWARGS,
        build_regressor=build_regressor,
        build_nn_from_params=build_nn_from_params,
        LightningPruningCallback=LightningPruningCallback,
        _nn_trainer_kwargs=_nn_trainer_kwargs,
        _suggest_lstm_params=_suggest_lstm_params,
        _parse_lstm_variant=_parse_lstm_variant,
        _build_lstm_from_best=_build_lstm_from_best,
    )


def _diff_family() -> SimpleNamespace:
    """``_diff_regression.py`` -- the diff family (default MSE loss)."""
    ES_NN = EarlyStopping(monitor="train_loss", patience=10, min_delta=1e-4, mode="min")

    NN_TRAINER_KWARGS = dict(
        accelerator          = "auto",
        enable_progress_bar  = True,
        enable_model_summary = True,
        log_every_n_steps    = 10,
        callbacks            = [ES_NN],
        gradient_clip_val    = 1.0,
    )

    def build_regressor(name: str):
        """Return a Darts forecasting model ready to ``.fit()`` on the DIFFED target.

        All GBDTs use plain regression losses; the LSTM uses the default MSE; ARIMA
        runs on the already-diffed series with d=0; the linear baseline has no
        tuning surface beyond the shared `INPUT_LAGS`.
        """
        name = name.lower()
        # ---------------- LSTM with default MSE loss ---------------------------
        if name == "lstm":
            from darts.models import BlockRNNModel
            return BlockRNNModel(
                model               = "LSTM",
                input_chunk_length  = INPUT_LAGS,
                output_chunk_length = OUTPUT_CHUNK_LEN,
                hidden_dim          = 32,
                n_rnn_layers        = 1,
                dropout             = 0.1,
                batch_size          = 64,
                n_epochs            = 30,
                random_state        = RANDOM_STATE,
                add_encoders = {
                    "cyclic": {
                        "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                               },
                    },
                    pl_trainer_kwargs   = NN_TRAINER_KWARGS,
            )
        raise ValueError(f"Unknown regressor name: {name!r}")

    def build_lstm_from_params(params: dict):
        """LSTM with default MSE loss — no log-link, no count likelihood."""
        from darts.models import BlockRNNModel
        return BlockRNNModel(**params)

    class LightningPruningCallback(PLCallback):
        """Optuna pruning callback that plays nicely with PL >= 2.x."""
        def __init__(self, trial: optuna.Trial, monitor: str):
            super().__init__()
            self.trial = trial
            self.monitor = monitor

        def _maybe_prune(self, trainer):
            score = trainer.callback_metrics.get(self.monitor)
            if score is None:
                return
            self.trial.report(float(score.detach().cpu()), step=trainer.current_epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at epoch {trainer.current_epoch} ({self.monitor}={float(score):.4f})"
                )

        def on_train_epoch_end(self, trainer, pl_module):
            self._maybe_prune(trainer)


    def _nn_trainer_kwargs(trial=None):
        callbacks = [EarlyStopping(monitor="train_loss", patience=5,
                                   min_delta=1e-4, mode="min")]
        if trial is not None:
            callbacks.append(LightningPruningCallback(trial, monitor="train_loss"))
        return dict(
            accelerator          = "auto",
            enable_progress_bar  = False,
            enable_model_summary = False,
            log_every_n_steps    = 10,
            gradient_clip_val    = 1.0,
            callbacks            = callbacks,
        )



    def _suggest_lstm_params(trial, input_chunk_length: int, model_type: str = "LSTM"):
        """RNN (LSTM or GRU) with default MSE loss — model type fixed per variant."""
        fc_choice = trial.suggest_categorical("hidden_fc_sizes", ["none", "32", "64", "64_32"])
        fc_map    = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}
        return dict(
            model               = model_type,
            input_chunk_length  = input_chunk_length,
            output_chunk_length = OUTPUT_CHUNK_LEN,
            hidden_dim          = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
            n_rnn_layers        = trial.suggest_int("n_rnn_layers", 1, 3),
            hidden_fc_sizes     = fc_map[fc_choice],
            dropout             = trial.suggest_float("dropout", 0.0, 0.4),
            batch_size          = trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            n_epochs            = 100,
            optimizer_kwargs    = {
                "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            },
            random_state        = RANDOM_STATE,
                    add_encoders        =                 {"cyclic": {
                        "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                               }},
            pl_trainer_kwargs   = _nn_trainer_kwargs(trial),
        )

    def _build_lstm_from_best(variant: str, best: dict):
        """Reconstruct the RNN build kwargs from study.best_params (flat dict)."""
        fc_map = {"none": [], "32": [32], "64": [64], "64_32": [64, 32]}
        b      = dict(best)
        model_type, icl = parse_lstm_variant(variant)
        params = dict(
            model               = model_type,
            input_chunk_length  = icl,
            output_chunk_length = OUTPUT_CHUNK_LEN,
            hidden_dim          = b["hidden_dim"],
            n_rnn_layers        = b["n_rnn_layers"],
            hidden_fc_sizes     = fc_map[b["hidden_fc_sizes"]],
            dropout             = b["dropout"],
            batch_size          = b["batch_size"],
            n_epochs            = 100,
            optimizer_kwargs    = {"lr": b["lr"], "weight_decay": b["weight_decay"]},
            random_state        = RANDOM_STATE,
                            add_encoders        =                 {"cyclic": {
                        "past": ["month", "week", "dayofyear", "dayofweek", "day"]
                               }},
            pl_trainer_kwargs   = _nn_trainer_kwargs(),
        )
        return lambda: build_lstm_from_params(params)

    return SimpleNamespace(
        ES_NN=ES_NN,
        NN_TRAINER_KWARGS=NN_TRAINER_KWARGS,
        build_regressor=build_regressor,
        build_lstm_from_params=build_lstm_from_params,
        LightningPruningCallback=LightningPruningCallback,
        _nn_trainer_kwargs=_nn_trainer_kwargs,
        _suggest_lstm_params=_suggest_lstm_params,
        _build_lstm_from_best=_build_lstm_from_best,
    )


COUNT = _count_family()
DIFF = _diff_family()
