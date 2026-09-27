"""Pydantic schemas for the data/series stage.

Deliberately minimal: only what `strikecast.data.series` needs today. Later
phases (models, backtest, tuning, tracking) extend this module.

Every default here reproduces the legacy behaviour exactly:

* the six window transforms and the ``WindowTransformer`` keyword arguments come
  from ``src/ts_specific_tools.py::build_ts_and_apply_window_transformer``;
* the 70/10/20 fractions come from ``TRAIN_FRAC, VAL_FRAC, TEST_FRAC`` in
  ``_regression_GBDT.py`` / ``_regression_LSTM.py`` / ``_diff_regression.py``;
* ``expdecay7``'s alpha is ``src/feature_tools.py::halflife_to_alpha(7)``.

**The one documented exception** (audit 2026-09-26, A1/D1) is the sixth window
transform. The thesis describes a 7-day-half-life leaky integrator
``s_t = x_t + 2^(-1/7) s_{t-1}``, but the legacy code passed ``alpha=2^(-1/7)``
to pandas' ``ewm``, where alpha weights *today's* value, so ``expdecay7`` is
almost the raw series. :attr:`WindowTransformConfig.expdecay` switches between
the two; its default is the FIXED filter (``"leaky"``, feature name ``leaky7``),
and everything that reproduces the thesis pins ``"legacy_alpha"`` explicitly
(the golden tests and ``configs/legacy/<experiment>.yaml``).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strikecast.data.feature_selection import FeatureSelectionConfig
    from strikecast.models.spec import RunContext

__all__ = [
    "DEFAULT_EWM_HALFLIFE",
    "DEFAULT_EXPDECAY",
    "EXPDECAY_FUNCTION_NAMES",
    "LEGACY_ADD_ENCODERS",
    "BacktestConfig",
    "CalibrationConfig",
    "CommonKwargsConfig",
    "DataConfig",
    "ExperimentConfig",
    "FeatureSelectionStageConfig",
    "ModelEntry",
    "NaiveScalesConfig",
    "Paradigm",
    "ParadigmConfig",
    "PrunerConfig",
    "SeedConfig",
    "SeriesConfig",
    "SplitConfig",
    "StageConfig",
    "StoreConfig",
    "TrackingConfig",
    "TransformConfig",
    "TuningConfig",
    "WindowTransformConfig",
    "available_threads",
    "default_window_transforms",
    "expdecay_transform",
    "halflife_to_alpha",
]

DEFAULT_EWM_HALFLIFE: float = 7.0


def halflife_to_alpha(halflife_days: float) -> float:
    """Mirror of ``src.feature_tools.halflife_to_alpha``.

    Kept as a local copy so that ``strikecast`` does not import the legacy
    ``src`` package. The formula is identical (``2 ** (-1 / halflife)``) and the
    golden test asserts bit equality with the legacy implementation.
    """
    return 2 ** (-1 / halflife_days)


def available_threads() -> int:
    """Mirror of ``src.prevalent_functions.get_available_threads`` (lines 245-249).

    ``ExperimentConfig.threads = "auto"`` resolves through this, which is what
    the legacy scripts did: every count/hurdle/damage builder was handed
    ``available_threads = get_available_threads()`` at import time. Writing the
    number into the YAML instead would pin one machine's core count into the
    run-store hash, so the *string* is the configured value and the number is
    resolved per run.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return min(cpu_count, 64)


#: The two forms of the sixth window transform (audit 2026-09-26, A1/D1).
#:
#: ``legacy_alpha``
#:     ``{"function": "mean", "mode": "ewm", "alpha": 2**(-1/7)}``, named
#:     ``expdecay7``: bit-identical to ``src/ts_specific_tools.py:46``. pandas'
#:     ``alpha`` weights the CURRENT value, so this is ``s_t = 0.906 x_t +
#:     0.094 s_{t-1}`` (normalised) -- almost the raw series, NOT the thesis'
#:     leaky integrator. Kept only to reproduce the thesis.
#: ``leaky``
#:     ``{"function": "sum", "mode": "ewm", "halflife": 7}``, named ``leaky7``:
#:     exactly the thesis equation ``s_t = x_t + 2^(-1/7) s_{t-1}`` with
#:     ``s_{-1} = 0`` (pandas' ``ewm(...).sum()`` with ``adjust=True`` IS the
#:     unnormalised recursion). The distinct name means a legacy selection can
#:     never silently subset the new columns.
ExpdecayMode = Literal["legacy_alpha", "leaky"]

#: The schema default: the FIXED filter (decision D1). Every thesis
#: reproduction pins ``"legacy_alpha"`` explicitly.
DEFAULT_EXPDECAY: ExpdecayMode = "leaky"

#: ``function_name`` of the sixth transform, per mode.
EXPDECAY_FUNCTION_NAMES: dict[str, str] = {"legacy_alpha": "expdecay7", "leaky": "leaky7"}


def expdecay_transform(mode: str = DEFAULT_EXPDECAY) -> dict[str, Any]:
    """The sixth window transform for one :data:`ExpdecayMode`."""
    if mode == "legacy_alpha":
        return {"function": "mean", "mode": "ewm", "alpha": halflife_to_alpha(DEFAULT_EWM_HALFLIFE),
                "function_name": EXPDECAY_FUNCTION_NAMES["legacy_alpha"]}
    if mode == "leaky":
        return {"function": "sum", "mode": "ewm", "halflife": DEFAULT_EWM_HALFLIFE,
                "function_name": EXPDECAY_FUNCTION_NAMES["leaky"]}
    raise ValueError(f"unknown expdecay mode {mode!r}; expected 'legacy_alpha' or 'leaky'")


def default_window_transforms(expdecay: str = DEFAULT_EXPDECAY) -> list[dict[str, Any]]:
    """The six window transforms of the legacy pipeline, in order.

    The first five are hard-coded in ``src/ts_specific_tools.py``; the sixth is
    :func:`expdecay_transform` of ``expdecay`` (``"legacy_alpha"`` reproduces
    the legacy list bit for bit).
    """
    return [
        {"function": "sum", "mode": "rolling", "window": 14, "min_periods": 1,
         "function_name": "rsum14"},
        {"function": "sum", "mode": "rolling", "window": 7, "min_periods": 1,
         "function_name": "rsum7"},
        {"function": "mean", "mode": "rolling", "window": 7, "min_periods": 1,
         "function_name": "rmean7"},
        {"function": "mean", "mode": "rolling", "window": 28, "min_periods": 1,
         "function_name": "rmean28"},
        {"function": "mean", "mode": "ewm", "span": 14, "function_name": "ewma14"},
        expdecay_transform(expdecay),
    ]


class WindowTransformConfig(BaseModel):
    """Configuration for the darts ``WindowTransformer`` on the past covariates.

    ``expdecay`` selects the sixth transform (:data:`ExpdecayMode`, audit
    2026-09-26 A1/D1). When ``transforms`` is not given it is derived from
    ``expdecay``; a hand-written ``transforms`` list that contradicts
    ``expdecay`` -- the other mode's transform, or this mode's name with other
    parameters -- is rejected. Build a legacy config with
    ``WindowTransformConfig(expdecay="legacy_alpha")``: ``model_copy(update=...)``
    does not re-validate and would leave the old mode's transform behind.
    """

    model_config = ConfigDict(extra="forbid")

    expdecay: ExpdecayMode = DEFAULT_EXPDECAY
    transforms: list[dict[str, Any]] = Field(default_factory=default_window_transforms)
    treat_na: Any = 0
    forecasting_safe: bool = True
    keep_non_transformed: bool = True
    include_current: bool = True
    keep_names: bool = False

    @model_validator(mode="before")
    @classmethod
    def _transforms_follow_expdecay(cls, value: Any) -> Any:
        """Derive ``transforms`` from ``expdecay`` when it is not given."""
        if isinstance(value, dict) and value.get("transforms") is None:
            value = dict(value)
            value["transforms"] = default_window_transforms(
                value.get("expdecay", DEFAULT_EXPDECAY)
            )
        return value

    @model_validator(mode="after")
    def _transforms_agree_with_expdecay(self) -> WindowTransformConfig:
        """Refuse a ``transforms`` list that contradicts :attr:`expdecay`."""
        expected = expdecay_transform(self.expdecay)
        for mode, name in EXPDECAY_FUNCTION_NAMES.items():
            for transform in self.transforms:
                if transform.get("function_name") != name:
                    continue
                if mode != self.expdecay:
                    raise ValueError(
                        f"series.window.expdecay={self.expdecay!r} but `transforms` contains "
                        f"the {mode!r} transform {name!r}; drop `transforms` to derive it "
                        "from `expdecay` (audit A1/D1)"
                    )
                if dict(transform) != expected:
                    raise ValueError(
                        f"series.window.transforms has a {name!r} transform {transform!r} "
                        f"that differs from the {self.expdecay!r} definition {expected!r}"
                    )
        return self

    def transformer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``darts...WindowTransformer(...)``."""
        return {
            "transforms": [dict(t) for t in self.transforms],
            "treat_na": self.treat_na,
            "forecasting_safe": self.forecasting_safe,
            "keep_non_transformed": self.keep_non_transformed,
            "include_current": self.include_current,
            "keep_names": self.keep_names,
        }


class SplitConfig(BaseModel):
    """Chronological 70/10/20 split, exactly as `split_series_list` performs it.

    The legacy code splits with ``split_after(0.7)`` and then ``split_after(1/3)``
    on the remainder; the fractions below are *documentation* of that split plus
    the two derived fractions (`train_val_end`, `cv_start_frac`) that
    ``get_covs_and_encodings`` returns. They are not re-derived from `train`/`val`
    when splitting, so that the legacy rounding behaviour is preserved bit for
    bit.
    """

    model_config = ConfigDict(extra="forbid")

    train: float = 0.70
    val: float = 0.10
    test: float = 0.20

    @property
    def train_val_end(self) -> float:
        """``TRAIN_VAL_END`` in the legacy code: train + val (0.80)."""
        return self.train + self.val

    @property
    def cv_start_frac(self) -> float:
        """``CV_START_VAL`` in the legacy code: train / (train + val) (0.875)."""
        return self.train / self.train_val_end


class SeriesConfig(BaseModel):
    """Everything `strikecast.data.series.build_bundle` needs beyond the panel."""

    model_config = ConfigDict(extra="forbid")

    group_col: str = "region"
    time_col: str = "event_date"
    static_cols: list[str] = Field(default_factory=lambda: ["Activity_Level"])
    window: WindowTransformConfig = Field(default_factory=WindowTransformConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)


class BacktestConfig(BaseModel):
    """The expanding-window loop every legacy runner implements.

    The loop is::

        start_idx = int(start_frac * n_total)
        for t0 in range(start_idx, n_total - horizon + 1, predict_stride):
            retrain = (t0 - start_idx) % retrain_stride == 0

    ``start_frac`` is a *fraction of the model-space reference series*, not a
    date: ``cv_start_frac`` (0.875) for the validation stage and
    ``train_val_end`` (0.7999999999999999, see flag F17) for the test stage.
    The defaults below are the values every in-scope family uses:
    ``horizon = OUTPUT_CHUNK_LEN = 7``, ``predict_stride = CV_STRIDE = 1`` and
    ``retrain_stride = OUTPUT_CHUNK_LEN = 7``.

    ``retrain_stride = None`` means "never retrain" and reproduces
    ``_chronos2.py::chronos2_rolling_long``, whose fixed predictor is fit once
    before the loop and never refit.
    """

    model_config = ConfigDict(extra="forbid")

    start_frac: float
    horizon: int = 7
    predict_stride: int = 1
    retrain_stride: int | None = 7


# =========================================================================== #
# Phase 3: the experiment configuration
#
# Everything below is APPENDED to the Phase-1 schema above and never modifies
# it.  The governing rule of the refactor (docs/REFACTOR_PLAN.md sec. 1,
# "Behaviour") applies to every default in this section: the default is the
# value the thesis actually ran, per family, as recorded in sec. 2.3 and in the
# legacy scripts.  Where the thesis did something questionable, the field
# exists so that it *can* be changed, the alternative is documented next to it,
# and the default stays legacy.
# =========================================================================== #

#: The three training paradigms of the thesis (mirror of
#: ``strikecast.backtest.grouping.Paradigm``, duplicated so that importing the
#: config package does not import darts).
Paradigm = Literal["global", "activity", "local"]

#: ``add_encoders`` as ``src/prevalent_functions.py::get_common_kwargs`` builds
#: it.  Every in-scope family calls that helper with its defaults, so this dict
#: is shared by count, diff, hurdle and damage alike.
LEGACY_ADD_ENCODERS: dict[str, Any] = {
    "cyclic": {"future": ["month", "week", "dayofyear", "dayofweek", "day"]},
}


class DataConfig(BaseModel):
    """Inputs and panel construction (``strikecast.data.load`` + ``.panel``).

    Defaults are the legacy module-level globals::

        DATA_FOLDER     = "./data"
        FIXED_DATA_PATH = construct_path(DATA_FOLDER, "fixed")     -> data/fixed
        DATASET_PATH    = construct_path(DATA_FOLDER, "dataset")   -> data/dataset
        TARGET          = "act_drone_strike_on_ua"

    shared verbatim by ``_regression_GBDT.py``, ``_regression_LSTM.py``,
    ``_diff_regression.py``, ``final_hurdle.ipynb`` cell 1 and
    ``damage_classifier.ipynb`` cell 1.

    ``activity_min_level`` records the tier filter of ``build_panel`` step 4,
    which the legacy code writes as ``Activity_Level != 0``: tier 0 ("little")
    is dropped, so the minimum kept level is 1.  Regions missing from the
    activity map become NaN and survive the ``!= 0`` comparison (flag F24);
    that quirk lives in ``build_panel`` and is not configurable.

    ``low_prevalence_ratio`` is the hard-coded ``ratio=0.1`` of
    ``remove_low_prevalence(..., specific="acled_other_")`` (flag F23/Q6).

    ``binarize`` lists the raw count columns turned into ``<col>_binary``:
    empty for the count and diff families, ``[TARGET]`` for the hurdle event
    classifier, and the four ``act_drone_infra_ua_*_intent`` columns for the
    damage family.

    ``panel_variant`` selects the binarisation position and the interaction
    block, i.e. which of the three legacy copies of the feature engineering is
    reproduced (flag F23):

    ``regressor``
        ``get_engineered_features`` -- interactions on, binarise *before* them.
    ``damage``
        ``get_engineered_features_damageclassifiers`` -- interactions on,
        binarise *after* the GDELT step, one frame per target.
    ``chronos``
        the inline copy in ``_chronos2.py`` -- no interactions, MultiIndex kept.
    """

    model_config = ConfigDict(extra="forbid")

    fixed_dir: str = "data/fixed"
    dataset_dir: str = "data/dataset"
    target: str = "act_drone_strike_on_ua"
    activity_min_level: int = 1
    low_prevalence_ratio: float = 0.1
    binarize: list[str] = Field(default_factory=list)
    panel_variant: Literal["regressor", "damage", "chronos"] = "regressor"

    @property
    def binarize_stage(self) -> Literal["early", "late"]:
        """``build_panel``'s ``binarize_stage`` for this panel variant (F23)."""
        return "late" if self.panel_variant == "damage" else "early"

    @property
    def add_interactions(self) -> bool:
        """``build_panel``'s ``add_interactions``; only Chronos turns it off."""
        return self.panel_variant != "chronos"

    @property
    def keep_index(self) -> bool:
        """``build_panel``'s ``keep_index``; only Chronos keeps the MultiIndex (Q3)."""
        return self.panel_variant == "chronos"

    @property
    def validate_target(self) -> bool:
        """``build_panel``'s ``validate_target`` (Q7).

        Only ``get_engineered_features`` raises when ``target_col`` is missing
        after engineering; the damage and Chronos wrappers do not, and the
        damage variant has no single target column at all
        (``build_panel_legacy_damage`` passes ``target=""``).
        """
        return self.panel_variant == "regressor"


class CommonKwargsConfig(BaseModel):
    """The shared darts forecasting skeleton: ``get_common_kwargs()``.

    Verbatim from ``src/prevalent_functions.py``::

        def get_common_kwargs(input_lags=7, output_chunk_len=7):
            return dict(
                lags                  = input_lags,
                lags_past_covariates  = [-1, -7, -14],
                lags_future_covariates= (2, output_chunk_len),
                output_chunk_length   = output_chunk_len,
                output_chunk_shift    = 0,
                add_encoders          = {"cyclic": {"future": [...]}} ,
            )

    Every in-scope family calls it with its defaults (flag F15: the past lags do
    *not* differ between families; only the archived log family used ``[-1]``).

    ``lags_future_covariates`` is a **2-tuple**, which darts reads as the span
    ``(n_past, n_future)`` -- not as two explicit lags.  A JSON round trip to a
    list would silently build a different model, so the field is typed as a
    tuple, a validator rejects any list whose length is not 2, and
    :meth:`as_kwargs` always hands darts a real ``tuple`` (flag F26).
    """

    model_config = ConfigDict(extra="forbid")

    lags: int = 7
    lags_past_covariates: list[int] = Field(default_factory=lambda: [-1, -7, -14])
    lags_future_covariates: tuple[int, int] = (2, 7)
    output_chunk_length: int = 7
    output_chunk_shift: int = 0
    add_encoders: dict[str, Any] = Field(
        default_factory=lambda: {"cyclic": {"future": list(LEGACY_ADD_ENCODERS["cyclic"]["future"])}}
    )

    @field_validator("lags_future_covariates", mode="before")
    @classmethod
    def _exactly_two_lags(cls, value: Any) -> Any:
        """Reject anything but a 2-element sequence (F26).

        JSON and OmegaConf both hand us a ``list``; that is accepted and
        converted, because the *length* is what carries the meaning.  A list of
        any other length is a wiring error, not a different lag specification,
        and is refused with a message that says why.
        """
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(
                    "lags_future_covariates must be a 2-tuple (n_past, n_future); "
                    f"got {len(value)} entries ({list(value)!r}). darts reads a tuple as a "
                    "span and a list as explicit lags, so any other length would silently "
                    "build a different model (flag F26)."
                )
            return tuple(value)
        raise ValueError(
            f"lags_future_covariates must be a 2-element sequence, got {type(value).__name__}"
        )

    @field_serializer("lags_future_covariates", when_used="always")
    def _dump_tuple(self, value: tuple[int, int]) -> tuple[int, int]:
        """Keep tuple-ness in ``model_dump()``.

        ``model_dump(mode="json")`` still yields a list, because JSON has no
        tuple type; re-validating that list restores the tuple, which is what
        makes the round trip safe.
        """
        return tuple(value)  # type: ignore[return-value]

    def as_kwargs(self) -> dict[str, Any]:
        """The dict ``get_common_kwargs()`` returned, ready to ``**``-splat."""
        return {
            "lags": self.lags,
            "lags_past_covariates": list(self.lags_past_covariates),
            "lags_future_covariates": tuple(self.lags_future_covariates),
            "output_chunk_length": self.output_chunk_length,
            "output_chunk_shift": self.output_chunk_shift,
            "add_encoders": json.loads(json.dumps(self.add_encoders)),
        }


class FeatureSelectionStageConfig(BaseModel):
    """The feature-selection pass, per family (flag F8, golden level G).

    ``selector`` names one of the four legacy configurations reproduced in
    ``strikecast.data.feature_selection``:

    ``countreg``
        ``_regression_GBDT.ipynb`` cell 28: ``build_regressor("lightgbm_tweedie")``
        on the level target, objective ``tweedie``, variance power 1.5,
        ``device_type="gpu"``, ``num_threads`` commented out.  The selected set
        is shared with the RNN variants.
    ``diffreg``
        ``_diff_regression.py``: objective ``regression``, ``device_type="cpu"``,
        ``num_threads=4`` (hard-coded in that script, it does *not* call
        ``get_available_threads()``), fitted on the **level** target while the
        models train on differences (flags F29, Q8).
    ``zipoisson_regressor`` / ``zipoisson_classifier``
        ``final_hurdle.ipynb`` cell 9: library-default LightGBM, Poisson with
        positive-only sample weights for the count head and binary with
        ``is_unbalance=True`` for the event head (flag F27).  The damage family
        reuses ``zipoisson_classifier`` verbatim
        (``damage_classifier.ipynb`` cell 10 ``get_feature_selector_classifier``)
        with ``per_key=True``.

    ``device`` and ``num_threads`` carry the legacy values of the selected
    family and are passed through to the builder; they exist because a GPU
    LightGBM build is not available everywhere (quirk Q9), not because the
    methodology should change.

    ``top_k`` counts *lagged* names, not base features, so ~52 past and ~8
    future base features survive (flag F25).

    ``cache``: the legacy runs cached the whole 9-tuple as
    ``features/<family>_saved_sets.pkl``.  Those cached sets are **data to
    load, not output to reproduce** (flag F16): the upstream column order
    depends on ``PYTHONHASHSEED``, so the legacy selection cannot be
    regenerated.  New runs pin ``PYTHONHASHSEED=0``; see
    :func:`strikecast.seeds.record_env`.

    Publication runs (audit 2026-09-26 A12/A13, decision D2) add two switches,
    both ``False`` by default so the legacy behaviour is unchanged:

    ``deterministic``
        fit the selector on CPU with LightGBM ``deterministic=True``,
        ``force_col_wise=True`` and the fixed :attr:`num_threads` (all
        required: a validator refuses ``device != "cpu"`` or
        ``num_threads=None``), and refuse to compute unless
        ``PYTHONHASHSEED=0``.
    ``require_cached``
        the data stage never computes a selection; it must already be in the
        run store, written once per family/head by ``strikecast featsel
        experiment=<name>``, or come from :attr:`cache_path`. A missing cache
        is a loud error, so no cv/tune/test job can silently re-select.
    """

    model_config = ConfigDict(extra="forbid")

    selector: Literal[
        "countreg", "diffreg", "zipoisson_regressor", "zipoisson_classifier"
    ] = "countreg"
    device: str = "gpu"
    num_threads: int | None = None
    top_k: int = 100
    cache: bool = True
    cache_path: str | None = None
    per_key: bool = False
    deterministic: bool = False
    require_cached: bool = False

    @model_validator(mode="after")
    def _deterministic_needs_cpu_and_threads(self) -> FeatureSelectionStageConfig:
        if self.deterministic:
            if self.device.lower() != "cpu":
                raise ValueError(
                    "feature_selection.deterministic=true needs device=cpu: LightGBM's "
                    f"`deterministic` has no effect on {self.device!r} (audit A12, Q9)"
                )
            if self.num_threads is None:
                raise ValueError(
                    "feature_selection.deterministic=true needs a fixed num_threads "
                    "(null falls back to the OpenMP default of the machine; audit A12)"
                )
        if self.require_cached and not self.cache and not self.cache_path:
            raise ValueError(
                "feature_selection.require_cached=true needs cache=true (or a cache_path): "
                "`strikecast featsel` must be able to write the selection it requires"
            )
        return self

    def build(self) -> FeatureSelectionConfig:
        """The ``FeatureSelectionConfig`` of this family, from the legacy builders."""
        from strikecast.data.feature_selection import (  # noqa: PLC0415  (keeps config light)
            countreg_config,
            diffreg_config,
            zipoisson_classifier_config,
            zipoisson_regressor_config,
        )

        if self.selector == "countreg":
            cfg = countreg_config(device_type=self.device, num_threads=self.num_threads)
        elif self.selector == "diffreg":
            cfg = diffreg_config(
                device_type=self.device,
                num_threads=4 if self.num_threads is None else self.num_threads,
            )
        elif self.selector == "zipoisson_regressor":
            cfg = zipoisson_regressor_config()
        else:
            cfg = zipoisson_classifier_config()
        if cfg.top_k != self.top_k:
            cfg = cfg.model_copy(update={"top_k": self.top_k})
        if self.deterministic:
            # A12/D2: CPU, fixed threads, LightGBM's deterministic mode. Every
            # selector already fixes `random_state=42`; the col-wise histogram
            # is what makes the fit thread-count invariant (audit B5).
            kwargs = dict(cfg.model_kwargs)
            kwargs.update(
                device_type="cpu",
                num_threads=int(self.num_threads),  # type: ignore[arg-type]
                deterministic=True,
            )
            if not kwargs.get("force_row_wise"):
                kwargs["force_col_wise"] = True
            cfg = cfg.model_copy(update={"model_kwargs": kwargs})
        return cfg


class TransformConfig(BaseModel):
    """Target transform, per experiment (``strikecast.transforms``).

    ``identity`` for the count, hurdle and damage families, ``diff`` for the
    differenced branch and the ARIMA baseline that lives inside it (sec. 2.3,
    "Target transform").  The composites assume the identity transform, because
    their ``prob`` channel is a probability rather than a count; pairing them
    with ``diff`` is a wiring error that :class:`ExperimentConfig` rejects
    (flag F71).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["identity", "diff"] = "identity"


class ModelEntry(BaseModel):
    """One line of the experiment's ``models:`` list.

    ``name`` is a registry key from ``strikecast.models.spec``; the authoritative
    spelling is the legacy variant string, which is also what the golden
    artefacts are named after (``golden/results/<family>/*_<variant>_tuned.*``
    and ``golden/converted/tuning/<ckpt>/<variant>/``).

    ``params`` overrides the spec's parameters for this experiment; empty means
    "use the spec's defaults, or the stored ``best_params.json`` when the model
    is tuned".  ``device`` overrides the family device of
    :attr:`ExperimentConfig.device` (sec. 1 "Device policy", flag F9).
    ``fallback`` reproduces flag F5: ``naive_mean`` is what the diff runners
    wrapped ARIMA in, on any exception.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    device: str | None = None
    fallback: Literal["naive_mean"] | None = None
    #: The paradigms this model runs in when a job matrix is built WITHOUT an
    #: explicit ``paradigm=`` selector (``scripts/slurm/make_jobs.py``,
    #: ``submit_all.py``): the thesis-faithful matrix of audit 2026-09-26
    #: C16/D8.  ``None`` = the experiment's own ``paradigms`` (the composed
    #: ``/paradigm`` group, i.e. Global only); ``[]`` = not part of the default
    #: matrix at all (a composite's components, which run inside the composite).
    #: Job-matrix metadata only: no stage reads it, so it never changes what a
    #: run computes, and an explicit selector still runs any paradigm.
    paradigms: list[Paradigm] | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: Any) -> Any:
        """Allow ``- lightgbm_poisson`` as shorthand for ``- {name: ...}``."""
        if isinstance(value, str):
            return {"name": value}
        return value


class ParadigmConfig(BaseModel):
    """One training paradigm.

    Activity and Local reuse the Global-tuned configuration (flag F14), which is
    what the thesis states and what the legacy wrappers do.
    """

    model_config = ConfigDict(extra="forbid")

    name: Paradigm = "global"

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value


class NaiveScalesConfig(BaseModel):
    """Where ``compute_naive_scales`` takes its denominator from (flags F6, F67).

    ``train`` is what every regression family does for both stages
    (``_regression_GBDT.py:719``, ``_diff_regression.py:603``).  ``train_val``
    is the hurdle family's **test** stage only::

        mae_scales_test, rmse_scales_test = compute_naive_scales(
            [tr.append(vl) for tr, vl in zip(train_target_r, val_target_r)],
            region_names, seasonality=7,
        )

    so the hurdle's test MASE/RMSSE have a different denominator from every
    number they are tabled against.  Preserved, per stage; flagged for the
    publication.
    """

    model_config = ConfigDict(extra="forbid")

    fit_on: Literal["train", "train_val"] = "train"
    seasonality: int = 7


class StageConfig(BaseModel):
    """One backtest stage (``cv`` or ``test``) of one experiment.

    ``start`` names the fraction rather than repeating its value, because both
    fractions are *derived* and one of them is a float-rounding artefact:
    ``train_val_end = 0.70 + 0.10 = 0.7999999999999999`` (flag F17) and
    ``cv_start_frac = 0.70 / 0.7999999999999999 = 0.875``.  Re-deriving them
    from :class:`SplitConfig` keeps that rounding exactly as the legacy code has
    it.

    ``retrain_stride`` is always written out, never defaulted: the two copies of
    the CV runner disagree on its default (``stride`` in the count family,
    ``OUTPUT_CHUNK_LEN`` in the diff twin) and every call site passed it
    explicitly, so neither default ever fired (flag F36).

    ``adapters`` records which post-processing preset each paradigm uses at this
    stage (flag F81).  In the count family the Activity and Local paradigms
    route the *validation* stage through ``run_final_test``, so the same model
    on the same folds is post-processed differently per paradigm: Global CV
    takes the 200-sample median (``for_cv``), Activity and Local CV do not
    (``for_test``).  Latent as the thesis ran it, because every in-scope RNN
    builder forces ``likelihood=None`` (flags F55, F57).
    """

    model_config = ConfigDict(extra="forbid")

    start: Literal["cv_start_frac", "train_val_end"]
    horizon: int = 7
    predict_stride: int = 1
    retrain_stride: int | None = 7
    adapters: dict[Paradigm, Literal["for_cv", "for_test", "for_tuning"]] = Field(
        default_factory=lambda: {
            "global": "for_cv",
            "activity": "for_test",
            "local": "for_test",
        }
    )
    naive_scales: NaiveScalesConfig = Field(default_factory=NaiveScalesConfig)

    def start_frac(self, split: SplitConfig) -> float:
        """Resolve :attr:`start` against the split fractions (F17, F19)."""
        return split.cv_start_frac if self.start == "cv_start_frac" else split.train_val_end

    def backtest(self, split: SplitConfig) -> BacktestConfig:
        """The :class:`BacktestConfig` the engine takes for this stage."""
        return BacktestConfig(
            start_frac=self.start_frac(split),
            horizon=self.horizon,
            predict_stride=self.predict_stride,
            retrain_stride=self.retrain_stride,
        )

    def adapter_for(self, paradigm: Paradigm) -> str:
        """The adapter preset of ``paradigm`` at this stage (F81)."""
        try:
            return self.adapters[paradigm]
        except KeyError:
            raise KeyError(
                f"stage has no adapter preset for paradigm {paradigm!r}; "
                f"known: {sorted(self.adapters)}"
            ) from None


class PrunerConfig(BaseModel):
    """``optuna.pruners.MedianPruner(n_warmup_steps=5)``.

    Identical in ``_regression_GBDT.py:1091``, ``_regression_LSTM.py:1157`` and
    ``_diff_regression.py:1221``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["median", "none"] = "median"
    n_warmup_steps: int = 5


class TuningConfig(BaseModel):
    """Optuna, exactly as the three tuning scripts configure it.

    ``objective`` has **no default** on purpose (flag F39): ``_score_fold_preds``
    has a different default metric in each script (``MASE_mean`` in the GBDT
    script, ``RMSSE_mean`` in the diff script) and both Optuna objectives pass
    ``RMSSE_mean`` explicitly, so neither default ever fired.  Every experiment
    YAML writes it out.

    ``n_trials`` is keyed by model kind.  Both entries are **50**, which is what
    every recorded study ran: ``OPTUNA_N_TRIALS = 50`` in all three scripts, and
    all 16 converted studies under ``golden/converted/tuning/`` have 50 trials,
    including the 15 RNN variants.  ``_regression_GBDT.py:1085`` contains
    ``n_trials = 25 if is_nn else OPTUNA_N_TRIALS``, but that script sets
    ``all_variants = GBM_VARIANTS``, so its two-entry ``NN_VARIANTS`` list was
    never tuned and the 25 never fired.  The RNNs were tuned by
    ``_regression_LSTM.py``, which uses ``n_trials = OPTUNA_N_TRIALS`` for
    every variant.  The alternative (25) is documented here and is not the
    default.

    ``sampler`` is ``TPESampler(seed=RANDOM_STATE)``; the seed comes from
    :class:`SeedConfig.tuning_seed`, never from the eval seeds.  Resuming a
    study from SQLite does not restore the sampler RNG state (flag F11).
    """

    model_config = ConfigDict(extra="forbid")

    objective: str
    n_trials: dict[str, int] = Field(default_factory=lambda: {"gbdt": 50, "rnn": 50})
    sampler: Literal["tpe"] = "tpe"
    pruner: PrunerConfig = Field(default_factory=PrunerConfig)
    direction: Literal["minimize", "maximize"] = "minimize"
    timeout_s: int | None = None
    storage: str = "sqlite:///{store_root}/{experiment}/tuning/{model}/optuna.sqlite3"
    load_if_exists: bool = True

    def trials_for(self, kind: str) -> int:
        """Trial budget for a model kind (``"gbdt"`` / ``"rnn"``)."""
        try:
            return self.n_trials[kind]
        except KeyError:
            raise KeyError(
                f"no trial budget for model kind {kind!r}; known: {sorted(self.n_trials)}"
            ) from None

    def storage_url(self, *, store_root: str, experiment: str, model: str) -> str:
        """Fill :attr:`storage`'s placeholders."""
        return self.storage.format(store_root=store_root, experiment=experiment, model=model)


class SeedConfig(BaseModel):
    """Tune once, evaluate over N seeds (sec. 1 "Seeds", sec. 5.4).

    ``tuning_seed`` is the legacy ``RANDOM_STATE = 42`` and drives the Optuna
    sampler under the Global paradigm only.  ``eval_seeds`` repeats the test
    stage; 42 is first so that the first seed reproduces the thesis.
    Deterministic models (``stochastic: false``) run once and are broadcast.
    """

    model_config = ConfigDict(extra="forbid")

    tuning_seed: int = 42
    eval_seeds: list[int] = Field(default_factory=lambda: [42, 1, 2, 3, 4])


class CalibrationConfig(BaseModel):
    """Per-horizon probability calibration (hurdle and damage families).

    ``method`` is ``final_hurdle.ipynb`` cell 22's ``CAL_METHOD = "sigmoid"``
    (Platt scaling on the logit, ``LogisticRegression(C=1e6, solver="lbfgs")``).

    ``threshold`` is the hard-coded 0.5 of ``_classif_metrics_hurdle`` (flag
    F70).  No call site passes another value and no threshold is selected on
    CV, so every F1/Precision/Recall in both notebooks is that operating point.
    Exposed as a field rather than left buried in the metric function; the
    default stays 0.5.

    ``cv_application = "in_sample"`` reproduces flag F69: the saved
    ``cv_calibrated_*.csv`` rows are calibrated by calibrators fitted on those
    same rows.  ``oof_splits``/``oof_random_state`` configure the 5-fold
    ``StratifiedKFold`` path, which the notebooks call in the diagnostic print
    cells only and which feeds no saved table; ``out_of_sample`` would make the
    CV view use it.  The test stage is genuinely out of sample either way,
    because it applies the CV-fitted calibrators.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["sigmoid", "isotonic", "venn_abers"] = "sigmoid"
    group_by: Literal["horizon"] = "horizon"
    threshold: float = 0.5
    cv_application: Literal["in_sample", "out_of_sample"] = "in_sample"
    oof_splits: int = 5
    oof_random_state: int = 42
    venn_abers: bool = False


class TrackingConfig(BaseModel):
    """Weights & Biases mirror of the local run store (sec. 5.5).

    The run store is the source of truth; the tracker never holds state the run
    store does not also hold.  Cluster nodes reach the internet, so runs are
    online by default.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["wandb", "noop"] = "wandb"
    mode: Literal["online", "offline"] = "online"
    project: str = "strikecast"
    entity: str | None = None
    group: str | None = None
    #: Where the backend writes its own run directories.  Only offline runs
    #: need it: on the cluster it points at scratch (``$WANDB_DIR``), and
    #: ``scripts/wandb_sync.sh -d <dir>`` syncs from there afterwards.  ``None``
    #: -- the default -- passes no ``dir`` to ``wandb.init``, which is what the
    #: tracker did before this field existed, so behaviour is unchanged.
    dir: str | None = None
    #: Fail loudly on a tracking error instead of disabling the mirror.  The
    #: default is ``False``: the run store is the source of truth (sec. 5.5) and
    #: a W&B outage must not abort a run whose predictions are on disk.  The
    #: P4 smoke test sets it, because there a broken mirror IS the failure.
    strict: bool = False
    #: Extra W&B tags, appended to the ``[family, kind]`` the model spec
    #: supplies.  Empty by default, so the tag set is unchanged.
    tags: list[str] = Field(default_factory=list)


class StoreConfig(BaseModel):
    """Where the local run store lives (sec. 5.3).  Git-ignored (sec. 10)."""

    model_config = ConfigDict(extra="forbid")

    root: str = "runs"


class ExperimentConfig(BaseModel):
    """One experiment family, fully resolved.

    ``split`` is *not* a field of its own: it lives inside :attr:`series`, which
    is what ``strikecast.data.series.build_bundle`` consumes, and is exposed
    here as a read-only property so there is exactly one place the 70/10/20
    fractions can be set.

    Stage identity in the run store is ``hash(resolved stage config + upstream
    hashes + seed)``; :meth:`resolved_hash` is the "resolved stage config" part.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    data: DataConfig = Field(default_factory=DataConfig)
    series: SeriesConfig = Field(default_factory=SeriesConfig)
    transform: TransformConfig = Field(default_factory=TransformConfig)
    common_kwargs: CommonKwargsConfig = Field(default_factory=CommonKwargsConfig)
    feature_selection: FeatureSelectionStageConfig = Field(
        default_factory=FeatureSelectionStageConfig
    )
    #: Extra, per-head feature selections, for families that ran more than one
    #: (flag F27).  Only the hurdle family does: ``final_hurdle.ipynb`` cell 10
    #: fits ``regressor_feature_selection`` (Poisson, positive-only sample
    #: weights) and ``event_classifier_feature_selection`` (binary,
    #: ``is_unbalance=True``) separately and subsets each head's covariates with
    #: its own top-100 (cell 11).  A head that is not listed falls back to
    #: :attr:`feature_selection`; see :meth:`feature_selection_for`.
    feature_selections: dict[str, FeatureSelectionStageConfig] = Field(default_factory=dict)
    models: list[ModelEntry] = Field(default_factory=list)
    paradigms: list[ParadigmConfig] = Field(default_factory=lambda: [ParadigmConfig()])
    stages: dict[str, StageConfig] = Field(default_factory=dict)
    tuning: TuningConfig | None = None
    seeds: SeedConfig = Field(default_factory=SeedConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    #: Family -> device string, exactly as the thesis ran it (sec. 1, flag F9).
    #: Count GBDTs on CPU; diff-branch XGBoost ``cuda`` and CatBoost ``GPU``
    #: with LightGBM on CPU; hurdle CatBoost CPU.  The strings are the legacy
    #: spellings, because each library spells its own device differently.
    device: dict[str, str] = Field(default_factory=dict)
    #: ``available_threads`` where the legacy builder passed it; ``None`` where
    #: it was commented out, so the library default applies, exactly as before.
    #: ``"auto"`` is the legacy value itself -- the scripts call
    #: ``get_available_threads()`` at import time -- resolved per run by
    #: :func:`available_threads`, so that one machine's core count never leaks
    #: into :meth:`resolved_hash`.
    threads: int | Literal["auto"] | None = None

    # ------------------------------------------------------------------ #
    # derived views
    # ------------------------------------------------------------------ #
    @property
    def split(self) -> SplitConfig:
        """The 70/10/20 split (F19); lives on :attr:`series`."""
        return self.series.split

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    @property
    def paradigm_names(self) -> list[Paradigm]:
        return [p.name for p in self.paradigms]

    def model_entry(self, name: str) -> ModelEntry:
        for entry in self.models:
            if entry.name == name:
                return entry
        raise KeyError(f"model {name!r} is not in experiment {self.name!r}: {self.model_names}")

    def stage(self, name: str) -> StageConfig:
        try:
            return self.stages[name]
        except KeyError:
            raise KeyError(
                f"experiment {self.name!r} has no stage {name!r}; known: {sorted(self.stages)}"
            ) from None

    @property
    def resolved_threads(self) -> int | None:
        """:attr:`threads` with ``"auto"`` resolved to a core count."""
        return available_threads() if self.threads == "auto" else self.threads

    def feature_selection_for(self, head: str | None = None) -> FeatureSelectionStageConfig:
        """The feature selection of one head, falling back to the family's (F27).

        ``head`` is ``None`` for the single-selection families (count, diff,
        damage) and one of :attr:`feature_selections`' keys for the hurdle
        family (``"classifier"`` / ``"regressor"``).
        """
        if head is None:
            return self.feature_selection
        return self.feature_selections.get(head, self.feature_selection)

    def device_for(self, model_name: str) -> str:
        """Device for one model: entry override, then the family map, then cpu.

        The family key is the first ``_``-separated token of the registry name
        (``lightgbm_poisson`` -> ``lightgbm``, ``lstm_w7`` -> ``lstm``), which is
        how the legacy scripts branch in ``build_regressor``.
        """
        entry = self.model_entry(model_name)
        if entry.device is not None:
            return entry.device
        return self.device.get(model_name.split("_", 1)[0], "cpu")

    def to_run_context(self, model_name: str, seed: int | None = None) -> RunContext:
        """The :class:`~strikecast.models.spec.RunContext` a builder takes.

        ``seed`` defaults to :attr:`SeedConfig.tuning_seed` (the legacy
        ``RANDOM_STATE = 42``); the seed sweep passes one of
        :attr:`SeedConfig.eval_seeds` instead.
        """
        from strikecast.models.spec import RunContext  # noqa: PLC0415  (keeps config light)

        return RunContext(
            seed=self.seeds.tuning_seed if seed is None else seed,
            device=self.device_for(model_name),
            threads=self.resolved_threads,
        )

    # ------------------------------------------------------------------ #
    # identity
    # ------------------------------------------------------------------ #
    def canonical_json(self) -> str:
        """Key-sorted, whitespace-free JSON dump; the hash input."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def resolved_hash(self) -> str:
        """sha256 of :meth:`canonical_json`, the run store's stage identity.

        Stable across dict key order (the dump is key-sorted) and across a
        JSON round trip; it changes whenever any field changes.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # wiring checks
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _check_wiring(self) -> ExperimentConfig:
        names = self.model_names
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate model names in experiment {self.name!r}: {duplicates}")

        paradigms = self.paradigm_names
        dupe_paradigms = sorted({p for p in paradigms if paradigms.count(p) > 1})
        if dupe_paradigms:
            raise ValueError(f"duplicate paradigms in experiment {self.name!r}: {dupe_paradigms}")

        for stage_name, stage in self.stages.items():
            missing = [p for p in paradigms if p not in stage.adapters]
            if missing:
                raise ValueError(
                    f"stage {stage_name!r} of experiment {self.name!r} has no adapter preset "
                    f"for paradigm(s) {missing} (flag F81)"
                )

        # F71: the composites assume the identity transform, because their
        # `prob` channel is a probability rather than a count.
        if self.transform.kind != "identity" and self.name in {"hurdle", "damage"}:
            raise ValueError(
                f"experiment {self.name!r} is a composite family and assumes the identity "
                f"target transform; transform.kind={self.transform.kind!r} would be applied to "
                "the probability channel as well (flag F71)"
            )
        return self
