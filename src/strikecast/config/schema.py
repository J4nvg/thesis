"""Pydantic schemas for the data/series stage.

Deliberately minimal: only what `strikecast.data.series` needs today. Later
phases (models, backtest, tuning, tracking) extend this module.

Every default here reproduces the legacy behaviour exactly:

* the six window transforms and the ``WindowTransformer`` keyword arguments come
  from ``src/ts_specific_tools.py::build_ts_and_apply_window_transformer``;
* the 70/10/20 fractions come from ``TRAIN_FRAC, VAL_FRAC, TEST_FRAC`` in
  ``_regression_GBDT.py`` / ``_regression_LSTM.py`` / ``_diff_regression.py``;
* ``expdecay7``'s alpha is ``src/feature_tools.py::halflife_to_alpha(7)``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_EWM_HALFLIFE",
    "SeriesConfig",
    "SplitConfig",
    "WindowTransformConfig",
    "default_window_transforms",
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


def default_window_transforms() -> list[dict[str, Any]]:
    """The six window transforms hard-coded in the legacy pipeline, in order."""
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
        {"function": "mean", "mode": "ewm", "alpha": halflife_to_alpha(DEFAULT_EWM_HALFLIFE),
         "function_name": "expdecay7"},
    ]


class WindowTransformConfig(BaseModel):
    """Configuration for the darts ``WindowTransformer`` on the past covariates."""

    model_config = ConfigDict(extra="forbid")

    transforms: list[dict[str, Any]] = Field(default_factory=default_window_transforms)
    treat_na: Any = 0
    forecasting_safe: bool = True
    keep_non_transformed: bool = True
    include_current: bool = True
    keep_names: bool = False

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
