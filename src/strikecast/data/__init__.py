"""Data pipeline: loading, panel construction, covariate split, series bundle,
feature selection and caching.

Every function here reproduces the legacy notebook behaviour exactly; the
quirks that had to be preserved are documented in each module's docstring.
"""

from .cache import cached_parquet, content_hash
from .covariates import DEFAULT_EXCLUDE_COLS, CovariateSplit, split_covariates
from .feature_selection import (
    FeatureSelection,
    FeatureSelectionConfig,
    parse_feature_names,
    rank_features_by_gain,
    select_top_k,
)
from .load import Inputs, load_inputs
from .panel import (
    PanelResult,
    build_panel,
    build_panel_legacy_chronos,
    build_panel_legacy_damage,
    build_panel_legacy_regressor,
)
from .series import (
    RawSeries,
    SeriesBundle,
    build_bundle,
    positive_only_weights,
    subset_components,
)
from .series import load as load_bundle
from .series import serialise as serialise_bundle

__all__ = [
    "DEFAULT_EXCLUDE_COLS",
    "CovariateSplit",
    "FeatureSelection",
    "FeatureSelectionConfig",
    "Inputs",
    "PanelResult",
    "RawSeries",
    "SeriesBundle",
    "build_bundle",
    "build_panel",
    "build_panel_legacy_chronos",
    "build_panel_legacy_damage",
    "build_panel_legacy_regressor",
    "cached_parquet",
    "content_hash",
    "load_bundle",
    "load_inputs",
    "parse_feature_names",
    "positive_only_weights",
    "rank_features_by_gain",
    "select_top_k",
    "serialise_bundle",
    "split_covariates",
    "subset_components",
]
