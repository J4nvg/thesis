"""Evaluation: metrics, aggregation and probability calibration.

Only :mod:`strikecast.evaluation.calibration` is re-exported today; the metric
and aggregation ports land here in the same phase and add their own names to
``__all__``.
"""

from .calibration import (
    CAL_EPS,
    CAL_METHOD,
    CalibrationResult,
    Calibrator,
    apply_calibrators_per_horizon,
    apply_sigmoid_calibrator,
    apply_venn_abers_per_horizon,
    calibrate_per_horizon,
    calibrators_from_json,
    calibrators_to_json,
    collect_venn_abers_data_per_horizon,
    fit_calibrators_per_horizon,
    fit_sigmoid_calibrator,
    oof_calibrated_probs,
    venn_abers_data_from_json,
    venn_abers_data_to_json,
)

__all__ = [
    "CAL_EPS",
    "CAL_METHOD",
    "CalibrationResult",
    "Calibrator",
    "apply_calibrators_per_horizon",
    "apply_sigmoid_calibrator",
    "apply_venn_abers_per_horizon",
    "calibrate_per_horizon",
    "calibrators_from_json",
    "calibrators_to_json",
    "collect_venn_abers_data_per_horizon",
    "fit_calibrators_per_horizon",
    "fit_sigmoid_calibrator",
    "oof_calibrated_probs",
    "venn_abers_data_from_json",
    "venn_abers_data_to_json",
]

# --------------------------------------------------------------------------- #
# Publication analyses (plan sec. 7). Additive: these modules consume saved
# predictions and metrics and never alter a pipeline. Appended so the imports
# above stay owned by whoever ported them.
# --------------------------------------------------------------------------- #
from .comparison import (
    DEFAULT_BLOCK_LENGTH,
    JOIN_KEYS,
    LOSSES,
    DMResult,
    PairSummary,
    block_bootstrap_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    compare_pair,
    diebold_mariano,
    paired_differences,
    pairwise_table,
    per_origin_mean,
    write_pairwise,
)
from .leaderboard import (
    LEADERBOARD_KEYS,
    STAGES,
    MetricRow,
    build_leaderboard,
    build_leaderboard_from_rows,
    collect_metric_rows,
    discover_metric_files,
    golden_metric_rows,
    leaderboard_rows,
    metric_frame,
    write_leaderboard,
)
from .seeds import (
    CI_COLUMNS,
    GROUP_KEYS,
    aggregate_seeds,
    bootstrap_mean_interval,
    broadcast_deterministic,
    leaderboard_ci,
    t_interval,
    write_leaderboard_ci,
)
from .stats import (
    ALPHA,
    FriedmanResult,
    NormalityReport,
    average_rank_table,
    build_block_matrices,
    critical_difference_diagram,
    friedman_nemenyi,
    nemenyi_critical_distance,
    normality_report,
    save_critical_difference_diagram,
    tied_with_best,
)

__all__ += [
    "ALPHA",
    "CI_COLUMNS",
    "DEFAULT_BLOCK_LENGTH",
    "DMResult",
    "FriedmanResult",
    "GROUP_KEYS",
    "JOIN_KEYS",
    "LEADERBOARD_KEYS",
    "LOSSES",
    "MetricRow",
    "NormalityReport",
    "PairSummary",
    "STAGES",
    "aggregate_seeds",
    "average_rank_table",
    "block_bootstrap_ci",
    "bootstrap_mean_interval",
    "broadcast_deterministic",
    "build_block_matrices",
    "build_leaderboard",
    "build_leaderboard_from_rows",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "collect_metric_rows",
    "compare_pair",
    "critical_difference_diagram",
    "diebold_mariano",
    "discover_metric_files",
    "friedman_nemenyi",
    "golden_metric_rows",
    "leaderboard_ci",
    "leaderboard_rows",
    "metric_frame",
    "nemenyi_critical_distance",
    "normality_report",
    "paired_differences",
    "pairwise_table",
    "per_origin_mean",
    "save_critical_difference_diagram",
    "t_interval",
    "tied_with_best",
    "write_leaderboard",
    "write_leaderboard_ci",
    "write_pairwise",
]
