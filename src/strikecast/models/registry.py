"""The populated model registry (plan §5.2, §8 P3).

:mod:`strikecast.models.spec` holds the *contract* -- ``ModelSpec``,
``register``, ``get_spec`` -- and nothing else; the specs themselves are
registered as an import side effect of the four spec modules. Importing this
module imports all four exactly once, so anything that asks the registry a
question gets the full lineup instead of whatever the caller happened to
import first::

    from strikecast.models.registry import get_spec, registered_names

    get_spec("lstm_w7", "diff")          # the MSE-on-differences RNN
    registered_names("count")            # the 21 count-family variants

The lineup is the one plan §2.1 inventories, and :data:`EXPECTED_COUNTS` pins
it: 6 GBDT + 15 RNN for ``count``; 3 GBDT + 6 RNN + ``linear`` + ``arima`` +
2 naives for ``diff``; the hurdle stage models plus the composite for
``hurdle``; the damage classifier plus the composite for ``damage``.

Import cost
-----------
Importing this module imports lightgbm, xgboost, catboost, torch and darts,
because the builders reference those classes at module level exactly as the
legacy scripts did. ``import strikecast.models`` therefore does NOT import it:
the package ``__init__`` resolves ``get_spec``, ``registered_names``,
``all_specs`` and the forecaster classes lazily, so importing the package stays
cheap and only a caller that actually asks the registry a question pays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import side effects: each module calls `register()` at import time.
from . import classical, classifiers, gbm, rnn  # noqa: F401
from .spec import (
    ModelKind,
    ModelSpec,
    RunContext,
    all_specs,
    get_spec,
    register,
    registered_experiments,
    registered_names,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "EXPECTED_COUNTS",
    "ModelKind",
    "ModelSpec",
    "RunContext",
    "all_specs",
    "get_spec",
    "register",
    "registered_experiments",
    "registered_names",
    "specs_by_experiment",
]

#: experiment family -> how many specs plan §2.1 puts in it. The unit test
#: checks the registry against this; nothing reads it at run time.
EXPECTED_COUNTS: Mapping[str, int] = {
    "count": len(gbm.COUNT_GBM_VARIANTS) + len(rnn.COUNT_VARIANTS),  # 6 + 15
    "diff": len(gbm.DIFF_GBM_VARIANTS)  # 3
    + len(rnn.DIFF_VARIANTS)  # 6
    + 2  # linear, arima
    + len(classical.NAIVE_NAMES),  # naive_last, naive_weekly
    "hurdle": 3,  # spe_event_classifier, catboost_tweedie_count_head, hurdle
    "damage": 2,  # spe_damage_classifier, damage
}


def specs_by_experiment() -> dict[str, list[ModelSpec]]:
    """``{experiment: [spec, ...]}`` in registration order."""
    out: dict[str, list[ModelSpec]] = {}
    for spec in all_specs():
        for experiment in spec.experiments:
            out.setdefault(experiment, []).append(spec)
    return out
