"""Stage 3b: the hurdle and damage families, end to end (plan §8 P5).

:mod:`strikecast.pipeline.run_stage` drives one darts model that returns one
output channel. The two composite families do not fit that shape:

* the **hurdle** runs two models over two panels and returns three channels
  (``prob``, ``count``, ``hurdle``), whose ``y_true`` comes from two DIFFERENT
  target lists -- the binary events for ``prob``, the counts for the other two;
* the **damage** family runs one classifier per infrastructure key over one
  panel per key and returns one channel per key.

Both also have a post-processing step no regression family has: a per-horizon
sigmoid calibration fitted on the CV classifier rows and applied in sample to
the CV view and out of sample to the test view (F69), and both write several
metric *components* per stage rather than one.

What one call to :func:`run_composite_stage` does
-------------------------------------------------
1. resolve the stage identity from the resolved stage config, **every head's**
   upstream hashes and the seed, and skip a stage already complete under it;
2. seed, snapshot ``config.yaml`` / ``env.json``, start the tracker;
3. assemble the stage's series with :mod:`strikecast.models.hurdle` and build
   the forecaster from the registry spec's builders;
4. run :class:`~strikecast.backtest.engine.ExpandingWindowBacktest` per group of
   the paradigm's partition, streaming parts to disk through a per-channel
   persist hook;
5. reload, calibrate, evaluate every component, write the metric views;
6. for the hurdle's global test stage, fit the two final models on train+val and
   write their native feature importances (``final_hurdle.ipynb`` cells E-F).

Paradigms
---------
``final_hurdle.ipynb`` runs global (cell 15), activity (cell 29) and local
(cell 34); ``damage_classifier.ipynb`` has no per-activity or per-region wrapper
and is global only, which is what ``configs/experiment/damage.yaml`` pins. A
damage run under another paradigm is refused rather than invented.

Metric components
-----------------
``store.write_metrics`` writes one ``global.json`` and one CSV per view. The
PRIMARY component therefore keeps the plain names -- ``global``, ``per_region``,
``per_horizon``, ``per_region_horizon`` -- so a leaderboard walking the run
store finds the family's headline numbers exactly where it finds every other
family's. Every other component is written as ``<component>@<view>.csv`` plus
``<component>@global.json``; the names mirror ``final_hurdle.ipynb`` cells 48-49
(``classifier_raw``, ``classifier_cal``, ``regressor``, ``hurdle_raw``,
``hurdle_cal``), which is also what ``golden/results/finalhurdle/`` stores.
Every component -- the primary included -- gets its ``@`` files, so a reader can
enumerate components uniformly; ``global.json`` and ``<primary>@global.json``
therefore hold the same row on purpose.

The primary component is ``hurdle_cal`` for the hurdle family -- the
Beta-calibrated hurdle that the thesis reports (Table 4, ``main.tex:1004-1006``;
audit 2026-09-26 B8, decision D6). It used to be ``hurdle_raw``, the notebook's
``cv_results`` / ``test_results``; that row is still written as
``hurdle_raw@<view>``. The damage family keeps ``<first key>_raw`` (not
reported in the thesis).

Calibrators across seeds
------------------------
CV runs only under ``seeds.tuning_seed`` (§5.4), so the calibrators live in the
tuning-seed run's ``artifacts/``. A test stage at ANY seed reads them from
``seed=<tuning_seed>`` of the same ``(experiment, model, paradigm)``
(:func:`calibration_key`), and a missing calibrator file or channel is a loud
:class:`CalibratorsMissing` raised before any fold runs -- never a silent
fallback to the raw probabilities (audit C18). The calibrators' sha256 is part
of the test stage's identity.

Activity views
--------------
Neither legacy aggregator (``evaluate_hurdle_long``, ``evaluate_classif_long``)
has a ``regions_activity`` parameter, so **no** stage of these two families
writes ``per_activity_level`` / ``per_activity_horizon``, not even under
``paradigm=activity``. Four views per component, never six. Preserved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from strikecast.backtest.engine import ExpandingWindowBacktest
from strikecast.backtest.grouping import partition, restore_order
from strikecast.backtest.hooks import ProgressHook
from strikecast.backtest.predictions import COLUMNS, PredictionSet
from strikecast.backtest.protocols import FoldResult
from strikecast.backtest.schedule import schedule_from_config
from strikecast.evaluation.calibration import (
    CAL_METHOD,
    apply_calibrators_per_horizon,
    calibrators_from_json,
    calibrators_to_json,
    fit_calibrators_per_horizon,
    oof_calibrated_probs,
)
from strikecast.models.hurdle import (
    damage_label,
    damage_series,
    hurdle_series,
    make_damage_forecaster,
    make_hurdle_forecaster,
    min_positive_samples_for,
)
from strikecast.pipeline import data_stage as _data_stage
from strikecast.pipeline.context import (
    get_spec,
    make_run_context,
    make_tracker,
    resolve_store_root,
    track,
    tracker_tags,
)
from strikecast.pipeline.run_stage import StageOutcome, resolve_params
from strikecast.seeds import record_env, seed_everything
from strikecast.store import RunKey, RunStore, stage_hash
from strikecast.transforms.identity import Identity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from darts import TimeSeries

    from strikecast.config.schema import ExperimentConfig, StageConfig
    from strikecast.pipeline.data_stage import CompositeData

__all__ = [
    "CALIBRATORS_ARTIFACT",
    "CalibratorsMissing",
    "Component",
    "calibration_key",
    "run_composite_experiment",
    "run_composite_stage",
]

logger = logging.getLogger(__name__)

#: Where the per-horizon calibrators live, under ``<run>/artifacts/``. The
#: artifacts directory belongs to the RUN KEY, not to the stage. CV runs only
#: under the tuning seed, so every test stage -- whatever its seed -- reads the
#: calibrators of ``(experiment, model, paradigm, seed=<tuning_seed>)``
#: (:func:`calibration_key`, audit C18), which is the notebook's "apply the
#: CV-fitted calibrators to the test predictions" (cell 27).
CALIBRATORS_ARTIFACT = "calibrators.json"


class CalibratorsMissing(RuntimeError):  # noqa: N818 - reads better at the raise site
    """A calibrated test stage found no CV-fitted calibrators (audit C18)."""

#: The hurdle family's leaderboard row (plain ``global.json``): the calibrated
#: hurdle the thesis reports (audit B8, decision D6).
HURDLE_PRIMARY_COMPONENT = "hurdle_cal"

#: The OOF diagnostic of ``final_hurdle.ipynb`` cell 21 /
#: ``damage_classifier.ipynb`` cell 23: printed there, never saved. Written here
#: so the print can be reproduced without re-running anything.
CALIBRATION_DIAGNOSTIC_ARTIFACT = "calibration_diagnostic.json"

#: ``TOP_N_FEATURES`` / ``PERM_N_REPEATS`` of ``final_hurdle.ipynb`` cell 47.
LEGACY_PERM_N_REPEATS = 10


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Component:
    """One named population of rows and the metric set it is scored with."""

    name: str
    frame: pd.DataFrame
    metric_set: str


def _as_prob_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A classifier frame in legacy spelling: ``y_pred`` renamed to ``y_prob``.

    ``final_hurdle.ipynb`` cell 16 does exactly this rename right after
    ``collect_predictions_long``, and every calibration helper reads
    ``long_df["y_prob"]``.
    """
    return frame.rename(columns={"y_pred": "y_prob"}).reset_index(drop=True)


def _aligned(left: pd.DataFrame, right: pd.DataFrame, what: str) -> None:
    """Both frames must describe the same rows, in the same order.

    The notebook multiplies ``long_df_c_cal["y_prob"].values`` by
    ``long_df_r["y_pred"].values`` POSITIONALLY (cell 22). That is only correct
    while the two collectors emitted the same rows in the same order, which they
    did. Here the channels come out of one ``load_predictions(legacy_order=True)``
    sort, so the same holds -- and is checked rather than assumed.
    """
    keys = ["region", "fold", "horizon"]
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        raise ValueError(
            f"{what}: the two channels are not row-aligned "
            f"({len(left)} vs {len(right)} rows); the positional product of "
            "`final_hurdle.ipynb` cell 22 would be wrong"
        )


# --------------------------------------------------------------------------- #
# per-channel persistence
# --------------------------------------------------------------------------- #
class ChannelPersistHook:
    """:class:`~strikecast.store.run_store.PersistHook` with per-channel actuals.

    The store's own hook attaches ONE actuals list to every channel, which is
    right for a single-output model and wrong for the hurdle: its ``prob``
    channel is scored against the binary events and its ``count`` / ``hurdle``
    channels against the counts (``final_hurdle.ipynb`` cell 16). Everything
    else -- one part per retrain window, the atomic write, the fold-index shift
    -- is the same behaviour, so the on-disk layout is identical.
    """

    def __init__(
        self,
        store: RunStore,
        run: RunKey,
        stage: str,
        retrain_stride: int | None,
        *,
        actuals_by_channel: Mapping[str, Sequence[TimeSeries]],
        region_names: Sequence[str],
    ) -> None:
        for channel, actuals in actuals_by_channel.items():
            if len(actuals) != len(region_names):
                raise ValueError(
                    f"channel {channel!r}: {len(actuals)} actual series for "
                    f"{len(region_names)} region names"
                )
        self.store = store
        self.run = run
        self.stage = stage
        self.retrain_stride = retrain_stride
        self.actuals_by_channel = {k: list(v) for k, v in actuals_by_channel.items()}
        self.region_names = list(region_names)
        self.writer = store.part_writer(run, stage)
        self.parts: list[Path] = []
        self._buffer: list[FoldResult] = []

    def on_fold(
        self, result: FoldResult, cumulative: dict[str, list[list[TimeSeries]]]
    ) -> None:
        del cumulative  # the hook never retains the engine's state (F61)
        if result.fold.retrain and self._buffer:
            self._flush(next_retrain_fold=result.fold.index)
        self._buffer.append(result)

    def close(self) -> Path | None:
        if not self._buffer:
            return None
        return self._flush(next_retrain_fold=None)

    def _flush(self, *, next_retrain_fold: int | None) -> Path:
        buffered = self._buffer
        self._buffer = []
        fold_from = buffered[0].fold.index
        fold_to = buffered[-1].fold.index

        channels = list(buffered[0].predictions)
        per_channel: dict[str, list[list[TimeSeries]]] = {
            channel: [[] for _ in self.region_names] for channel in channels
        }
        for res in buffered:
            for channel, region_preds in res.predictions.items():
                for r_idx, pred in enumerate(region_preds):
                    per_channel[channel][r_idx].append(pred)

        predictions = PredictionSet.concat(
            PredictionSet.from_fold_preds(
                self._actuals(channel), {channel: per_channel[channel]}, self.region_names
            )
            for channel in channels
        )
        frame = predictions.frame
        if fold_from and len(frame):
            frame = frame.copy()
            frame["fold"] = frame["fold"] + int(fold_from)
            predictions = PredictionSet(frame.loc[:, list(COLUMNS)])

        path = self.writer.write(
            predictions, fold_from, fold_to, next_retrain_fold=next_retrain_fold
        )
        self.parts.append(path)
        return path

    def _actuals(self, channel: str) -> list[TimeSeries]:
        try:
            return self.actuals_by_channel[channel]
        except KeyError:
            raise KeyError(
                f"no actuals for channel {channel!r}; known: "
                f"{sorted(self.actuals_by_channel)}"
            ) from None


def _replay(hook: ChannelPersistHook, folds: Sequence[Any], fold_preds: dict) -> None:
    """Feed an already-computed grouped result through the hook.

    A grouped paradigm runs one engine per group, so no single engine sees the
    whole region list. Replaying the restored-order result keeps the part layout
    identical to the streamed case -- the same trick
    :func:`strikecast.pipeline.run_stage._persist_fold_preds` uses.
    """
    channels = list(fold_preds)
    for i, fold in enumerate(folds):
        predictions = {
            channel: [per_region[i] for per_region in fold_preds[channel]]
            for channel in channels
        }
        hook.on_fold(FoldResult(fold=fold, predictions=predictions), {})
    hook.close()


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _calibrator_path(store: RunStore, key: RunKey) -> Path:
    return store.artifacts_dir(key) / CALIBRATORS_ARTIFACT


def calibration_key(cfg: ExperimentConfig, key: RunKey) -> RunKey:
    """The run whose CV stage fitted ``key``'s calibrators: same experiment,
    model and paradigm, ``seed = seeds.tuning_seed`` (CV is single-seed, §5.4)."""
    return RunKey(key.experiment, key.model, key.paradigm, int(cfg.seeds.tuning_seed))


def _require_calibrators(
    cfg: ExperimentConfig, store: RunStore, key: RunKey, stage_name: str
) -> str | None:
    """sha256 of the calibrators a calibrated non-CV stage will apply.

    ``None`` when the stage fits them itself (``cv``) or calibration is off.
    Raises :class:`CalibratorsMissing` -- BEFORE any fold runs -- when the
    tuning-seed CV stage has not written them.
    """
    if not cfg.calibration.enabled or stage_name == "cv":
        return None
    source = calibration_key(cfg, key)
    path = _calibrator_path(store, source)
    if not path.is_file():
        raise CalibratorsMissing(
            f"{key.relative()}/{stage_name}: no {CALIBRATORS_ARTIFACT} at {path}. The "
            f"calibrators are fitted by the CV stage of {source.relative()} (seed = "
            "seeds.tuning_seed); run it first. A calibrated test stage never falls back "
            "to the raw probabilities (audit C18)."
        )
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_calibrators(store: RunStore, key: RunKey) -> dict[str, dict[int, Any]] | None:
    """The CV stage's calibrators, keyed by channel; ``None`` when absent."""
    path = _calibrator_path(store, key)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {channel: calibrators_from_json(block) for channel, block in payload.items()}


def _calibrate(
    store: RunStore,
    key: RunKey,
    stage_name: str,
    prob_frames: Mapping[str, pd.DataFrame],
    cfg: ExperimentConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Calibrated probabilities per classifier channel, plus what to persist.

    * ``cv`` fits one calibrator per horizon on all CV rows of that channel and
      applies them **in sample** -- F69, ``final_hurdle.ipynb`` cell 22 and
      ``damage_classifier.ipynb`` cell 24 -- and writes them to
      ``artifacts/calibrators.json``;
    * ``test`` loads that file from the TUNING-SEED run
      (:func:`calibration_key`) and applies it, which is cell 27 / cell 29.

    A missing file or channel raises :class:`CalibratorsMissing` (audit C18):
    silently keeping the raw probabilities would report an uncalibrated model
    under the calibrated name, and fitting on the test rows is the one thing
    the notebooks were careful not to do.
    """
    method = cfg.calibration.method if cfg.calibration.enabled else CAL_METHOD
    extras: dict[str, Any] = {}

    if not cfg.calibration.enabled:
        return {c: df["y_prob"].to_numpy(dtype=float) for c, df in prob_frames.items()}, extras

    if stage_name == "cv":
        calibrators = {
            channel: fit_calibrators_per_horizon(frame, method=method)
            for channel, frame in prob_frames.items()
        }
        extras["calibrators_path"] = store.write_artifact(
            key,
            CALIBRATORS_ARTIFACT,
            {c: calibrators_to_json(v) for c, v in calibrators.items()},
        )
        extras["diagnostic"] = _oof_diagnostic(prob_frames, cfg)
    else:
        source = calibration_key(cfg, key)
        loaded = _load_calibrators(store, source)
        if loaded is None:
            raise CalibratorsMissing(
                f"{key.relative()}/{stage_name}: no {CALIBRATORS_ARTIFACT} under "
                f"{source.relative()}/artifacts/ -- run the cv stage of the tuning seed "
                "first (audit C18)"
            )
        calibrators = loaded
        extras["calibrators_source"] = str(source.relative())

    probs = {}
    for channel, frame in prob_frames.items():
        per_horizon = calibrators.get(channel)
        if per_horizon is None:
            raise CalibratorsMissing(
                f"{key.relative()}/{stage_name}: the calibrators hold no channel "
                f"{channel!r} (have {sorted(calibrators)}); refit them with the cv stage"
            )
        probs[channel] = apply_calibrators_per_horizon(frame, per_horizon, method=method)
    return probs, extras


def _oof_diagnostic(
    prob_frames: Mapping[str, pd.DataFrame], cfg: ExperimentConfig
) -> dict[str, Any]:
    """``final_hurdle.ipynb`` cell 21: Brier / ROC-AUC / PR-AUC before vs after.

    The 5-fold shuffled ``StratifiedKFold`` mixes rows across regions, folds and
    dates, so this is a calibration diagnostic and NOT an out-of-fold backtest
    (F69, F116). The notebooks print it and save nothing; it is written here as
    an artifact so the print survives.
    """
    from strikecast.evaluation.metrics import classification_metrics  # noqa: PLC0415

    out: dict[str, Any] = {}
    for channel, frame in prob_frames.items():
        y_true, raw, cal = oof_calibrated_probs(
            frame,
            method=cfg.calibration.method,
            n_splits=cfg.calibration.oof_splits,
            random_state=cfg.calibration.oof_random_state,
            group_col=cfg.calibration.group_by,
        )
        out[channel] = {
            "raw": classification_metrics(y_true, raw, cfg.calibration.threshold),
            "oof_calibrated": classification_metrics(y_true, cal, cfg.calibration.threshold),
        }
    return out


# --------------------------------------------------------------------------- #
# component assembly
# --------------------------------------------------------------------------- #
def _hurdle_components(
    preds: PredictionSet, calibrated: Mapping[str, np.ndarray]
) -> tuple[list[Component], dict[str, pd.DataFrame]]:
    """The five components of ``final_hurdle.ipynb`` cells 48-49."""
    prob = _as_prob_frame(preds.for_channel("prob").frame)
    count = preds.for_channel("count").frame.reset_index(drop=True)
    hurdle = preds.for_channel("hurdle").frame.reset_index(drop=True)
    _aligned(prob, count, "prob vs count")
    _aligned(hurdle, count, "hurdle vs count")

    prob_cal = prob.copy()
    prob_cal["y_prob"] = calibrated["prob"]

    hurdle_cal = hurdle.copy()
    hurdle_cal["y_pred"] = prob_cal["y_prob"].to_numpy() * count["y_pred"].to_numpy()

    # F65: the count head is scored on POSITIVE-EVENT DAYS ONLY. The same rows
    # scored unfiltered land in the notebook's `regressor_per_*.csv`, which is a
    # different file and a different number; only the filtered population is a
    # `_cv_res` / `_test_res` component, so only that one is written here.
    count_positive = count[count["y_true"] > 0].copy().reset_index(drop=True)

    components = [
        Component("classifier_raw", prob, "classification"),
        Component("classifier_cal", prob_cal, "classification"),
        Component("regressor", count_positive, "hurdle"),
        Component("hurdle_raw", hurdle, "hurdle"),
        Component("hurdle_cal", hurdle_cal, "hurdle"),
    ]
    return components, {"prob": prob}


def _damage_components(
    preds: PredictionSet, calibrated: Mapping[str, np.ndarray], keys: Sequence[str]
) -> tuple[list[Component], dict[str, pd.DataFrame]]:
    """Raw and sigmoid-calibrated classifier metrics per damage key.

    ``damage_classifier.ipynb`` cells 19/24 (CV) and 28/29 (test) evaluate
    exactly these two populations per key; the component names carry the short
    label the notebook prints (``health``, ``education``, ...) with the full key
    kept as the channel.
    """
    components: list[Component] = []
    prob_frames: dict[str, pd.DataFrame] = {}
    for key in keys:
        raw = _as_prob_frame(preds.for_channel(key).frame)
        prob_frames[key] = raw
        cal = raw.copy()
        cal["y_prob"] = calibrated[key]
        label = damage_label(key)
        components.append(Component(f"{label}_raw", raw, "classification"))
        components.append(Component(f"{label}_cal", cal, "classification"))
    return components, prob_frames


# --------------------------------------------------------------------------- #
# metric views
# --------------------------------------------------------------------------- #
def _evaluate_components(
    components: Sequence[Component],
    primary: str,
    mae_scales: Mapping[str, float],
    rmse_scales: Mapping[str, float],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """``(views for write_metrics, component -> global row)``.

    ``activity_by_region`` is deliberately never passed: neither
    ``evaluate_hurdle_long`` nor ``evaluate_classif_long`` has that parameter,
    so these families write four views per component under every paradigm.
    """
    from strikecast.evaluation.aggregate import evaluate  # noqa: PLC0415

    views: dict[str, Any] = {}
    globals_by_component: dict[str, dict[str, Any]] = {}
    for component in components:
        result = evaluate(
            component.frame,
            mae_scales,
            rmse_scales,
            activity_by_region=None,
            metric_set=component.metric_set,  # type: ignore[arg-type]
            threshold=threshold,
        )
        globals_by_component[component.name] = dict(result["global"])
        for view in ("per_region", "per_horizon", "per_region_horizon"):
            views[f"{component.name}@{view}"] = result[view]
            if component.name == primary:
                # the primary ALSO keeps the plain name, so a leaderboard
                # walking the store finds this family where it finds every other
                views[view] = result[view]
        if component.name == primary:
            views["global"] = result["global"]
    return views, globals_by_component


def _write_component_globals(
    store: RunStore, key: RunKey, stage: str, globals_by_component: Mapping[str, dict]
) -> dict[str, Path]:
    """One ``<component>@global.json`` per non-primary component.

    ``RunStore.write_metrics`` only treats the literal view name ``global`` as
    JSON (everything else must be a frame), so the secondary components' global
    rows are written here, next to their CSVs and in the same shape as the
    notebook's ``global_<key>.json``. ``RunStore.read_metrics`` ignores them,
    which is why they carry the ``@`` marker rather than looking like a view.
    """
    directory = store.metrics_dir(key, stage)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, row in globals_by_component.items():
        path = directory / f"{name}@global.json"
        path.write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
        written[f"{name}@global"] = path
    return written


def _naive_scales(
    bundle: Any, stage_cfg: StageConfig
) -> tuple[dict[str, float], dict[str, float]]:
    """F6 / F67: train for the CV stage, train+val for the hurdle's test stage.

    The scales always come from the PRIMARY head -- the count bundle for the
    hurdle (``compute_naive_scales(train_target_r, ...)``, cell 17). The damage
    family has no scaled metric, so its scales are never read.
    """
    from strikecast.evaluation.metrics import compute_naive_scales  # noqa: PLC0415

    if stage_cfg.naive_scales.fit_on == "train_val":
        targets = [
            tr.append(vl)
            for tr, vl in zip(bundle.target_train, bundle.target_val, strict=True)
        ]
    else:
        targets = list(bundle.target_train)
    return compute_naive_scales(
        targets, bundle.region_names, seasonality=stage_cfg.naive_scales.seasonality
    )


# --------------------------------------------------------------------------- #
# feature importance (final_hurdle.ipynb cells E-G)
# --------------------------------------------------------------------------- #
def _importance_frame(model: Any, feature_names: Sequence[str]) -> pd.DataFrame | None:
    """``src/evaluation_tools.py::feature_importances_per_horizon``, verbatim.

    One column per horizon estimator, ``feature_importances_`` first and
    ``get_feature_importance()`` as the CatBoost fallback; ``None`` when the
    model exposes neither, which is the legacy contract.
    """
    try:
        estimators = model.model.estimators_
    except AttributeError:
        return None
    imps: dict[str, Any] = {"Feature": list(feature_names)}
    for h, est in enumerate(estimators, start=1):
        try:
            imps[f"h{h}_importance"] = est.feature_importances_
        except AttributeError:
            try:
                imps[f"h{h}_importance"] = est.get_feature_importance()
            except Exception:  # noqa: BLE001 - the legacy `except Exception: return None`
                return None
    return pd.DataFrame(imps)


def _with_mean_importance(frame: pd.DataFrame) -> pd.DataFrame:
    """Cell F's ``mean_importance`` column and descending sort."""
    cols = [c for c in frame.columns if c.startswith("h") and c.endswith("_importance")]
    out = frame.copy()
    out["mean_importance"] = out[cols].mean(axis=1)
    return out.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def _hurdle_importances(
    builders: Any, series: Any, train_val_end: float
) -> dict[str, pd.DataFrame]:
    """Cells E and F: one clean fit on train+val, then native importances.

    This is NOT the rolling backtest: the notebook fits both heads once on
    everything up to the test start and reads the importances off that single
    model, "representing the final model's learned feature weights".
    """
    classifier_builder, regressor_builder = tuple(builders)
    reference = series.binary_targets[0]
    cutoff = reference.time_index[int(train_val_end * len(reference))]
    logger.info("feature importance: fitting the final heads on data up to %s", cutoff.date())

    clf = classifier_builder()
    clf.fit(
        series=[ts.drop_after(cutoff) for ts in series.binary_targets],
        past_covariates=series.clf_past,
        future_covariates=series.clf_future,
    )
    reg = regressor_builder()
    reg.fit(
        series=[ts.drop_after(cutoff) for ts in series.count_targets],
        past_covariates=series.reg_past,
        future_covariates=series.reg_future,
        sample_weight=[w.drop_after(cutoff) for w in series.weights],
    )

    out: dict[str, pd.DataFrame] = {}
    for name, model in (("classifier", clf), ("regressor", reg)):
        frame = _importance_frame(model, model.lagged_feature_names)
        if frame is None:
            logger.warning("no native feature importances for the %s head", name)
            continue
        out[f"feature_importance_{name}_per_horizon"] = _with_mean_importance(frame)
    return out


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def run_composite_stage(
    cfg: ExperimentConfig,
    model_name: str,
    paradigm: str,
    seed: int,
    stage_name: str,
    data: CompositeData,
    *,
    store: RunStore | None = None,
    tracker: Any = None,
    force: bool = False,
    progress_every: int = 0,
    importance: bool | None = None,
    max_folds: int | None = None,
) -> StageOutcome:
    """Run (or skip) one composite ``(model, paradigm, seed, stage)`` backtest.

    ``importance`` defaults to "yes for the hurdle's global test stage, no
    otherwise", which is where ``final_hurdle.ipynb`` produced the two
    ``feature_importance_*_per_horizon.csv`` files. Permutation importance
    (cell G) is NOT run: nothing under ``golden/results/finalhurdle/`` holds its
    output, and ten repeats over seven horizons and ~500 lagged features costs
    hours. :func:`_importance_frame` and the fitted heads are what a later
    permutation pass would need.

    ``max_folds`` truncates the schedule. It exists for golden runs on a time
    budget and is part of the stage identity, so a fold-limited stage can never
    be mistaken for a complete one.
    """
    if not isinstance(data, _data_stage.CompositeData):
        raise TypeError(
            f"model {model_name!r} is a composite; run_composite_stage needs the "
            "CompositeData that prepare_data() returns for the hurdle and damage "
            f"families, got {type(data).__name__}"
        )

    store = store if store is not None else RunStore(resolve_store_root(cfg))
    spec = get_spec(model_name, cfg.name)
    if spec.kind != "composite":
        raise ValueError(
            f"model {model_name!r} has kind {spec.kind!r}; run_composite_stage drives "
            "composites only"
        )
    family = spec.family
    if family == "damage" and str(paradigm) != "global":
        raise ValueError(
            "the damage family is global only: `damage_classifier.ipynb` defines no "
            f"per-activity or per-region wrapper, so paradigm={paradigm!r} has no "
            "legacy counterpart"
        )

    stage_cfg = cfg.stage(stage_name)
    backtest_cfg = stage_cfg.backtest(cfg.split)
    params, params_source = resolve_params(cfg, spec, model_name, store)

    key = RunKey(cfg.name, model_name, str(paradigm), int(seed))
    resolved_stage = {
        "experiment": cfg.name,
        "model": model_name,
        "paradigm": str(paradigm),
        "stage": stage_name,
        "backtest": backtest_cfg.model_dump(mode="json"),
        "naive_scales": stage_cfg.naive_scales.model_dump(mode="json"),
        "calibration": cfg.calibration.model_dump(mode="json"),
        "common_kwargs": cfg.common_kwargs.model_dump(mode="json"),
        "params": params,
        "device": cfg.device_for(model_name),
        "threads": cfg.threads,
        "heads": list(data.heads),
        "max_folds": max_folds,
    }
    # C18: a calibrated test stage applies the tuning-seed CV calibrators;
    # refuse up front when they are missing, and let them move the identity.
    key_for_calibration = RunKey(cfg.name, model_name, str(paradigm), int(seed))
    calibrators_digest = _require_calibrators(cfg, store, key_for_calibration, stage_name)
    if calibrators_digest is not None:
        resolved_stage["calibrators"] = calibrators_digest
    digest = stage_hash(resolved_stage, list(data.upstream), int(seed))

    if not force and store.is_complete(key, stage_name, digest):
        logger.info("skip %s/%s: complete under the same identity", key.relative(), stage_name)
        return StageOutcome(key, stage_name, digest, skipped=True, params_source=params_source)

    owns_tracker = tracker is None
    tracker = tracker if tracker is not None else make_tracker(cfg)

    seed_everything(int(seed))
    # `head_past_lags` (one entry per head) reaches `_build_hurdle` through the
    # context; damage heads are legacy selections, so theirs are None.
    ctx = make_run_context(cfg, model_name, int(seed), data)

    store.start_stage(key, stage_name, digest)
    store.write_config(key, cfg.model_dump(mode="json"))
    env = record_env()
    try:
        tracked_id = tracker.start(
            key,
            cfg.model_dump(mode="json"),
            tags=tracker_tags(cfg, spec),
            stage=stage_name,
        )
    except Exception as exc:  # pragma: no cover - the mirror never fails a run
        logger.warning("tracker.start failed: %s", exc)
        tracked_id = None
    if tracked_id:
        env = {**env, "tracker_run_id": tracked_id}
    store.write_env(key, env)
    if tracked_id:
        # §5.5: the run directory must name its mirror. The store owns the
        # merge, so `env.json` keeps the id of every stage, not just this one.
        store.record_tracker_run_id(key, tracked_id, stage=stage_name)

    try:
        series = _stage_series(cfg, data, family, stage_name)
        builders = spec.build(params, ctx)
        folds = schedule_from_config(series.level_targets[0], backtest_cfg)
        if max_folds is not None:
            folds = folds[: int(max_folds)]

        persist = ChannelPersistHook(
            store,
            key,
            stage_name,
            backtest_cfg.retrain_stride,
            actuals_by_channel=series.actuals,
            region_names=series.region_names,
        )
        _run_backtest(
            family=family,
            builders=builders,
            series=series,
            paradigm=str(paradigm),
            backtest_cfg=backtest_cfg,
            activity_by_region=data.activity_by_region,
            persist=persist,
            folds=folds,
            progress_every=progress_every,
            max_folds=max_folds,
        )

        preds = store.load_predictions(key, stage_name, legacy_order=True)
        views, component_globals, extras = _score(
            cfg, data, store, key, stage_name, stage_cfg, family, preds, series
        )
        metric_paths = store.write_metrics(key, stage_name, views)
        metric_paths.update(
            _write_component_globals(store, key, stage_name, component_globals)
        )

        if "diagnostic" in extras:
            store.write_artifact(key, CALIBRATION_DIAGNOSTIC_ARTIFACT, extras["diagnostic"])

        want_importance = (
            (family == "hurdle" and stage_name == "test" and str(paradigm) == "global")
            if importance is None
            else importance
        )
        if want_importance and family == "hurdle":
            if stage_name != "test":
                # Cell 50 slices `full_target_c` at `TRAIN_VAL_END`, which only
                # means "the test start" on the FULL series; on the CV view the
                # same fraction lands somewhere inside validation.
                logger.warning(
                    "feature importance is a test-stage artefact (cells E-F fit on "
                    "train+val of the FULL series); skipping it for stage %r",
                    stage_name,
                )
            else:
                for name, frame in _hurdle_importances(
                    builders, series, data.bundle.train_val_end
                ).items():
                    store.write_artifact(key, f"{name}.parquet", frame)

        store.complete_stage(key, stage_name)
        track(tracker, "log_tables", views, stage=stage_name)
        track(tracker, "log_artifact", store.predictions_dir(key, stage_name), "predictions")
        track(tracker, "log_artifact", store.metrics_dir(key, stage_name), "metrics")
    except Exception as exc:
        store.fail_stage(key, stage_name, f"{type(exc).__name__}: {exc}")
        if owns_tracker:
            track(tracker, "finish", "failed")
        raise
    else:
        if owns_tracker:
            track(tracker, "finish")

    logger.info(
        "%s/%s: %d folds, %d rows, %d metric files",
        key.relative(),
        stage_name,
        len(folds),
        len(preds),
        len(metric_paths),
    )
    return StageOutcome(
        key,
        stage_name,
        digest,
        skipped=False,
        n_folds=len(folds),
        n_rows=len(preds),
        metric_paths=metric_paths,
        params_source=params_source,
    )


def _stage_series(
    cfg: ExperimentConfig, data: CompositeData, family: str, stage_name: str
) -> Any:
    if family == "hurdle":
        return hurdle_series(
            data.head("classifier").bundle, data.head("regressor").bundle, stage_name
        )
    if family == "damage":
        return damage_series(data.bundles, stage_name)
    raise ValueError(
        f"experiment {cfg.name!r}: no composite wiring for family {family!r}; "
        "expected 'hurdle' or 'damage'"
    )


def _run_backtest(
    *,
    family: str,
    builders: Any,
    series: Any,
    paradigm: str,
    backtest_cfg: Any,
    activity_by_region: Mapping[str, Any],
    persist: ChannelPersistHook,
    folds: Sequence[Any],
    progress_every: int,
    max_folds: int | None,
) -> None:
    """Drive the engine under the paradigm's partition.

    One group streams straight through the persist hook; several groups run one
    engine each -- which is what the legacy ``*_per_activity`` /
    ``*_per_region`` wrappers do -- and the restored-order result is replayed
    through the same hook afterwards.
    """
    groups = partition(series.region_names, paradigm, activity_by_region)  # type: ignore[arg-type]
    min_positive = min_positive_samples_for(paradigm) if family == "hurdle" else None

    def build(indices: Sequence[int] | None) -> Any:
        if family == "hurdle":
            return make_hurdle_forecaster(
                builders, series, indices=indices, min_positive_samples=min_positive
            )
        return make_damage_forecaster(builders, series, indices=indices)

    hooks: list[Any] = [ProgressHook(progress_every)] if progress_every else []

    if len(groups) == 1:
        engine = ExpandingWindowBacktest(backtest_cfg, Identity(), hooks=[persist, *hooks])
        for i, _ in enumerate(
            engine.iter_folds(build(None), list(series.level_targets)), start=1
        ):
            if max_folds is not None and i >= int(max_folds):
                break
        persist.close()
        return

    results = []
    for group in groups:
        logger.info("group %s (%d regions)", group.label, len(group.indices))
        engine = ExpandingWindowBacktest(backtest_cfg, Identity(), hooks=hooks)
        collected: dict[str, list[list[TimeSeries]]] = {}
        targets = [series.level_targets[i] for i in group.indices]
        for i, result in enumerate(
            engine.iter_folds(build(group.indices), targets), start=1
        ):
            for channel, region_preds in result.predictions.items():
                bucket = collected.setdefault(channel, [[] for _ in group.indices])
                for r_idx, pred in enumerate(region_preds):
                    bucket[r_idx].append(pred)
            if max_folds is not None and i >= int(max_folds):
                break
        results.append((group, collected))

    fold_preds = restore_order(results, len(series.level_targets))
    _replay(persist, folds, fold_preds)


def _score(
    cfg: ExperimentConfig,
    data: CompositeData,
    store: RunStore,
    key: RunKey,
    stage_name: str,
    stage_cfg: StageConfig,
    family: str,
    preds: PredictionSet,
    series: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Calibrate, build the components and evaluate them."""
    if family == "hurdle":
        prob_frames = {"prob": _as_prob_frame(preds.for_channel("prob").frame)}
        calibrated, extras = _calibrate(store, key, stage_name, prob_frames, cfg)
        components, _ = _hurdle_components(preds, calibrated)
        # B8/D6: the thesis reports the Beta-calibrated hurdle (Table 4).
        primary = HURDLE_PRIMARY_COMPONENT
    else:
        keys = list(series.keys)
        prob_frames = {k: _as_prob_frame(preds.for_channel(k).frame) for k in keys}
        calibrated, extras = _calibrate(store, key, stage_name, prob_frames, cfg)
        components, _ = _damage_components(preds, calibrated, keys)
        primary = components[0].name

    mae_scales, rmse_scales = _naive_scales(data.bundle, stage_cfg)
    views, component_globals = _evaluate_components(
        components, primary, mae_scales, rmse_scales, cfg.calibration.threshold
    )
    return views, component_globals, extras


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def run_composite_experiment(
    cfg: ExperimentConfig,
    *,
    data: CompositeData | None = None,
    models: Sequence[str] | None = None,
    paradigms: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    stages: Sequence[str] = ("cv", "test"),
    store: RunStore | None = None,
    force: bool = False,
    progress_every: int = 0,
    max_folds: int | None = None,
) -> list[StageOutcome]:
    """Run the composite entries of ``cfg`` over paradigms, seeds and stages.

    The same two §5.4 rules as :func:`strikecast.pipeline.run_stage.run_experiment`:
    CV stays single-seed under the tuning seed, and a deterministic model runs
    once. Both composite specs are ``stochastic=True`` (the SPE classifier draws
    a fresh under-sample per boosting iteration and CatBoost's bootstrap is
    seeded), so both are swept.

    ``stages`` order matters: ``cv`` must run before ``test`` under the same run
    key, because the test stage applies the calibrators the CV stage wrote
    (cell 27).
    """
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    if data is None:
        data = _data_stage.prepare_composite_data(cfg, store)

    names = list(models) if models else [
        name for name in cfg.model_names if get_spec(name, cfg.name).kind == "composite"
    ]
    paradigm_names = list(paradigms) if paradigms else [str(p) for p in cfg.paradigm_names]
    seed_list = [int(s) for s in (seeds if seeds else cfg.seeds.eval_seeds)]
    tuning_seed = int(cfg.seeds.tuning_seed)

    outcomes: list[StageOutcome] = []
    for model_name in names:
        spec = get_spec(model_name, cfg.name)
        for paradigm in paradigm_names:
            for stage_name in stages:
                if stage_name == "cv":
                    stage_seeds = [tuning_seed]
                elif not spec.stochastic:
                    stage_seeds = seed_list[:1]
                else:
                    stage_seeds = seed_list
                for seed in stage_seeds:
                    outcomes.append(
                        run_composite_stage(
                            cfg,
                            model_name,
                            paradigm,
                            seed,
                            stage_name,
                            data,
                            store=store,
                            force=force,
                            progress_every=progress_every,
                            max_folds=max_folds,
                        )
                    )
    return outcomes
