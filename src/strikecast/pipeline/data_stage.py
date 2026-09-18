"""Stage 1: config -> panel -> SeriesBundle -> feature selection.

The three shared artefacts of plan §5.3 ::

    runs/<experiment>/shared/
      panel.<hash>.parquet          + panel.<hash>.meta.json (global weather cols)
      series.<hash>/                  SeriesBundle (parquet + manifest.json)
      feature_selection.<hash>.json

are built once per experiment and reused by every model, paradigm, seed and
stage. Each hash is a content hash of the resolved config slice that produced
the artefact plus the hashes of the artefacts upstream of it, so changing
``data:`` invalidates all three while changing only ``feature_selection:``
invalidates the last one.

Order, exactly as every legacy notebook has it (plan §2.2, steps 1-7):

1. :func:`strikecast.data.load.load_inputs`;
2. :func:`strikecast.data.panel.build_panel`;
3. :func:`strikecast.data.covariates.split_covariates`;
4. :func:`strikecast.data.series.build_bundle` (window transforms, static
   encoding, 70/10/20 split, CV view, plus the un-windowed RNN past covariates);
5. :func:`strikecast.data.feature_selection.select_top_k` on
   ``(target_train, past_covs, future_covs)`` -- the level target even for the
   diff family (F29, Q8) -- or the cached selection;
6. :func:`strikecast.data.series.subset_components`, which re-runs the whole
   encoding and split on the subset lists, as the notebooks do.

The feature selection is **data to load, not output to reproduce** (F16): the
legacy sets came out of a process whose ``PYTHONHASHSEED`` is unknowable. Point
``feature_selection.cache_path`` at a converted golden set to run the thesis'
own features; leave it unset to select afresh under ``PYTHONHASHSEED=0``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strikecast.data.cache import content_hash
from strikecast.pipeline.context import resolve_store_root
from strikecast.store import RunStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from strikecast.config.schema import ExperimentConfig
    from strikecast.data.load import Inputs
    from strikecast.data.series import SeriesBundle

__all__ = [
    "DataArtifacts",
    "FeatureSets",
    "build_or_load_bundle",
    "build_or_load_features",
    "build_or_load_panel",
    "prepare_data",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSets:
    """The surviving covariate components, and where they came from.

    ``source`` is one of ``"computed"`` (a fresh LightGBM gain ranking),
    ``"store"`` (this experiment's own cached JSON), ``"cache_path"`` (the
    converted legacy set named by ``feature_selection.cache_path``) or
    ``"all"`` (feature selection disabled, every component kept).
    """

    past_keep: list[str]
    future_keep: list[str]
    source: str
    hash: str
    path: Path | None = None


@dataclass(frozen=True)
class DataArtifacts:
    """Everything the downstream stages need, plus the upstream hashes.

    ``bundle`` is the POST-feature-selection bundle: the lists the legacy
    notebooks hold after their second ``get_covs_and_encodings`` call.
    """

    bundle: SeriesBundle
    features: FeatureSets
    panel_hash: str
    series_hash: str
    activity_by_region: dict[str, Any] = field(default_factory=dict)
    panel: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def region_names(self) -> list[str]:
        return list(self.bundle.region_names)

    @property
    def upstream(self) -> tuple[str, str, str]:
        """The three hashes a stage identity is built on (§5.3)."""
        return (self.panel_hash, self.series_hash, self.features.hash)


# --------------------------------------------------------------------------- #
# hashes
# --------------------------------------------------------------------------- #
def _panel_hash(cfg: ExperimentConfig) -> str:
    return content_hash("panel", cfg.data.model_dump(mode="json"))


def _series_hash(cfg: ExperimentConfig, panel_hash: str) -> str:
    return content_hash("series", cfg.series.model_dump(mode="json"), panel_hash)


def _features_hash(cfg: ExperimentConfig, panel_hash: str, series_hash: str) -> str:
    return content_hash(
        "feature_selection",
        cfg.feature_selection.model_dump(mode="json"),
        cfg.common_kwargs.model_dump(mode="json"),
        panel_hash,
        series_hash,
    )


def _store_for(cfg: ExperimentConfig, store: RunStore | None) -> RunStore:
    return store if store is not None else RunStore(resolve_store_root(cfg))


# --------------------------------------------------------------------------- #
# 1-2. inputs and panel
# --------------------------------------------------------------------------- #
def build_or_load_panel(
    cfg: ExperimentConfig,
    store: RunStore | None = None,
    *,
    inputs: Inputs | None = None,
    cache: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, list[str], str]:
    """The long panel, its global weather columns and its content hash.

    ``global_weather_columns`` cannot be recovered from the panel alone (their
    order is process-dependent, panel quirk Q2, and it carries into the future
    covariate order), so they are cached next to the parquet in a sidecar JSON.
    """
    import pandas as pd  # noqa: PLC0415

    store = _store_for(cfg, store)
    digest = _panel_hash(cfg)
    path = store.shared_path(cfg.name, "panel", digest)
    meta_path = path.with_suffix(".meta.json")

    if cache and not force and path.exists() and meta_path.exists():
        logger.info("panel: cache hit %s", path)
        panel = pd.read_parquet(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return panel, list(meta["global_weather_columns"]), digest

    from strikecast.data.load import load_inputs  # noqa: PLC0415
    from strikecast.data.panel import build_panel  # noqa: PLC0415

    if inputs is None:
        inputs = load_inputs(cfg.data.fixed_dir, cfg.data.dataset_dir)

    result = build_panel(
        inputs,
        target=cfg.data.target,
        binarize=list(cfg.data.binarize),
        low_prevalence_ratio=cfg.data.low_prevalence_ratio,
        add_interactions=cfg.data.add_interactions,
        binarize_stage=cfg.data.binarize_stage,
    )
    if len(result.panels) != 1:
        raise NotImplementedError(
            f"experiment {cfg.name!r} produced {len(result.panels)} panels; the multi-target "
            "damage family is P5, not P3"
        )
    panel = result.panel

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path, index=False)
        meta_path.write_text(
            json.dumps(
                {
                    "global_weather_columns": list(result.global_weather_columns),
                    "rows": int(len(panel)),
                    "columns": int(panel.shape[1]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("panel: wrote %s (%d rows)", path, len(panel))

    return panel, list(result.global_weather_columns), digest


# --------------------------------------------------------------------------- #
# 3-4. covariate split and series bundle
# --------------------------------------------------------------------------- #
def build_or_load_bundle(
    cfg: ExperimentConfig,
    store: RunStore | None,
    panel: pd.DataFrame,
    global_weather_columns: list[str],
    activity_by_region: dict[str, Any] | None = None,
    *,
    panel_hash: str | None = None,
    cache: bool = True,
    force: bool = False,
) -> tuple[SeriesBundle, str]:
    """The PRE-selection :class:`SeriesBundle` and its content hash.

    The bundle keeps its un-encoded lists (``bundle.raw``), because
    :func:`strikecast.data.series.subset_components` re-encodes from them.
    """
    from strikecast.data.covariates import split_covariates  # noqa: PLC0415
    from strikecast.data.series import build_bundle, serialise  # noqa: PLC0415
    from strikecast.data.series import load as load_bundle

    store = _store_for(cfg, store)
    panel_hash = panel_hash or _panel_hash(cfg)
    digest = _series_hash(cfg, panel_hash)
    directory = store.shared_path(cfg.name, "series", digest)

    if cache and not force and (directory / "manifest.json").exists():
        logger.info("series bundle: cache hit %s", directory)
        return load_bundle(directory), digest

    split = split_covariates(panel, list(global_weather_columns), cfg.data.target)
    bundle = build_bundle(
        panel,
        cfg.data.target,
        list(split.past_covariates),
        list(split.future_covariates),
        cfg.series,
        activity_by_region=activity_by_region,
    )
    if cache:
        serialise(bundle, directory, time_col=cfg.series.time_col)
        logger.info("series bundle: wrote %s (%d regions)", directory, len(bundle))
    return bundle, digest


# --------------------------------------------------------------------------- #
# 5. feature selection
# --------------------------------------------------------------------------- #
def _sets_from_payload(payload: dict[str, Any]) -> tuple[list[str], list[str]] | None:
    """Read past/future components from either JSON schema.

    ``FeatureSelection.to_json`` writes ``past_keep``/``future_keep``; the
    converted legacy sets under ``golden/converted/feature_sets/`` write
    ``past_covariate_components``/``future_covariate_components``.
    """
    for past_key, future_key in (
        ("past_keep", "future_keep"),
        ("past_covariate_components", "future_covariate_components"),
    ):
        if past_key in payload and future_key in payload:
            return sorted(payload[past_key]), sorted(payload[future_key])
    return None


def build_or_load_features(
    cfg: ExperimentConfig,
    store: RunStore | None,
    bundle: SeriesBundle,
    *,
    panel_hash: str | None = None,
    series_hash: str | None = None,
    force: bool = False,
) -> FeatureSets:
    """The selected covariate components, from cache or from a fresh ranking.

    Resolution order:

    1. ``feature_selection.cache_path`` -- the converted legacy set. This is the
       only way to reproduce the thesis' own features (F16), so it wins over
       everything.
    2. ``shared/feature_selection.<hash>.json`` written by a previous run, when
       ``feature_selection.cache`` is on.
    3. a fresh :func:`strikecast.data.feature_selection.select_top_k`.
    """
    store = _store_for(cfg, store)
    panel_hash = panel_hash or _panel_hash(cfg)
    series_hash = series_hash or _series_hash(cfg, panel_hash)
    digest = _features_hash(cfg, panel_hash, series_hash)
    path = store.shared_path(cfg.name, "feature_selection", digest)

    cache_path = cfg.feature_selection.cache_path
    if cache_path:
        legacy = Path(cache_path).expanduser()
        if not legacy.is_file():
            raise FileNotFoundError(
                f"feature_selection.cache_path={cache_path!r} does not exist; unset it to "
                "select features afresh (F16)"
            )
        sets = _sets_from_payload(json.loads(legacy.read_text(encoding="utf-8")))
        if sets is None:
            raise ValueError(
                f"{legacy} holds neither past_keep/future_keep nor "
                "past_covariate_components/future_covariate_components"
            )
        logger.info("feature selection: loaded the cached legacy set %s", legacy)
        return FeatureSets(sets[0], sets[1], "cache_path", digest, legacy)

    if cfg.feature_selection.cache and not force and path.exists():
        sets = _sets_from_payload(json.loads(path.read_text(encoding="utf-8")))
        if sets is not None:
            logger.info("feature selection: cache hit %s", path)
            return FeatureSets(sets[0], sets[1], "store", digest, path)

    from strikecast.data.feature_selection import select_top_k  # noqa: PLC0415

    selection = select_top_k(
        bundle.target_train,
        bundle.past_covs,
        bundle.future_covs,
        cfg.feature_selection.build(),
    )
    if cfg.feature_selection.cache:
        selection.to_json(path)
        logger.info("feature selection: wrote %s", path)
    return FeatureSets(
        list(selection.past_keep),
        list(selection.future_keep),
        "computed",
        digest,
        path if cfg.feature_selection.cache else None,
    )


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #
def prepare_data(
    cfg: ExperimentConfig,
    store: RunStore | None = None,
    *,
    inputs: Inputs | None = None,
    cache: bool = True,
    force: bool = False,
    keep_panel: bool = False,
) -> DataArtifacts:
    """Run the whole data stage and return everything downstream needs."""
    from strikecast.data.series import subset_components  # noqa: PLC0415

    store = _store_for(cfg, store)

    panel, weather_cols, panel_hash = build_or_load_panel(
        cfg, store, inputs=inputs, cache=cache, force=force
    )
    activity = dict(inputs.activity_by_region) if inputs is not None else None
    if activity is None:
        from strikecast.data.load import load_inputs  # noqa: PLC0415

        # The activity map is tiny and is NOT in the panel cache, so it is read
        # back even on a cache hit; `partition(paradigm="activity")` needs it.
        activity = dict(
            load_inputs(cfg.data.fixed_dir, cfg.data.dataset_dir).activity_by_region
        )

    bundle, series_hash = build_or_load_bundle(
        cfg,
        store,
        panel,
        weather_cols,
        activity,
        panel_hash=panel_hash,
        cache=cache,
        force=force,
    )
    features = build_or_load_features(
        cfg, store, bundle, panel_hash=panel_hash, series_hash=series_hash, force=force
    )
    selected = subset_components(bundle, features.past_keep, features.future_keep)

    return DataArtifacts(
        bundle=selected,
        features=features,
        panel_hash=panel_hash,
        series_hash=series_hash,
        activity_by_region=dict(bundle.activity_by_region or activity or {}),
        panel=panel if keep_panel else None,
    )
