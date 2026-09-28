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
   ``(target_train, past_covs, future_covs)`` -- or the cached selection. The
   legacy ``diffreg`` fits the LEVEL target (F29, Q8); the figure's ``diff_l2``
   fits the differenced one (:func:`strikecast.data.series.model_space_parts`);
6. :func:`strikecast.data.series.subset_components`, which re-runs the whole
   encoding and split on the subset lists, as the notebooks do.

The figure protocol (plan 2026-09-28; the selector name decides it, see
``FeatureSelectionStageConfig.protocol``) changes three things here: the
selection keeps exact ``(feature, lag)`` pairs (:attr:`FeatureSets.past_lags`,
persisted in the JSON and REQUIRED on a cache hit), every future covariate is
kept, and a ranking table ``feature_selection.<hash>.ranking.csv`` is written
next to the JSON. One experiment can hold several selections (count:
``count_poisson`` + ``count_tweedie``); :func:`prepare_by_selection` builds one
:class:`DataArtifacts` per distinct selection
(``ExperimentConfig.selection_groups``) and :func:`select_features` caches all
of them.

The feature selection is **data to load, not output to reproduce** (F16): the
legacy sets came out of a process whose ``PYTHONHASHSEED`` is unknowable. Point
``feature_selection.cache_path`` at a converted golden set to run the thesis'
own features; leave it unset to select afresh under ``PYTHONHASHSEED=0``.

Audit 2026-09-26 (A12-A14, D2) adds three guards:

* every selection this module writes records its **provenance** (window
  config, selector config, upstream hashes, ``PYTHONHASHSEED``, library
  versions), and a cached selection -- ``cache_path`` or the store -- made under
  a different window/selector config is REFUSED (:class:`FeatureCacheMismatch`).
  The converted golden JSONs carry no provenance and are accepted only under
  ``series.window.expdecay = "legacy_alpha"`` and for their own selector;
* ``feature_selection.require_cached`` makes the selection a separate step
  (``strikecast featsel experiment=<name>``, :func:`select_features`); every
  other caller then loads it and fails loudly when it is missing
  (:class:`FeatureSelectionMissing`) instead of re-selecting per job;
* the panel hash covers the **content** of the four input files, not only
  their paths, so an edited parquet can never reuse a stale panel.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
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
    "CompositeData",
    "DataArtifacts",
    "FeatureCacheMismatch",
    "FeatureSelectionMissing",
    "FeatureSets",
    "build_or_load_bundle",
    "build_or_load_features",
    "build_or_load_panel",
    "head_configs",
    "is_composite_family",
    "prepare_composite_data",
    "check_selection_provenance",
    "input_digests",
    "prepare_by_selection",
    "prepare_data",
    "ranking_csv_path",
    "require_pythonhashseed_zero",
    "select_features",
    "selection_provenance",
]

logger = logging.getLogger(__name__)


class FeatureSelectionMissing(RuntimeError):  # noqa: N818 - reads better at the raise site
    """``feature_selection.require_cached`` is on and no cached selection exists."""


class FeatureCacheMismatch(ValueError):  # noqa: N818
    """A cached selection was made under a different window/selector config (A13)."""


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

    ``past_lags`` is the figure protocol's exact ``(feature, lag)`` selection,
    ``((component, (lag, ...)), ...)`` with the components in ``past_keep``
    order; ``None`` = legacy, every kept component gets all
    ``common_kwargs.lags_past_covariates``. Downstream builders turn it into
    darts per-component ``lags_past_covariates`` (plain ``list[int]`` values).
    """

    past_keep: list[str]
    future_keep: list[str]
    source: str
    hash: str
    path: Path | None = None
    past_lags: tuple[tuple[str, tuple[int, ...]], ...] | None = None


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


@dataclass(frozen=True)
class CompositeData(DataArtifacts):
    """One :class:`DataArtifacts` per head, for the hurdle and damage families.

    The hurdle family builds **two** panels and two bundles -- the unbinarised
    count panel for the CatBoost head and the binarised event panel for the SPE
    classifier -- and runs **two** feature selections (F27,
    ``final_hurdle.ipynb`` cells 3, 10 and 11). The damage family builds one
    panel, one bundle and one selection **per damage key**
    (``damage_classifier.ipynb`` cells 4, 12 and 13).

    ``CompositeData`` is a :class:`DataArtifacts` in its own right: the inherited
    fields are the PRIMARY head's, so everything that already consumes a
    ``DataArtifacts`` -- ``run_experiment``, the naive scales, the region names,
    the activity partition -- keeps working unchanged. :attr:`heads` is what
    :mod:`strikecast.pipeline.composite_stage` reads.

    The primary head is the one the family's headline numbers are scaled and
    partitioned on: the ``regressor`` head for the hurdle (its count targets are
    what ``compute_naive_scales`` runs on, ``final_hurdle.ipynb`` cell 17) and
    the FIRST damage key for the damage family (the key both legacy loops take
    their schedule from, F68).
    """

    heads: dict[str, DataArtifacts] = field(default_factory=dict)
    primary: str = ""
    family: str = ""

    def head(self, name: str) -> DataArtifacts:
        try:
            return self.heads[name]
        except KeyError:
            raise KeyError(
                f"no head {name!r} in this {self.family or 'composite'} data; "
                f"known: {sorted(self.heads)}"
            ) from None

    @property
    def head_names(self) -> list[str]:
        return list(self.heads)

    @property
    def bundles(self) -> dict[str, Any]:
        """``head -> post-selection SeriesBundle``, in head order."""
        return {name: art.bundle for name, art in self.heads.items()}

    @property
    def upstream(self) -> tuple[str, ...]:  # type: ignore[override]
        """EVERY head's three hashes, head-major, so the stage identity moves
        when any panel, bundle or selection of any head moves."""
        out: list[str] = []
        for name, art in self.heads.items():
            out.extend((name, *art.upstream))
        return tuple(out)


# --------------------------------------------------------------------------- #
# hashes
# --------------------------------------------------------------------------- #
#: The four files ``strikecast.data.load.load_inputs`` reads, relative to
#: ``data.fixed_dir`` / ``data.dataset_dir``.
INPUT_FILES: tuple[tuple[str, str], ...] = (
    ("fixed_dir", "regions.txt"),
    ("fixed_dir", "regions_activity_cat.json"),
    ("fixed_dir", "actors.json"),
    ("dataset_dir", "master_combined_timeseries.parquet"),
)

_FILE_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_sha256(path: Path) -> str | None:
    """sha256 of one file (memoised on path, size and mtime); ``None`` if absent."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_DIGESTS.get(key)
    if cached is None:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        cached = _FILE_DIGESTS[key] = digest.hexdigest()
    return cached


def input_digests(cfg: ExperimentConfig) -> dict[str, str | None]:
    """``file name -> sha256`` of the panel's input files (A14).

    A missing file hashes to ``None`` (unit tests pass in-memory inputs), so the
    digest is still deterministic.
    """
    out: dict[str, str | None] = {}
    for attr, name in INPUT_FILES:
        out[name] = _file_sha256(Path(getattr(cfg.data, attr)).expanduser() / name)
    return out


def _panel_hash(cfg: ExperimentConfig) -> str:
    """Config slice AND input-file content (A14): a changed parquet is a new panel."""
    return content_hash("panel", cfg.data.model_dump(mode="json"), input_digests(cfg))


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
# composite families: one head per panel
# --------------------------------------------------------------------------- #
#: The head whose artefacts a :class:`CompositeData` exposes as its own.
HURDLE_PRIMARY_HEAD = "regressor"

#: The ``feature_selections`` keys the hurdle family's two selections use
#: (``configs/experiment/hurdle.yaml``; F27).
HURDLE_HEADS: tuple[str, str] = ("regressor", "classifier")

#: The selectors fitted with positive-only sample weights (F27, Q7):
#: ``final_hurdle.ipynb`` cell 10 is the only legacy selection that passes
#: ``sample_weight``; the figure's ``hurdle_tweedie_pos`` keeps that weighting
#: because the CatBoost count head it feeds trains on it (plan 2026-09-28).
WEIGHTED_SELECTORS: frozenset[str] = frozenset({"zipoisson_regressor", "hurdle_tweedie_pos"})


def is_composite_family(cfg: ExperimentConfig) -> bool:
    """Does this experiment need more than one panel?

    Decided from the CONFIG SHAPE, never from the experiment name:

    * ``data.panel_variant == "damage"`` -- the late-binarisation variant
      produces one frame per ``data.binarize`` entry, so every entry is a head;
    * ``feature_selections`` naming both hurdle heads -- two selections means
      two covariate subsets, which means two bundles (F27).
    """
    if cfg.data.panel_variant == "damage":
        return True
    return all(head in cfg.feature_selections for head in HURDLE_HEADS)


def head_configs(cfg: ExperimentConfig) -> dict[str, ExperimentConfig]:
    """``head -> the single-panel config that builds that head's artefacts``.

    Each head is an ordinary :class:`ExperimentConfig` with ``data`` and
    ``feature_selection`` narrowed to that head, so the whole existing data
    stage -- panel, bundle, selection, subset -- runs unchanged per head and
    each artefact lands in the SAME ``shared/`` directory under its own content
    hash (the hash covers ``data`` and ``feature_selection``, which is exactly
    what differs).

    Hurdle (``final_hurdle.ipynb`` cell 3)::

        regressor   target=<T>          binarize=[]     selection=hurdle_tweedie_pos
        classifier  target=<T>_binary   binarize=[<T>]  selection=hurdle_binary

    (``legacy=hurdle``: ``zipoisson_regressor`` / ``zipoisson_classifier``.)

    Damage (``damage_classifier.ipynb`` cells 4 and 12): one head per entry of
    ``data.binarize``, each ``target=<key>_binary`` with ``binarize=[key]`` and
    the shared ``zipoisson_classifier`` selection re-fitted per key.

    A non-composite experiment returns ``{"": cfg}`` -- one unnamed head, which
    is what :func:`prepare_data` already builds.
    """
    if not is_composite_family(cfg):
        return {"": cfg}

    if cfg.data.panel_variant == "damage":
        if not cfg.data.binarize:
            raise ValueError(
                f"experiment {cfg.name!r} is the damage panel variant but lists no "
                "`data.binarize` targets; there is nothing to build a head from"
            )
        return {
            key: cfg.model_copy(
                update={
                    "data": cfg.data.model_copy(
                        update={"target": f"{key}_binary", "binarize": [key]}
                    ),
                    "feature_selection": cfg.feature_selection_for(key),
                }
            )
            for key in cfg.data.binarize
        }

    target = cfg.data.target
    if not target:
        raise ValueError(f"experiment {cfg.name!r} needs `data.target` for its hurdle heads")
    return {
        "regressor": cfg.model_copy(
            update={
                "data": cfg.data.model_copy(update={"target": target, "binarize": []}),
                "feature_selection": cfg.feature_selection_for("regressor"),
            }
        ),
        "classifier": cfg.model_copy(
            update={
                "data": cfg.data.model_copy(
                    update={"target": f"{target}_binary", "binarize": [target]}
                ),
                "feature_selection": cfg.feature_selection_for("classifier"),
            }
        ),
    }


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
        raise ValueError(
            f"experiment {cfg.name!r} produced {len(result.panels)} panels from "
            f"data.binarize={list(cfg.data.binarize)}; one panel is one cache entry, "
            "so a multi-target family must go through head_configs()/"
            "prepare_composite_data(), which narrows data.binarize to one key per head"
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
PastLags = tuple[tuple[str, tuple[int, ...]], ...]


def _sets_from_payload(
    payload: dict[str, Any],
    protocol: str = "legacy",
    source: Path | str | None = None,
) -> tuple[list[str], list[str], PastLags | None] | None:
    """Read ``(past, future, past_lags)`` from either JSON schema.

    ``past_lags`` is the schema-2 ``[[component, [lags]], ...]`` list, or
    ``None`` for a legacy payload. Under the figure ``protocol`` a payload
    WITHOUT ``past_lags`` is not a usable selection -- downstream would silently
    fall back to all lags of every kept component -- so it raises
    :class:`FeatureCacheMismatch`.

    ``FeatureSelection.to_json`` writes ``past_keep``/``future_keep``; the
    converted legacy sets under ``golden/converted/feature_sets/`` write
    ``past_covariate_components``/``future_covariate_components``.

    **The recorded order is kept, never sorted** (F16). ``subset_components``
    iterates these names exactly as given and the resulting TimeSeries
    component order is the column order the GBDTs are fitted on, so sorting
    them re-orders every lagged feature and changes LightGBM's tie-breaking:
    measured on the real diff panel, sorting moved the CV ``MASE_mean`` of
    ``lightgbm`` by 6.9% against ``golden/results/diff``, while keeping the
    order reproduces it to 1e-9 (level E, ``tests/golden/test_pipeline_equality.py``).
    The cached names are data to load, and their order is part of that data.
    """
    from strikecast.data.feature_selection import past_lags_from_json  # noqa: PLC0415

    for past_key, future_key in (
        ("past_keep", "future_keep"),
        ("past_covariate_components", "future_covariate_components"),
    ):
        if past_key in payload and future_key in payload:
            past_lags = past_lags_from_json(payload.get("past_lags"))
            if protocol == "figure" and past_lags is None:
                raise FeatureCacheMismatch(
                    f"{source or 'the cached selection'} has no `past_lags`: it is not a "
                    "figure-protocol selection (schema 2), so the exact (feature, lag) pairs "
                    "are unknown. Re-run `strikecast featsel` for this configuration "
                    "(plan 2026-09-28)."
                )
            return list(payload[past_key]), list(payload[future_key]), past_lags
    return None


def _selection_sample_weight(
    cfg: ExperimentConfig, bundle: SeriesBundle
) -> list[Any] | None:
    """``final_hurdle.ipynb`` cell 10's ``sample_weight``, or ``None`` (F27/Q7).

    ``full_weights`` there is ``make_positive_only_weights(target_series_list_r)``
    -- the UN-ENCODED full target list -- and every weight series is then
    ``slice_intersect``ed with the training target it belongs to. Applies to
    every selector in :data:`WEIGHTED_SELECTORS` (``zipoisson_regressor``,
    ``hurdle_tweedie_pos``).
    """
    if cfg.feature_selection.selector not in WEIGHTED_SELECTORS:
        return None

    from strikecast.data.series import positive_only_weights  # noqa: PLC0415

    source = bundle.raw.target if bundle.raw is not None else bundle.target_full
    weights = positive_only_weights(source)
    return [
        w.slice_intersect(ts) for w, ts in zip(weights, bundle.target_train, strict=True)
    ]


def _fs_identity(fs: Any) -> dict[str, Any]:
    """The feature-selection fields that change WHICH features are selected.

    ``cache``, ``cache_path``, ``require_cached`` and ``per_key`` say where a
    selection lives, not what it is, and are left out.
    """
    return fs.model_dump(
        mode="json", include={"selector", "device", "num_threads", "top_k", "deterministic"}
    )


def _library_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out: dict[str, str | None] = {}
    for package in ("lightgbm", "darts", "pandas", "numpy"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


def selection_provenance(
    cfg: ExperimentConfig, panel_hash: str, series_hash: str, digest: str
) -> dict[str, Any]:
    """What a cached selection records about how it was made (A13).

    ``window``, ``feature_selection`` (:func:`_fs_identity`), ``common_kwargs``
    and ``target`` are what :func:`check_selection_provenance` compares; the
    rest is for the reader.
    """
    return {
        "schema": 2,
        "experiment": cfg.name,
        "target": cfg.data.target,
        "protocol": cfg.feature_selection.protocol,
        "transform": cfg.transform.kind,
        "expdecay": cfg.series.window.expdecay,
        "window": cfg.series.window.model_dump(mode="json"),
        "feature_selection": _fs_identity(cfg.feature_selection),
        "common_kwargs": cfg.common_kwargs.model_dump(mode="json"),
        "panel_hash": panel_hash,
        "series_hash": series_hash,
        "features_hash": digest,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "versions": _library_versions(),
    }


def check_selection_provenance(
    cfg: ExperimentConfig, payload: dict[str, Any], source: Path | str
) -> None:
    """Refuse a cached selection made under another window/selector config (A13).

    * A payload WITH ``provenance`` must match this run's window config,
      selector identity, lag skeleton and target exactly.
    * A payload WITHOUT it is a converted thesis set
      (``golden/converted/feature_sets/*.json``): those were selected on the
      legacy ``expdecay7`` columns, so they are accepted only under
      ``series.window.expdecay = "legacy_alpha"``, and only for the selector
      (``name``) and target they were made for.

    Raises :class:`FeatureCacheMismatch`.
    """
    window = cfg.series.window
    provenance = payload.get("provenance")
    if provenance is None:
        if window.expdecay != "legacy_alpha":
            raise FeatureCacheMismatch(
                f"{source} is a thesis feature selection without provenance: it was made on "
                "the LEGACY expdecay7 window features and is only valid with "
                f"series.window.expdecay=legacy_alpha (this run: {window.expdecay!r}). Use "
                f"`legacy={cfg.name}` to reproduce the thesis, or run `strikecast featsel "
                f"experiment={cfg.name}` to select on the current features (audit A13)."
            )
        name = payload.get("name")
        if name and not str(cfg.feature_selection.selector).startswith(str(name)):
            raise FeatureCacheMismatch(
                f"{source} is the {name!r} selection but this run's selector is "
                f"{cfg.feature_selection.selector!r}"
            )
        targets = payload.get("target_components")
        if targets and cfg.data.target not in targets:
            raise FeatureCacheMismatch(
                f"{source} was selected for target(s) {targets}, not {cfg.data.target!r}"
            )
        return

    expected = {
        "window": window.model_dump(mode="json"),
        "feature_selection": _fs_identity(cfg.feature_selection),
        "common_kwargs": cfg.common_kwargs.model_dump(mode="json"),
        "target": cfg.data.target,
    }
    if cfg.feature_selection.protocol == "figure":
        # the figure's diff_l2 target follows the transform (Q8 resolved)
        expected["transform"] = cfg.transform.kind
    diffs = [key for key, want in expected.items() if provenance.get(key) != want]
    if diffs:
        detail = "; ".join(
            f"{key}: cached {provenance.get(key)!r} != run {expected[key]!r}" for key in diffs
        )
        raise FeatureCacheMismatch(
            f"{source} was selected under a different configuration ({detail}). Re-run "
            f"`strikecast featsel experiment={cfg.name}` for this configuration (audit A13)."
        )


def require_pythonhashseed_zero(what: str = "feature selection") -> None:
    """Hard-fail unless ``PYTHONHASHSEED=0`` (F16, A12, D2).

    The upstream covariate column order -- and so LightGBM's tie-breaking --
    depends on the hash seed; a selection made under another seed is a
    different selection.
    """
    seed = os.environ.get("PYTHONHASHSEED")
    if seed != "0":
        raise RuntimeError(
            f"{what} requires PYTHONHASHSEED=0 (got {seed!r}); the covariate column order "
            "depends on the hash seed (F16). Re-run as `PYTHONHASHSEED=0 strikecast ...`."
        )


def build_or_load_features(
    cfg: ExperimentConfig,
    store: RunStore | None,
    bundle: SeriesBundle,
    *,
    panel_hash: str | None = None,
    series_hash: str | None = None,
    force: bool = False,
    compute: bool | None = None,
) -> FeatureSets:
    """The selected covariate components, from cache or from a fresh ranking.

    Resolution order:

    1. ``feature_selection.cache_path`` -- the converted legacy set. This is the
       only way to reproduce the thesis' own features (F16), so it wins over
       everything.
    2. ``shared/feature_selection.<hash>.json`` written by a previous run
       (typically ``strikecast featsel``), when ``feature_selection.cache`` is on.
    3. a fresh :func:`strikecast.data.feature_selection.select_top_k` -- only
       when ``compute`` allows it. ``compute=None`` (the default) means "unless
       ``feature_selection.require_cached``"; the ``featsel`` command passes
       ``True``. Otherwise a missing cache raises :class:`FeatureSelectionMissing`.

    Both caches go through :func:`check_selection_provenance` (A13), and under
    the figure protocol both must carry ``past_lags`` (schema 2), which every
    returned :class:`FeatureSets` then holds -- the store-hit path included.

    The ``zipoisson_regressor`` and ``hurdle_tweedie_pos`` selectors are fitted
    with **positive-only sample weights** (F27, Q7): ``final_hurdle.ipynb``
    cell 10 is the only legacy selection that passes ``sample_weight``, and it
    passes ``[w.slice_intersect(ts) for w, ts in zip(full_weights,
    train_target_r)]``. Reproduced here from the bundle's own lists, so a
    caller cannot forget it.

    The target is ``bundle.target_train`` -- the LEVEL target, also for the
    legacy ``diffreg`` (Q8) -- except for a figure selector under
    ``transform.kind == "diff"`` (``diff_l2``), which is fitted on the
    DIFFERENCED ``model_space_parts(bundle)["target_train"]``, the target the
    diff models actually train on.
    """
    store = _store_for(cfg, store)
    panel_hash = panel_hash or _panel_hash(cfg)
    series_hash = series_hash or _series_hash(cfg, panel_hash)
    digest = _features_hash(cfg, panel_hash, series_hash)
    path = store.shared_path(cfg.name, "feature_selection", digest)
    protocol = cfg.feature_selection.protocol

    cache_path = cfg.feature_selection.cache_path
    if cache_path:
        legacy = Path(cache_path).expanduser()
        if not legacy.is_file():
            raise FileNotFoundError(
                f"feature_selection.cache_path={cache_path!r} does not exist; unset it to "
                "select features afresh (F16)"
            )
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        check_selection_provenance(cfg, payload, legacy)
        sets = _sets_from_payload(payload, protocol, legacy)
        if sets is None:
            raise ValueError(
                f"{legacy} holds neither past_keep/future_keep nor "
                "past_covariate_components/future_covariate_components"
            )
        logger.info("feature selection: loaded the cached legacy set %s", legacy)
        return FeatureSets(sets[0], sets[1], "cache_path", digest, legacy, past_lags=sets[2])

    if cfg.feature_selection.cache and not force and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        check_selection_provenance(cfg, payload, path)
        sets = _sets_from_payload(payload, protocol, path)
        if sets is not None:
            logger.info("feature selection: cache hit %s", path)
            return FeatureSets(sets[0], sets[1], "store", digest, path, past_lags=sets[2])

    allowed = (not cfg.feature_selection.require_cached) if compute is None else compute
    if not allowed:
        raise FeatureSelectionMissing(
            f"no cached feature selection for experiment {cfg.name!r} (target "
            f"{cfg.data.target!r}, selector {cfg.feature_selection.selector!r}, "
            f"expdecay {cfg.series.window.expdecay!r}) at {path}. "
            "feature_selection.require_cached is on, so cv/tune/test never re-select: run "
            f"`PYTHONHASHSEED=0 strikecast featsel experiment={cfg.name}` (with the same "
            "overrides and --store-root) first (audit A12/D2)."
        )
    if cfg.feature_selection.deterministic:
        require_pythonhashseed_zero()

    from strikecast.data.feature_selection import select_top_k  # noqa: PLC0415

    target_train = bundle.target_train
    if protocol == "figure" and cfg.transform.kind == "diff":
        from strikecast.data.series import model_space_parts  # noqa: PLC0415

        target_train = model_space_parts(bundle)["target_train"]

    selection = select_top_k(
        target_train,
        bundle.past_covs,
        bundle.future_covs,
        cfg.feature_selection.build(),
        sample_weight=_selection_sample_weight(cfg, bundle),
        protocol=protocol,
    )
    if cfg.feature_selection.cache:
        selection.to_json(
            path, provenance=selection_provenance(cfg, panel_hash, series_hash, digest)
        )
        ranking = selection.write_ranking_csv(ranking_csv_path(path))
        logger.info("feature selection: wrote %s (+ %s)", path, ranking.name)
    return FeatureSets(
        list(selection.past_keep),
        list(selection.future_keep),
        "computed",
        digest,
        path if cfg.feature_selection.cache else None,
        past_lags=selection.past_lags,
    )


def ranking_csv_path(selection_json: Path) -> Path:
    """``feature_selection.<hash>.json`` -> ``feature_selection.<hash>.ranking.csv``."""
    name = selection_json.name
    stem = name[: -len(".json")] if name.endswith(".json") else name
    return selection_json.with_name(f"{stem}.ranking.csv")


def select_features(
    cfg: ExperimentConfig,
    store: RunStore | None = None,
    *,
    inputs: Inputs | None = None,
    force: bool = False,
) -> dict[str, FeatureSets]:
    """``strikecast featsel``: compute and cache every selection ONCE.

    One selection per head (hurdle: ``regressor``, ``classifier``), per damage
    key, or -- for count/diff -- per DISTINCT selection the models route to
    (``ExperimentConfig.selection_groups``: count = ``count_tweedie`` as the
    default plus ``poisson``; diff = ``diff_l2``), each written to
    ``<store>/<experiment>/shared/feature_selection.<hash>.json`` with its
    provenance and ranking CSV. Existing selections are loaded (and
    provenance-checked), not recomputed, unless ``force``. Requires
    ``PYTHONHASHSEED=0`` whatever the ``deterministic`` flag says (A12, D2).
    Returns ``head -> FeatureSets``: the composite head name, ``""`` for the
    top-level selection, or the ``feature_selections`` key of any other group.
    """
    require_pythonhashseed_zero("`strikecast featsel`")
    store = _store_for(cfg, store)
    if inputs is None:
        from strikecast.data.load import load_inputs  # noqa: PLC0415

        inputs = load_inputs(cfg.data.fixed_dir, cfg.data.dataset_dir)

    if is_composite_family(cfg):
        targets = head_configs(cfg)
    else:
        targets = {key or "": cfg.for_selection(key) for key in cfg.selection_groups()}
        if not targets:  # an experiment without models still has its own selection
            targets = {"": cfg}

    out: dict[str, FeatureSets] = {}
    for name, head_cfg in targets.items():
        if not head_cfg.feature_selection.cache and not head_cfg.feature_selection.cache_path:
            raise ValueError(
                f"experiment {cfg.name!r} head {name!r}: feature_selection.cache is off, so "
                "a precomputed selection could never be read back"
            )
        panel, weather_cols, panel_hash = build_or_load_panel(head_cfg, store, inputs=inputs)
        bundle, series_hash = build_or_load_bundle(
            head_cfg,
            store,
            panel,
            weather_cols,
            dict(inputs.activity_by_region),
            panel_hash=panel_hash,
        )
        out[name] = build_or_load_features(
            head_cfg,
            store,
            bundle,
            panel_hash=panel_hash,
            series_hash=series_hash,
            force=force,
            compute=True,
        )
        logger.info(
            "featsel %s head %r: %s (%d past, %d future) -> %s",
            cfg.name,
            name,
            out[name].source,
            len(out[name].past_keep),
            len(out[name].future_keep),
            out[name].path,
        )
    return out


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
    compute_features: bool | None = None,
) -> DataArtifacts:
    """Run the whole data stage and return everything downstream needs.

    ``compute_features`` is :func:`build_or_load_features`' ``compute``: ``None``
    computes a missing selection unless ``feature_selection.require_cached``.

    A composite family (hurdle, damage -- see :func:`is_composite_family`) gets a
    :class:`CompositeData`, which IS a :class:`DataArtifacts` carrying the
    primary head's artefacts plus every head under ``.heads``. Callers that only
    need one bundle therefore need no change.
    """
    if cfg.data.panel_variant == "chronos":
        # Audit C15: Chronos-2 takes the AutoGluon frame -- no darts bundle, no
        # feature selection, no window features (F28, F123).
        from strikecast.pipeline.chronos_stage import prepare_chronos_data  # noqa: PLC0415

        return prepare_chronos_data(cfg, store, inputs=inputs)
    if is_composite_family(cfg):
        return prepare_composite_data(
            cfg,
            store,
            inputs=inputs,
            cache=cache,
            force=force,
            keep_panel=keep_panel,
            compute_features=compute_features,
        )
    return _prepare_one(
        cfg,
        store,
        inputs=inputs,
        cache=cache,
        force=force,
        keep_panel=keep_panel,
        compute_features=compute_features,
    )


def prepare_by_selection(
    cfg: ExperimentConfig,
    store: RunStore | None = None,
    models: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str | None, tuple[ExperimentConfig, DataArtifacts, list[str]]]:
    """:func:`prepare_data` once per distinct selection the ``models`` route to.

    ``key -> (cfg.for_selection(key), its DataArtifacts, the model names)``, in
    the order of ``ExperimentConfig.selection_groups(models)`` (``None`` = the
    top-level selection). A composite family (hurdle, damage) has no per-model
    keys, so it is one ``None`` group holding its :class:`CompositeData`.
    ``kwargs`` are :func:`prepare_data`'s; ``inputs`` is loaded once and shared
    when there is more than one group.
    """
    store = _store_for(cfg, store)
    groups = cfg.selection_groups(models)
    if (
        len(groups) > 1
        and kwargs.get("inputs") is None
        and cfg.data.panel_variant != "chronos"
    ):
        from strikecast.data.load import load_inputs  # noqa: PLC0415

        kwargs["inputs"] = load_inputs(cfg.data.fixed_dir, cfg.data.dataset_dir)

    out: dict[str | None, tuple[ExperimentConfig, DataArtifacts, list[str]]] = {}
    for key, names in groups.items():
        sub = cfg.for_selection(key)
        logger.info(
            "data stage: selection %r (%s) for %d model(s)",
            key,
            sub.feature_selection.selector,
            len(names),
        )
        out[key] = (sub, prepare_data(sub, store, **kwargs), list(names))
    return out


def prepare_composite_data(
    cfg: ExperimentConfig,
    store: RunStore | None = None,
    *,
    inputs: Inputs | None = None,
    cache: bool = True,
    force: bool = False,
    keep_panel: bool = False,
    compute_features: bool | None = None,
) -> CompositeData:
    """Run the data stage once per head and collect the results.

    Every head goes through the SAME ``shared/`` cache; the hashes differ
    because :func:`head_configs` narrows ``data`` and ``feature_selection`` per
    head and both are hashed. Two runs of the same head therefore hit the cache,
    and no head can ever read another head's panel.

    ``inputs`` is loaded once and reused by every head: ``load_inputs`` reads
    three small files and ``build_panel`` never mutates what it is given (panel
    quirk Q1).
    """
    store = _store_for(cfg, store)
    heads = head_configs(cfg)
    if inputs is None:
        from strikecast.data.load import load_inputs  # noqa: PLC0415

        inputs = load_inputs(cfg.data.fixed_dir, cfg.data.dataset_dir)

    built: dict[str, DataArtifacts] = {}
    for name, head_cfg in heads.items():
        logger.info("data stage: head %r (target %r)", name, head_cfg.data.target)
        built[name] = _prepare_one(
            head_cfg,
            store,
            inputs=inputs,
            cache=cache,
            force=force,
            keep_panel=keep_panel,
            compute_features=compute_features,
        )

    primary = HURDLE_PRIMARY_HEAD if HURDLE_PRIMARY_HEAD in built else next(iter(built))
    lead = built[primary]
    return CompositeData(
        bundle=lead.bundle,
        features=lead.features,
        panel_hash=lead.panel_hash,
        series_hash=lead.series_hash,
        activity_by_region=dict(lead.activity_by_region),
        panel=lead.panel,
        heads=built,
        primary=primary,
        family=cfg.name,
    )


def _prepare_one(
    cfg: ExperimentConfig,
    store: RunStore | None = None,
    *,
    inputs: Inputs | None = None,
    cache: bool = True,
    force: bool = False,
    keep_panel: bool = False,
    compute_features: bool | None = None,
) -> DataArtifacts:
    """The single-panel data stage: panel -> bundle -> selection -> subset."""
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
        cfg,
        store,
        bundle,
        panel_hash=panel_hash,
        series_hash=series_hash,
        force=force,
        compute=compute_features,
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
