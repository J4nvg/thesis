"""Darts ``TimeSeries`` construction, window transforms, static encoding and splits.

This module replaces the legacy trio
``src/ts_specific_tools.py::build_ts_and_apply_window_transformer``,
``::get_covs_and_encodings`` and ``::split_series_list`` (plus the
``raw_past_covs_list`` block and the *second* ``get_covs_and_encodings`` call
that every notebook makes for the RNN raw covariates) with a single
:class:`SeriesBundle`.

Behaviour-preserving: the operations, their order, and every keyword argument
are exactly the legacy ones. Legacy oddities are documented, never fixed:

* ``Q1`` ``TRAIN_VAL_END`` is ``0.70 + 0.10 == 0.7999999999999999`` in binary
  floating point, and ``split_before`` is called with that value, not with
  ``0.8``. :attr:`SplitConfig.train_val_end` reproduces the same expression so
  the CV view keeps its exact length. A consequence worth knowing: the CV view
  is *shorter* than train + val. On the real panel (847 days) train is 593 and
  val is 85 (678 points), while ``split_before(TRAIN_VAL_END)`` yields 676 --
  the CV view stops two days before the end of val, so the last two validation
  days are never rolled over. Preserved, not fixed.
* ``Q2`` the 70/10/20 split is done with two relative ``split_after`` calls
  (``0.7`` then ``1/3`` of the remainder), so the realised test fraction depends
  on rounding, not on ``SplitConfig.test``. ``SplitConfig`` fractions are
  documentation of the intent plus the source of the two derived fractions.
* ``Q3`` the static-covariate encoders are **three separate**
  ``StaticCovariatesTransformer`` instances (target / past / future), each fit on
  its own list, and a **fourth** one is fit on the un-windowed raw past
  covariates by the second legacy ``get_covs_and_encodings`` call. Nothing is
  shared, so the ``region`` label encoding is fit independently four times.
* ``Q4`` the encoders are fit on the **full-length** series (train + val + test),
  before any split. This is a leak of the region/activity label encoding only,
  and is preserved (see plan §4, in the spirit of F1).
* ``Q5`` after feature selection the notebooks ``subset_safe`` the *un-encoded*
  lists and then re-run the whole encoding + split. :func:`subset_components`
  reproduces that, which is why the bundle carries the un-encoded lists in
  :class:`RawSeries`.
* ``Q6`` ``subset_safe`` is called with a ``set`` of wanted names, so the
  resulting component order follows the iteration order of that set. We iterate
  the argument exactly as given rather than sorting it.
* ``Q7`` the second legacy ``get_covs_and_encodings`` call (for the RNN raw past
  covariates) happens *after* feature selection and is passed the already
  subset future covariates. Only its ``full_past_covs`` return value is used, so
  the other arguments have no effect on any downstream result; we therefore
  build the raw past covariates once, from the un-subset raw list.

Design note on the un-encoded lists: they are kept in an explicit
:class:`RawSeries` dataclass on ``SeriesBundle.raw`` rather than as three
private ``_*_raw`` fields. They are genuinely a coherent group (the inputs the
legacy code re-feeds into ``get_covs_and_encodings``), they are serialised, and
a named object documents itself better than three underscored fields.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer, WindowTransformer

from strikecast.config.schema import SeriesConfig, SplitConfig

__all__ = [
    "RawSeries",
    "SeriesBundle",
    "build_bundle",
    "load",
    "model_space_parts",
    "positive_only_weights",
    "serialise",
    "split_series_list",
    "subset_components",
    "to_autogluon_frame",
]

SERIALISATION_VERSION = 1

# The lists that are persisted, in (attribute name, file stem) form.
_BUNDLE_LISTS: tuple[tuple[str, str], ...] = (
    ("target_full", "target_full"),
    ("target_train", "target_train"),
    ("target_val", "target_val"),
    ("target_test", "target_test"),
    ("target_cv_view", "target_cv_view"),
    ("past_covs", "past_covs"),
    ("raw_past_covs", "raw_past_covs"),
    ("future_covs", "future_covs"),
)
_RAW_LISTS: tuple[tuple[str, str], ...] = (
    ("target", "raw_target"),
    ("past", "raw_past"),
    ("future", "raw_future"),
)


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
@dataclass
class RawSeries:
    """The un-encoded (pre ``StaticCovariatesTransformer``) series lists.

    ``target``/``past``/``future`` are what the legacy notebooks keep in
    ``target_series_list`` / ``past_covs_list`` / ``future_covs_list`` and
    re-feed into ``get_covs_and_encodings`` after feature selection; ``past`` is
    the *windowed* past covariate list and ``raw_past`` the un-windowed one used
    by the RNNs.
    """

    target: list[TimeSeries]
    past: list[TimeSeries]
    future: list[TimeSeries]
    raw_past: list[TimeSeries]


@dataclass
class SeriesBundle:
    """Replaces the legacy 9-tuple returned by ``get_covs_and_encodings``."""

    region_names: list[str]
    target_full: list[TimeSeries]
    target_train: list[TimeSeries]
    target_val: list[TimeSeries]
    target_test: list[TimeSeries]
    target_cv_view: list[TimeSeries]
    past_covs: list[TimeSeries]
    raw_past_covs: list[TimeSeries]
    future_covs: list[TimeSeries]
    train_val_end: float
    cv_start_frac: float
    fractions: SplitConfig
    activity_by_region: dict[str, Any] = field(default_factory=dict)
    raw: RawSeries | None = None

    def __len__(self) -> int:
        return len(self.region_names)


# --------------------------------------------------------------------------- #
# legacy primitives
# --------------------------------------------------------------------------- #
def split_series_list(
    series_list: Sequence[TimeSeries],
) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
    """Exact copy of ``src.ts_specific_tools.split_series_list``."""
    train_list: list[TimeSeries] = []
    val_list: list[TimeSeries] = []
    test_list: list[TimeSeries] = []

    for ts in series_list:
        train, temp = ts.split_after(0.7)
        val, test = temp.split_after(1 / 3)
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    return train_list, val_list, test_list


def positive_only_weights(target_list: Sequence[TimeSeries]) -> list[TimeSeries]:
    """Copy of ``src.ts_specific_tools.make_positive_only_weights``.

    Weight 1.0 where ``y > 0``, 0.0 where ``y == 0``. Keeps the time index.
    """
    weights: list[TimeSeries] = []
    for ts in target_list:
        vals = ts.values().ravel()
        w = (vals > 0).astype(float)
        weights.append(
            TimeSeries.from_times_and_values(
                ts.time_index, w, static_covariates=ts.static_covariates
            )
        )
    return weights


def _encode_statics(series_list: Sequence[TimeSeries]) -> list[TimeSeries]:
    """One freshly fit ``StaticCovariatesTransformer`` per list (Q3)."""
    return cast(
        "list[TimeSeries]", StaticCovariatesTransformer().fit_transform(list(series_list))
    )


def _region_name(ts: TimeSeries) -> str:
    sc = ts.static_covariates
    if sc is None:
        raise ValueError("target series carry no static covariates; expected a 'region' column")
    return sc["region"].iloc[0]


def _encode_and_split(
    target_series_list: Sequence[TimeSeries],
    past_covs_list: Sequence[TimeSeries],
    future_covs_list: Sequence[TimeSeries],
    split: SplitConfig,
) -> dict[str, Any]:
    """The body of the legacy ``get_covs_and_encodings`` (minus the prints)."""
    # Capture names BEFORE encoding -- afterwards `region` is a float code.
    region_names = [_region_name(ts) for ts in target_series_list]

    target_encoded = _encode_statics(target_series_list)
    past_encoded = _encode_statics(past_covs_list)
    future_encoded = _encode_statics(future_covs_list)

    train_target, val_target, test_target = split_series_list(target_encoded)

    train_val_end = split.train_val_end  # 0.7999999999999999 (Q1)
    cv_start_frac = split.cv_start_frac  # 0.875
    target_for_cv = [ts.split_before(train_val_end)[0] for ts in target_encoded]

    return {
        "region_names": region_names,
        "target_full": target_encoded,
        "target_train": train_target,
        "target_val": val_target,
        "target_test": test_target,
        "target_cv_view": target_for_cv,
        "past_covs": past_encoded,
        "future_covs": future_encoded,
        "train_val_end": train_val_end,
        "cv_start_frac": cv_start_frac,
    }


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_bundle(
    panel: pd.DataFrame,
    target: str,
    past_covariates: list[str],
    future_covariates: list[str],
    config: SeriesConfig,
    activity_by_region: Mapping[str, Any] | None = None,
) -> SeriesBundle:
    """Build every series list the pipeline needs, exactly as the legacy code does.

    Steps, in the legacy order:

    1. ``TimeSeries.from_group_dataframe`` for target (with ``static_cols``),
       past covariates and future covariates;
    2. ``WindowTransformer`` over the past covariates (the six legacy windows);
    3. ``from_group_dataframe`` again for the *un-windowed* past covariates
       (``raw_past_covs_list`` in the notebooks);
    4. ``get_covs_and_encodings``: region names before encoding, three separate
       static encoders, ``split_series_list``, ``split_before(TRAIN_VAL_END)``;
    5. a fourth static encoder for the raw past covariates (the notebooks'
       second ``get_covs_and_encodings`` call, whose ``full_past_covs`` return
       value becomes ``full_raw_past_covs_LSTM``).
    """
    group_col, time_col = config.group_col, config.time_col

    target_series_list = TimeSeries.from_group_dataframe(
        panel,
        group_cols=group_col,
        time_col=time_col,
        value_cols=target,
        static_cols=list(config.static_cols),
    )
    past_covs_raw = TimeSeries.from_group_dataframe(
        panel,
        group_cols=group_col,
        time_col=time_col,
        value_cols=list(past_covariates),
    )
    future_covs_list = TimeSeries.from_group_dataframe(
        panel,
        group_cols=group_col,
        time_col=time_col,
        value_cols=list(future_covariates),
    )

    window_transformer = WindowTransformer(**config.window.transformer_kwargs())
    past_covs_list = cast("list[TimeSeries]", window_transformer.transform(past_covs_raw))

    # The RNN branch re-reads the un-windowed past covariates from the panel.
    raw_past_covs_list = TimeSeries.from_group_dataframe(
        panel,
        group_cols=group_col,
        time_col=time_col,
        value_cols=list(past_covariates),
    )

    parts = _encode_and_split(target_series_list, past_covs_list, future_covs_list, config.split)
    raw_past_encoded = _encode_statics(raw_past_covs_list)  # Q3/Q7

    return SeriesBundle(
        raw_past_covs=raw_past_encoded,
        fractions=config.split,
        activity_by_region=dict(activity_by_region or {}),
        raw=RawSeries(
            target=list(target_series_list),
            past=list(past_covs_list),
            future=list(future_covs_list),
            raw_past=list(raw_past_covs_list),
        ),
        **parts,
    )


def subset_components(
    bundle: SeriesBundle,
    past_keep: Iterable[str],
    future_keep: Iterable[str],
) -> SeriesBundle:
    """Reproduce the post-feature-selection step of every notebook.

    ``subset_safe`` is applied to the **un-encoded** past and future lists and
    the whole encoding + split is then re-run (the notebooks literally re-call
    ``get_covs_and_encodings(target_series_list, past_covs_list,
    future_covs_list, ...)``), which means four brand new static encoders.

    ``past_keep``/``future_keep`` are iterated exactly as given: the legacy call
    passes a ``set``, so the resulting component order is that set's iteration
    order (Q6).
    """
    if bundle.raw is None:
        raise ValueError(
            "subset_components needs the un-encoded lists; this bundle has raw=None"
        )

    past_subset = [_subset_safe(ts, past_keep) for ts in bundle.raw.past]
    future_subset = [_subset_safe(ts, future_keep) for ts in bundle.raw.future]

    parts = _encode_and_split(
        bundle.raw.target, past_subset, future_subset, bundle.fractions
    )
    raw_past_encoded = _encode_statics(bundle.raw.raw_past)

    return SeriesBundle(
        raw_past_covs=raw_past_encoded,
        fractions=bundle.fractions,
        activity_by_region=dict(bundle.activity_by_region),
        raw=replace(bundle.raw, past=past_subset, future=future_subset),
        **parts,
    )


def model_space_parts(bundle: SeriesBundle) -> dict[str, Any]:
    """The diff family's MODEL-SPACE lists: difference first, then encode + split.

    ``Diff().forward(bundle.raw.target)`` on the un-encoded FULL target list,
    then the legacy encode+split (:func:`_encode_and_split`) with the bundle's
    own un-encoded past/future lists and fractions -- literally
    ``_diff_regression.py`` lines 234-239. Differencing the full list before
    the split is what makes the CV view one step longer than
    ``Diff.forward(level CV view)`` (F80). Two consumers share it:

    * ``run_stage``'s F80 model-space CV target (``["target_cv_view"]``);
    * the figure's ``diff_l2`` selector, fitted on ``["target_train"]``, the
      differenced target the diff models train on (resolves Q8; plan
      2026-09-28). The legacy ``diffreg`` keeps the level target.

    Returns the same dict as :func:`_encode_and_split`.
    """
    if bundle.raw is None:
        raise ValueError(
            "model_space_parts needs the bundle's un-encoded lists (F80); this bundle "
            "has raw=None"
        )
    from strikecast.transforms.diff import Diff  # noqa: PLC0415

    diffed = Diff().forward(list(bundle.raw.target))
    return _encode_and_split(diffed, bundle.raw.past, bundle.raw.future, bundle.fractions)


def _subset_safe(ts: TimeSeries, wanted: Iterable[str]) -> TimeSeries:
    """Copy of ``src.general_tools.subset_safe`` (order follows ``wanted``)."""
    available = set(ts.components)
    return ts[[c for c in wanted if c in available]]


# --------------------------------------------------------------------------- #
# AutoGluon adapter (not implemented: AutoGluon cannot be installed, see
# pyproject.toml)
# --------------------------------------------------------------------------- #
def to_autogluon_frame(bundle: SeriesBundle, *args: Any, **kwargs: Any) -> Any:
    """Stub for the Chronos-2 adapter. **Not implemented.**

    ``autogluon.timeseries`` cannot be installed alongside the frozen core
    (pandas 3.0.2), so this adapter is intentionally left unimplemented. The
    legacy logic it must reproduce is ``_chronos2.py`` §3, verbatim::

        ts_df = (
            for_global
            .reset_index()
            .rename(columns={"region": "item_id", "event_date": "timestamp"})
            .sort_values(["item_id", "timestamp"])
            .set_index(["item_id", "timestamp"])
        )
        static_df = (
            ts_df.reset_index()[["item_id", "Activity_Level"]]
                 .drop_duplicates()
                 .set_index("item_id")
        )
        ts_df = ts_df.drop(columns=["Activity_Level"])
        tsdf = TimeSeriesDataFrame(ts_df, static_features=static_df)

        n = int(tsdf.num_timesteps_per_item().min())
        test_size = int(round(TEST_FRAC * n))
        train_data, test_data_ag = tsdf.train_test_split(prediction_length=test_size)

    Note that Chronos builds its frame from the **panel**, not from the darts
    series, and splits by ``round(0.2 * n)`` steps rather than by
    ``split_after`` fractions -- flag F3 in the refactor plan. Both details must
    survive whenever this is implemented.
    """
    raise NotImplementedError(
        "to_autogluon_frame requires autogluon.timeseries, which is not installable "
        "against the frozen core (see pyproject.toml). Legacy logic is in the docstring."
    )


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
def _statics_record(ts: TimeSeries) -> dict[str, Any]:
    sc = ts.static_covariates
    if sc is None:
        return {}
    return {
        "index_name": sc.index.name,
        "index": [str(i) for i in sc.index],
        "columns": [str(c) for c in sc.columns],
        "dtypes": {str(c): str(sc[c].dtype) for c in sc.columns},
        "values": {str(c): sc[c].tolist() for c in sc.columns},
    }


def _statics_frame(record: Mapping[str, Any]) -> pd.DataFrame | None:
    if not record:
        return None
    df = pd.DataFrame({c: record["values"][c] for c in record["columns"]})
    df.index = pd.Index(record["index"], name=record["index_name"])
    for col, dtype in record["dtypes"].items():
        df[col] = df[col].astype(dtype)
    return df


def _list_to_frame(
    series_list: Sequence[TimeSeries], region_names: Sequence[str], time_col: str
) -> pd.DataFrame:
    frames = []
    for name, ts in zip(region_names, series_list, strict=True):
        df = ts.to_dataframe(copy=True)
        df.index.name = time_col
        df = df.reset_index()
        if "region" in ts.components:
            raise ValueError("component named 'region' collides with the grouping column")
        df.insert(0, "region", name)
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def _frame_to_list(
    df: pd.DataFrame,
    region_names: Sequence[str],
    time_col: str,
    statics: Sequence[Mapping[str, Any]],
    freq: str | None,
) -> list[TimeSeries]:
    out: list[TimeSeries] = []
    value_cols = [c for c in df.columns if c not in ("region", time_col)]
    for name, record in zip(region_names, statics, strict=True):
        sub = df.loc[df["region"] == name, [time_col, *value_cols]].reset_index(drop=True)
        out.append(
            TimeSeries.from_dataframe(
                sub,
                time_col=time_col,
                value_cols=value_cols,
                freq=freq,
                static_covariates=_statics_frame(record),
            )
        )
    return out


def serialise(bundle: SeriesBundle, dir: str | Path, time_col: str = "event_date") -> Path:
    """Write a bundle to ``dir`` as parquet + ``manifest.json``.

    One parquet per series list (all regions concatenated, with a ``region``
    column), the static covariates in the manifest's ``statics`` sidecar, and no
    pickles anywhere.
    """
    out = Path(dir)
    out.mkdir(parents=True, exist_ok=True)

    lists = list(_BUNDLE_LISTS)
    statics: dict[str, list[dict[str, Any]]] = {}
    freqs: dict[str, str | None] = {}

    def _dump(stem: str, series_list: Sequence[TimeSeries]) -> None:
        frame = _list_to_frame(series_list, bundle.region_names, time_col)
        frame.to_parquet(out / f"{stem}.parquet", index=False)
        statics[stem] = [_statics_record(ts) for ts in series_list]
        freq = series_list[0].freq if series_list else None
        freqs[stem] = None if freq is None else getattr(freq, "freqstr", str(freq))

    for attr, stem in lists:
        _dump(stem, getattr(bundle, attr))
    if bundle.raw is not None:
        for attr, stem in _RAW_LISTS:
            _dump(stem, getattr(bundle.raw, attr))
        _dump("raw_raw_past", bundle.raw.raw_past)

    manifest = {
        "version": SERIALISATION_VERSION,
        "time_col": time_col,
        "region_names": list(bundle.region_names),
        "activity_by_region": dict(bundle.activity_by_region),
        "train_val_end": bundle.train_val_end,
        "cv_start_frac": bundle.cv_start_frac,
        "fractions": bundle.fractions.model_dump(),
        "has_raw": bundle.raw is not None,
        "freqs": freqs,
        "statics": statics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def load(dir: str | Path) -> SeriesBundle:
    """Inverse of :func:`serialise`."""
    src = Path(dir)
    manifest = json.loads((src / "manifest.json").read_text())
    time_col = manifest["time_col"]
    region_names = list(manifest["region_names"])
    statics = manifest["statics"]
    freqs = manifest["freqs"]

    def _read(stem: str) -> list[TimeSeries]:
        frame = pd.read_parquet(src / f"{stem}.parquet")
        return _frame_to_list(frame, region_names, time_col, statics[stem], freqs.get(stem))

    kwargs = {attr: _read(stem) for attr, stem in _BUNDLE_LISTS}
    raw = None
    if manifest.get("has_raw"):
        raw = RawSeries(
            target=_read("raw_target"),
            past=_read("raw_past"),
            future=_read("raw_future"),
            raw_past=_read("raw_raw_past"),
        )

    return SeriesBundle(
        region_names=region_names,
        train_val_end=float(manifest["train_val_end"]),
        cv_start_frac=float(manifest["cv_start_frac"]),
        fractions=SplitConfig(**manifest["fractions"]),
        activity_by_region=dict(manifest["activity_by_region"]),
        raw=raw,
        **kwargs,
    )


def _values_equal(a: TimeSeries, b: TimeSeries) -> bool:
    """Small helper used by the tests and by `strikecast verify` later."""
    return bool(np.array_equal(a.values(), b.values())) and a.time_index.equals(b.time_index)
