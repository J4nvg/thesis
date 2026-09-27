"""Stage 4: the cross-model, cross-seed and pairwise tables, written under ``report/``.

Deliberately thin. Everything it writes is computed by the ``evaluation``
modules -- :mod:`strikecast.evaluation.leaderboard` (the seed-42 leaderboard
that reproduces ``golden/results/<family>/leaderboard.csv``),
:mod:`strikecast.evaluation.seeds` (the multi-seed ``leaderboard_ci.csv`` of
plan §7.1), :mod:`strikecast.evaluation.comparison` (the paired differences,
block bootstrap, Diebold-Mariano and Cliff's delta of §7.2) and
:mod:`strikecast.evaluation.stats` (the Friedman/Nemenyi machinery ported from
``analyse_results.ipynb``). This module only decides *which* experiment,
*which* seed, *which* runs and *where*, and renders ``summary.md`` from the
frames the other outputs already produced.

What lands in ``runs/<experiment>/report/`` (plan §7.1, §7.2):

``leaderboard.csv``
    One seed, the legacy column order and sort.
``leaderboard_ci.csv``
    Per ``(model, paradigm, stage, metric)``: mean, SD, t-interval and a
    bootstrap over the evaluation seeds, with the ``deterministic`` marker.
``pairwise_rmse.csv`` / ``pairwise_mae.csv``
    One row per ordered-by-insertion model pair per breakdown (global, per
    horizon, per activity tier): mean loss difference, percent improvement,
    block-bootstrap 95% CI over forecast origins, DM p-value and Cliff's delta.
``family_comparison.csv``
    §7.2 item 5: per-seed metric of family A minus family B on the same seed,
    mean and t-interval, at both the "family as a whole" (mean) and "best
    configuration" (best) levels.
``cd_rmse.svg`` / ``cd_mae.svg``
    Demsar critical-difference diagrams regenerated from the run store.
``summary.md``
    The tables the paper needs: the Table 4 layout extended with ``± seed SD``
    and ``[bootstrap 95% CI]``, the pairwise difference and p-value matrices,
    the family comparison and the rank statistics.

Nothing here touches a training pipeline: every analysis consumes the
``PredictionSet``s and metric views already in the run store (§7).

Notes
-----

``Q1`` **Everything degrades, nothing raises.** A store with one model, one
    seed, no predictions or no stochastic model still produces a report: the
    sections that cannot be computed are skipped, logged, and named as missing
    in ``summary.md``. That is plan §7's requirement, and it is also what the
    laptop store looks like in practice.

``Q2`` **Labels are ``<model>@<paradigm>``.** A pairwise comparison is over
    *runs*, and the same model under two paradigms is two runs with different
    forecasts. Keeping the paradigm in the label is what lets §7.2's "best
    configuration per family" be read off the table.

``Q3`` **One channel per run.** A composite forecaster writes several channels
    (hurdle: ``prob`` / ``count`` / ``hurdle``) and pairing on
    ``(region, origin_date, horizon)`` would fan out across them, so a run is
    narrowed to a single channel first: the caller's ``channel=``, else the
    only channel present, else the first of :data:`PREFERRED_CHANNELS`. A
    multi-channel run that matches none of those is skipped with a warning
    rather than silently pooled.

``Q4`` **The comparison stage defaults to ``test``.** §7.2's numbers are the
    paper's test-window numbers; the CV stage is single-seed by design (§5.4).
    ``comparison_stage=`` overrides it.

``Q5`` **The critical-difference diagram intersects the design.** Cell 30 of
    the notebook asserted a complete ``(region, horizon) x model`` matrix.
    Models backtested on different schedules do not have one, so the blocks
    that are not shared by every model are dropped before the Friedman test and
    the number kept is reported in ``summary.md``. With the thesis' own runs the
    matrix is already complete and nothing is dropped. See plan flag F131.

``Q6`` **The "bootstrap 95% CI" of the Table 4 layout is over seeds.** It comes
    from ``leaderboard_ci.csv``, which bootstraps the *seed* population. §7.1's
    other bootstrap -- the moving-block resample of forecast origins -- exists
    only for pair *differences* today (``pairwise_*.csv``), not per model per
    metric. Flagged as F130; the column is labelled for what it is.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strikecast.pipeline.context import get_spec, resolve_store_root
from strikecast.store import RunStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from strikecast.config.schema import ExperimentConfig

__all__ = ["LOSS_METRICS", "PREFERRED_CHANNELS", "STAGES", "render_summary", "report"]

logger = logging.getLogger(__name__)

#: The stages a report covers.
STAGES: tuple[str, ...] = ("cv", "test")

#: ``report/pairwise_<name>.csv`` and ``report/cd_<name>.svg`` per loss. The
#: names are the metrics the thesis quotes; the losses are the ones plan §7.2
#: asks for ("for squared and absolute error").
LOSS_METRICS: dict[str, str] = {"rmse": "squared", "mae": "absolute"}

#: Which channel of a multi-channel run to compare on, in order (Q3).
PREFERRED_CHANNELS: tuple[str, ...] = ("y_pred", "hurdle")

#: Directories under ``runs/<experiment>/`` that are not models.
_RESERVED = frozenset({"shared", "tuning", "report"})


def _stochastic_map(cfg: ExperimentConfig) -> dict[str, bool]:
    """``{model: spec.stochastic}``: which models the seed sweep applies to (§5.4)."""
    out: dict[str, bool] = {}
    for name in cfg.model_names:
        try:
            out[name] = bool(get_spec(name, cfg.name).stochastic)
        except (KeyError, ImportError) as exc:  # a family whose deps are absent
            logger.debug("no spec for %s: %s", name, exc)
    return out


# --------------------------------------------------------------------------- #
# run discovery and prediction frames
# --------------------------------------------------------------------------- #
def _discover_runs(
    store: RunStore,
    experiment: str,
    *,
    seed: int,
    models: Sequence[str] | None = None,
    paradigms: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """``[(model, paradigm)]`` that have a ``seed=<seed>`` directory, sorted."""
    exp_dir = store.experiment_dir(experiment)
    if not exp_dir.is_dir():
        return []
    model_set = None if models is None else set(models)
    paradigm_set = None if paradigms is None else set(paradigms)
    found: list[tuple[str, str]] = []
    for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        if model_dir.name in _RESERVED:
            continue
        if model_set is not None and model_dir.name not in model_set:
            continue
        for paradigm_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if paradigm_set is not None and paradigm_dir.name not in paradigm_set:
                continue
            if (paradigm_dir / f"seed={int(seed)}").is_dir():
                found.append((model_dir.name, paradigm_dir.name))
    return found


def _select_channel(frame: pd.DataFrame, label: str, channel: str | None) -> pd.DataFrame | None:
    """Narrow a run's predictions to one channel (Q3); ``None`` when ambiguous."""
    channels = [str(c) for c in frame["channel"].unique()]
    if channel is not None and channel in channels:
        chosen = channel
    elif len(channels) == 1:
        chosen = channels[0]
    else:
        chosen = next((c for c in PREFERRED_CHANNELS if c in channels), "")
        if not chosen:
            logger.warning(
                "%s has channels %s and none is a comparison default; skipping it",
                label,
                sorted(channels),
            )
            return None
    return frame[frame["channel"] == chosen].reset_index(drop=True)


def _prediction_frames(
    store: RunStore,
    experiment: str,
    *,
    stage: str,
    seed: int,
    models: Sequence[str] | None = None,
    paradigms: Sequence[str] | None = None,
    channel: str | None = None,
) -> dict[str, pd.DataFrame]:
    """``{"<model>@<paradigm>": long prediction frame}`` for one stage and seed (Q2)."""
    from strikecast.store import RunKey  # noqa: PLC0415

    frames: dict[str, pd.DataFrame] = {}
    for model, paradigm in _discover_runs(
        store, experiment, seed=seed, models=models, paradigms=paradigms
    ):
        key = RunKey(experiment, model, paradigm, int(seed))
        if not store.part_paths(key, stage):
            continue
        frame = store.load_predictions(key, stage).frame
        if frame.empty:
            continue
        label = f"{model}@{paradigm}"
        narrowed = _select_channel(frame, label, channel)
        if narrowed is not None and not narrowed.empty:
            frames[label] = narrowed
    return frames


def _activity_map(cfg: ExperimentConfig, store: RunStore) -> dict[str, Any] | None:
    """The ``{region: activity level}`` map, for the per-tier breakdown.

    The run store is the source of truth (plan §1), so the series bundle's
    ``manifest.json`` is read first; the fixed-data JSON the data stage itself
    loads from is the fallback. ``None`` when neither is there, which drops the
    tier breakdown and nothing else.
    """
    shared = store.experiment_dir(cfg.name) / "shared"
    if shared.is_dir():
        for bundle in sorted(shared.glob("series.*"), reverse=True):
            manifest = bundle / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - unreadable
                logger.debug("cannot read %s: %s", manifest, exc)
                continue
            activity = payload.get("activity_by_region")
            if activity:
                return dict(activity)
    path = Path(cfg.data.fixed_dir) / "regions_activity_cat.json"
    if path.is_file():
        try:
            from strikecast.data._legacy_helpers import (  # noqa: PLC0415
                load_region_activity_dictionary,
            )

            return dict(load_region_activity_dictionary(str(path)))
        except Exception as exc:  # noqa: BLE001 - a missing fixture is not an error here
            logger.debug("cannot read %s: %s", path, exc)
    logger.info("no activity map found; the per-tier breakdown is skipped")
    return None


# --------------------------------------------------------------------------- #
# the outputs
# --------------------------------------------------------------------------- #
def _write_leaderboard(
    store: RunStore, cfg: ExperimentConfig, seed: int, stages: Sequence[str], out_dir: Path
) -> tuple[pd.DataFrame | None, Path | None]:
    """``report/leaderboard.csv`` for one seed, or the fallback reader."""
    try:
        from strikecast.evaluation.leaderboard import (  # noqa: PLC0415
            build_leaderboard_from_rows,
            collect_metric_rows,
            write_leaderboard,
        )
    except ImportError as exc:
        logger.info("evaluation.leaderboard is not available (%s); using the fallback", exc)
        frame = _fallback_leaderboard(store, cfg.name, seed, stages)
        path = out_dir / "leaderboard.csv"
        frame.to_csv(path, index=False)
        return frame, path

    rows = collect_metric_rows(store.root, experiment=cfg.name, stages=stages, seeds=[int(seed)])
    if not rows:
        logger.warning(
            "no %s metrics under %s for seed %d; the leaderboard is empty",
            cfg.name,
            store.root,
            seed,
        )
    try:
        frame = build_leaderboard_from_rows(rows)
    except KeyError as exc:
        # A metric set without `MASE_mean` (the classification views) cannot
        # take the legacy sort. Report what is there rather than nothing.
        logger.warning("cannot sort the leaderboard (%s); falling back to the raw reader", exc)
        frame = _fallback_leaderboard(store, cfg.name, seed, stages)
    return frame, write_leaderboard(frame, out_dir / "leaderboard.csv")


def _write_pairwise(
    frames: Mapping[str, pd.DataFrame],
    out_dir: Path,
    *,
    activity_by_region: Mapping[str, Any] | None,
    channel: str | None,
    n_boot: int,
    block_length: int,
    alpha: float,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """``report/pairwise_<metric>.csv`` for both losses, on all three breakdowns."""
    from strikecast.evaluation.comparison import (  # noqa: PLC0415
        pairwise_breakdowns,
        write_pairwise,
    )

    tables: dict[str, pd.DataFrame] = {}
    written: dict[str, Path] = {}
    for name, loss in LOSS_METRICS.items():
        table = pairwise_breakdowns(
            frames,
            loss=loss,
            activity_by_region=activity_by_region,
            channel=channel,
            n_boot=n_boot,
            block_length=block_length,
            alpha=alpha,
            seed=seed,
        )
        tables[name] = table
        written[f"pairwise_{name}"] = write_pairwise(table, out_dir / f"pairwise_{name}.csv")
        logger.info("pairwise_%s.csv: %d rows", name, len(table))
    return tables, written


def _write_cd_diagrams(
    frames: Mapping[str, pd.DataFrame], out_dir: Path, *, stage: str
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    """``report/cd_<metric>.svg`` plus the Friedman results behind them (Q5)."""
    notes: dict[str, str] = {}
    if len(frames) < 3:
        notes["*"] = f"needs at least 3 models for a Friedman test, found {len(frames)}"
        return {}, {}, notes
    try:
        from strikecast.evaluation.stats import (  # noqa: PLC0415
            build_block_matrices,
            friedman_nemenyi,
            save_critical_difference_diagram,
        )
    except ImportError as exc:  # pragma: no cover - matplotlib/posthocs absent
        notes["*"] = f"evaluation.stats is not importable ({exc})"
        return {}, {}, notes

    m_rmse, m_mae = build_block_matrices(dict(frames), require_complete=False)
    matrices = {"rmse": m_rmse, "mae": m_mae}
    results: dict[str, Any] = {}
    written: dict[str, Path] = {}
    for name, matrix in matrices.items():
        complete = matrix.dropna(axis=0, how="any")
        dropped = int(len(matrix) - len(complete))
        if dropped:
            notes[f"{name}_dropped"] = (
                f"{dropped} of {len(matrix)} (region, horizon) blocks are not shared "
                "by every model and were dropped before the Friedman test"
            )
        if len(complete) < 2 or complete.shape[1] < 3:
            notes[name] = (
                f"only {len(complete)} shared blocks over {complete.shape[1]} models; "
                "no critical-difference diagram"
            )
            continue
        try:
            result = friedman_nemenyi(complete)
        except (ValueError, TypeError) as exc:
            notes[name] = f"Friedman failed: {exc}"
            continue
        results[name] = result
        try:
            written[f"cd_{name}"] = save_critical_difference_diagram(
                result,
                out_dir / f"cd_{name}.svg",
                title=f"{name.upper()} - critical difference ({stage})",
            )
        except Exception as exc:  # noqa: BLE001 - a backend failure must not sink the report
            notes[name] = f"diagram not drawn: {exc}"
    return results, written, notes


def _write_family_comparison(
    store: RunStore,
    cfg: ExperimentConfig,
    *,
    stages: Sequence[str],
    eval_seeds: Sequence[int],
    stochastic: Mapping[str, bool],
    alpha: float,
    out_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """``report/family_comparison.csv``: §7.2 item 5, at both family levels."""
    import pandas as pd  # noqa: PLC0415

    from strikecast.evaluation.comparison import (  # noqa: PLC0415
        FAMILY_COLUMNS,
        family_comparison,
        write_family_comparison,
    )
    from strikecast.evaluation.leaderboard import collect_metric_rows, metric_frame  # noqa: PLC0415
    from strikecast.evaluation.seeds import broadcast_deterministic  # noqa: PLC0415

    rows = collect_metric_rows(
        store.root, experiment=cfg.name, stages=stages, seeds=list(eval_seeds)
    )
    rows = broadcast_deterministic(rows, stochastic=stochastic, eval_seeds=list(eval_seeds))
    frame = metric_frame(rows)
    parts = [
        family_comparison(
            frame, stochastic=stochastic, aggregate=aggregate, alpha=alpha
        )
        for aggregate in ("mean", "best")
    ]
    parts = [p for p in parts if not p.empty]
    table = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=list(FAMILY_COLUMNS))
    )
    return table, write_family_comparison(table, out_dir / "family_comparison.csv")


# --------------------------------------------------------------------------- #
# summary.md
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 4) -> str:
    """A number as the summary prints it; ``-`` for anything missing."""
    import numpy as np  # noqa: PLC0415

    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}g}"


def _md_table(frame: pd.DataFrame, *, digits: int = 4, max_rows: int | None = None) -> str:
    """A markdown table; every float goes through :func:`_fmt`."""
    if frame is None or frame.empty:
        return "_(empty)_"
    shown = frame if max_rows is None else frame.head(max_rows)
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    lines = [header, rule]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    if max_rows is not None and len(frame) > max_rows:
        lines.append(f"| ... | {len(frame) - max_rows} more rows in the CSV |" + " |" * 0)
    return "\n".join(lines)


def _ci_lookup(ci: pd.DataFrame | None) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """``{(model, paradigm, stage, metric): ci row}`` from ``leaderboard_ci.csv``."""
    if ci is None or ci.empty:
        return {}
    needed = {"model", "paradigm", "stage", "metric"}
    if not needed.issubset(ci.columns):
        return {}
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in ci.to_dict("records"):
        key = (
            str(record["model"]),
            str(record["paradigm"]),
            str(record["stage"]),
            str(record["metric"]),
        )
        out[key] = record
    return out


def _table4_cell(value: Any, ci_row: dict[str, Any] | None, digits: int) -> str:
    """``0.412 ± 0.013 [0.395, 0.428]``, or a ``det.`` marker (plan §7.1)."""
    import numpy as np  # noqa: PLC0415

    cell = _fmt(value, digits)
    if ci_row is None:
        return cell
    if bool(ci_row.get("deterministic", False)):
        return f"{cell} (det.)"
    parts = [cell]
    sd = ci_row.get("std")
    if sd is not None and np.isfinite(float(sd)):
        parts.append(f"± {_fmt(sd, 2)}")
    lo, hi = ci_row.get("boot_lo"), ci_row.get("boot_hi")
    if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
        parts.append(f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]")
    n = ci_row.get("n")
    if n is not None and int(n) > 1:
        parts.append(f"(n={int(n)})")
    return " ".join(parts)


def _table4(leaderboard: pd.DataFrame, ci: pd.DataFrame | None, digits: int = 4) -> str:
    """The Table 4 layout extended with ``± seed SD`` and ``[bootstrap 95% CI]``."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if leaderboard is None or leaderboard.empty:
        return "_No leaderboard rows: the store holds no metrics for this seed._"
    lookup = _ci_lookup(ci)
    keys = [c for c in ("split", "paradigm", "model") if c in leaderboard.columns]
    metrics = [
        c
        for c in leaderboard.columns
        if c not in keys and pd.api.types.is_numeric_dtype(leaderboard[c])
    ]
    records: list[dict[str, str]] = []
    for row in leaderboard.to_dict("records"):
        record = {k: str(row.get(k, "")) for k in keys}
        for metric in metrics:
            value = row.get(metric)
            key = (
                str(row.get("model", "")),
                str(row.get("paradigm", "")),
                str(row.get("split", "")),
                str(metric),
            )
            record[metric] = _table4_cell(
                value if value is not None and not (isinstance(value, float) and np.isnan(value))
                else None,
                lookup.get(key),
                digits,
            )
        records.append(record)
    return _md_table(pd.DataFrame.from_records(records), digits=digits)


def _matrix(table: pd.DataFrame, column: str, *, antisymmetric: bool) -> pd.DataFrame:
    """A square model x model matrix from the pooled rows of a pairwise table.

    ``pairwise_breakdowns`` emits each unordered pair once, so the mirror cell
    is filled here: with ``-value`` for a difference (A minus B flips) and with
    the same value for a p-value.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    labels = sorted(set(table["model_a"]) | set(table["model_b"]))
    out = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for row in table.to_dict("records"):
        a, b, value = str(row["model_a"]), str(row["model_b"]), row[column]
        out.loc[a, b] = value
        out.loc[b, a] = -float(value) if antisymmetric and value == value else value
    return out


def _matrix_md(matrix: pd.DataFrame, digits: int = 3) -> str:
    """A square matrix as markdown, with the row labels in the first column."""
    if matrix.empty:
        return "_(empty)_"
    header = "| A \\ B | " + " | ".join(str(c) for c in matrix.columns) + " |"
    rule = "|" + "|".join("---" for _ in range(len(matrix.columns) + 1)) + "|"
    lines = [header, rule]
    for label, row in matrix.iterrows():
        lines.append(f"| **{label}** | " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def render_summary(
    *,
    experiment: str,
    store_root: Path | str,
    stage: str,
    seed: int,
    eval_seeds: Sequence[int],
    leaderboard: pd.DataFrame | None,
    ci: pd.DataFrame | None,
    pairwise: Mapping[str, pd.DataFrame] | None = None,
    family: pd.DataFrame | None = None,
    friedman: Mapping[str, Any] | None = None,
    cd_paths: Mapping[str, Path] | None = None,
    cd_notes: Mapping[str, str] | None = None,
    labels: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> str:
    """Render ``report/summary.md`` from the frames the other outputs produced.

    Plain markdown, no templating engine, no numbers of its own: every figure
    in it comes from one of the CSVs written next to it.
    """
    from strikecast.evaluation.stats import (  # noqa: PLC0415
        ALPHA,
        average_rank_table,
        tied_with_best,
    )

    pairwise = dict(pairwise or {})
    friedman = dict(friedman or {})
    cd_paths = dict(cd_paths or {})
    cd_notes = dict(cd_notes or {})

    out: list[str] = [
        f"# Report: {experiment}",
        "",
        f"- store: `{store_root}`",
        f"- leaderboard seed: `{seed}`; evaluation seeds: "
        f"`{', '.join(str(s) for s in eval_seeds) or 'none'}`",
        f"- comparison stage: `{stage}`",
        f"- runs compared: {len(labels)}"
        + (f" (`{'`, `'.join(labels)}`)" if labels else ""),
        "",
    ]
    for note in notes:
        out += [f"> {note}", ""]

    out += [
        "## 1. Leaderboard (Table 4 layout)",
        "",
        "Cells read `value ± seed SD [bootstrap 95% CI over seeds] (n=seeds)`. "
        "`(det.)` marks a model declared deterministic (plan §5.4): it ran once and "
        "is broadcast across seeds, so it gets a point value and never a zero-width "
        "interval. The bootstrap resamples the **seed** population; the "
        "evaluation-sample block bootstrap over forecast origins is reported for pair "
        "differences in `pairwise_*.csv` (plan flag F130).",
        "",
        _table4(leaderboard, ci),
        "",
        "Full per-metric intervals, including the Student-t interval of §7.1, are in "
        "`leaderboard_ci.csv`.",
        "",
        "## 2. Pairwise differences (plan §7.2 items 1-3)",
        "",
    ]
    if not pairwise or all(t.empty for t in pairwise.values()):
        out += [
            "_Not available: fewer than two comparable runs with stored predictions "
            "for this stage and seed._",
            "",
        ]
    else:
        for name, table in pairwise.items():
            loss = LOSS_METRICS.get(name, name)
            out += [f"### {name.upper()} (`{loss}` loss)", ""]
            if table.empty:
                out += ["_(no comparable pairs)_", ""]
                continue
            pooled = table[table["breakdown"] == "global"]
            if pooled.empty:
                out += ["_(no pooled rows)_", ""]
            else:
                out += [
                    "Mean loss difference, A minus B (negative means the ROW model is better):",
                    "",
                    _matrix_md(_matrix(pooled, "mean_diff", antisymmetric=True)),
                    "",
                    "Diebold-Mariano p-values (HAC/Newey-West, bandwidth `horizon - 1`, "
                    "Harvey-Leybourne-Newbold corrected):",
                    "",
                    _matrix_md(_matrix(pooled, "dm_p", antisymmetric=False)),
                    "",
                    "Per pair, pooled over horizons and regions "
                    "(`pct_improvement` > 0 means A is better; the CI is a moving-block "
                    "bootstrap over forecast origins):",
                    "",
                    _md_table(
                        pooled[
                            [
                                "model_a",
                                "model_b",
                                "mean_diff",
                                "pct_improvement",
                                "ci_lo",
                                "ci_hi",
                                "dm_p",
                                "cliffs_delta_a_better",
                                "cliffs_magnitude",
                                "n_origins",
                            ]
                        ],
                        max_rows=25,
                    ),
                    "",
                ]
            for breakdown, heading in (
                ("horizon", "Per horizon"),
                ("tier", "Per activity tier"),
            ):
                sub = table[table["breakdown"] == breakdown]
                if sub.empty:
                    continue
                out += [
                    f"{heading} (plan §7.2 item 6) -- full table in `pairwise_{name}.csv`:",
                    "",
                    _md_table(
                        sub[
                            [
                                "model_a",
                                "model_b",
                                "scope",
                                "mean_diff",
                                "pct_improvement",
                                "dm_p",
                                "cliffs_delta_a_better",
                            ]
                        ],
                        max_rows=25,
                    ),
                    "",
                ]

    out += ["## 3. Family comparison (plan §7.2 item 5)", ""]
    if family is None or family.empty:
        out += [
            "_Not available: the store has no two model families with a shared "
            "evaluation seed._",
            "",
        ]
    else:
        out += [
            "Per-seed metric of family A minus family B on the SAME seed; "
            "`mean_diff` with its Student-t interval over seeds. `deterministic` "
            "marks a pair where both families are deterministic, so the per-seed "
            "differences are identical by construction and no interval is reported.",
            "",
            _md_table(
                family[
                    [
                        "stage",
                        "metric",
                        "aggregate",
                        "family_a",
                        "family_b",
                        "n_seeds",
                        "mean_diff",
                        "pct_improvement",
                        "t_lo",
                        "t_hi",
                        "p_value",
                        "deterministic",
                    ]
                ],
                max_rows=40,
            ),
            "",
        ]

    out += ["## 4. Rank statistics (plan §7.2 item 4)", ""]
    if not friedman:
        reason = cd_notes.get("*") or "; ".join(f"{k}: {v}" for k, v in cd_notes.items())
        out += [f"_Not available: {reason or 'no complete block matrix'}._", ""]
    else:
        for name, result in friedman.items():
            tied = tied_with_best(result)
            out += [
                f"### {name.upper()}",
                "",
                f"- Friedman chi2 = {_fmt(result.chi)}, p = {_fmt(result.p)} "
                f"(k = {result.k} models, N = {result.N} blocks, alpha = {ALPHA})",
                f"- Nemenyi critical distance = {_fmt(result.CD)}",
                f"- within one CD of the best: `{'`, `'.join(tied)}`",
            ]
            note = cd_notes.get(f"{name}_dropped")
            if note:
                out.append(f"- {note}")
            path = cd_paths.get(f"cd_{name}")
            if path is not None:
                out.append(f"- diagram: `{Path(path).name}`")
            out += ["", _md_table(average_rank_table({name: result}).reset_index()), ""]

    out += [
        "---",
        "",
        "Generated by `strikecast report` (plan §7). Every number above is read from "
        "the CSVs in this directory; nothing here recomputes a pipeline.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def report(
    cfg: ExperimentConfig,
    *,
    store: RunStore | None = None,
    seed: int = 42,
    eval_seeds: Sequence[int] | None = None,
    stages: Sequence[str] = STAGES,
    comparison_stage: str | None = None,
    models: Sequence[str] | None = None,
    paradigms: Sequence[str] | None = None,
    channel: str | None = None,
    n_boot: int = 1000,
    block_length: int = 7,
    alpha: float = 0.05,
    boot_seed: int = 0,
    comparisons: bool = True,
) -> dict[str, Path]:
    """Write every plan §7 table for one experiment; returns the paths written.

    Degrades rather than raises (Q1): whatever the store supports is written and
    the rest is named as missing in ``summary.md``.
    """
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    out_dir = store.report_dir(cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    notes: list[str] = []

    frame, path = _write_leaderboard(store, cfg, seed, stages, out_dir)
    if path is not None:
        written["leaderboard"] = path

    seeds = [int(s) for s in (eval_seeds if eval_seeds is not None else cfg.seeds.eval_seeds)]
    stochastic = _stochastic_map(cfg)
    ci = None
    try:
        from strikecast.evaluation.seeds import leaderboard_ci  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - the module is a hard dependency
        logger.info("evaluation.seeds is not available (%s); no CI table", exc)
        notes.append(f"no `leaderboard_ci.csv`: {exc}")
    else:
        ci = leaderboard_ci(
            store.root,
            experiment=cfg.name,
            stages=stages,
            eval_seeds=seeds,
            stochastic=stochastic,
            out_path=out_dir / "leaderboard_ci.csv",
        )
        written["leaderboard_ci"] = out_dir / "leaderboard_ci.csv"

    stage = comparison_stage or ("test" if "test" in tuple(stages) else tuple(stages)[-1])
    frames: dict[str, pd.DataFrame] = {}
    pairwise: dict[str, pd.DataFrame] = {}
    family = None
    friedman: dict[str, Any] = {}
    cd_notes: dict[str, str] = {}
    cd_written: dict[str, Path] = {}

    if comparisons:
        frames = _prediction_frames(
            store,
            cfg.name,
            stage=stage,
            seed=seed,
            models=models,
            paradigms=paradigms,
            channel=channel,
        )
        if len(frames) < 2:
            notes.append(
                f"no pairwise tables: {len(frames)} run(s) with stored `{stage}` "
                f"predictions at seed {seed}."
            )
            logger.info("report: %d comparable run(s); skipping the pairwise tables", len(frames))
        else:
            activity = _activity_map(cfg, store)
            pairwise, pair_paths = _write_pairwise(
                frames,
                out_dir,
                activity_by_region=activity,
                channel=channel,
                n_boot=n_boot,
                block_length=block_length,
                alpha=alpha,
                seed=boot_seed,
            )
            written.update(pair_paths)
        friedman, cd_written, cd_notes = _write_cd_diagrams(frames, out_dir, stage=stage)
        written.update(cd_written)

        family, family_path = _write_family_comparison(
            store,
            cfg,
            stages=stages,
            eval_seeds=seeds,
            stochastic=stochastic,
            alpha=alpha,
            out_dir=out_dir,
        )
        written["family_comparison"] = family_path

    summary = render_summary(
        experiment=cfg.name,
        store_root=store.root,
        stage=stage,
        seed=seed,
        eval_seeds=seeds,
        leaderboard=frame,
        ci=ci,
        pairwise=pairwise,
        family=family,
        friedman=friedman,
        cd_paths=cd_written,
        cd_notes=cd_notes,
        labels=sorted(frames),
        notes=notes,
    )
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    written["summary"] = summary_path

    logger.info(
        "report: %d leaderboard rows, %d CI rows, %d compared runs -> %s",
        0 if frame is None else len(frame),
        0 if ci is None else len(ci),
        len(frames),
        out_dir,
    )
    return written


def _fallback_leaderboard(
    store: RunStore, experiment: str, seed: int, stages: Sequence[str]
) -> pd.DataFrame:
    """One row per ``(model, paradigm, stage)`` from the stored ``global.json``.

    The minimum the plan asks of ``strikecast report`` when
    :mod:`strikecast.evaluation.leaderboard` cannot be imported or cannot sort
    what it found: the seed-42 leaderboard, sorted by ``MASE_mean`` as every
    legacy leaderboard cell is.
    """
    import pandas as pd  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    exp_dir = store.experiment_dir(experiment)
    if exp_dir.is_dir():
        for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            if model_dir.name in _RESERVED:
                continue
            for paradigm_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                seed_dir = paradigm_dir / f"seed={int(seed)}"
                for stage in stages:
                    path = seed_dir / stage / "metrics" / "global.json"
                    if not path.is_file():
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "split": stage,
                            "paradigm": paradigm_dir.name,
                            "model": model_dir.name,
                            **{k: v for k, v in payload.items()},
                        }
                    )
    frame = pd.DataFrame(rows)
    if not frame.empty and "MASE_mean" in frame.columns:
        frame = frame.sort_values("MASE_mean").reset_index(drop=True)
    return frame
