"""Stage 4: the cross-model and cross-seed tables, written under ``report/``.

Deliberately thin. Everything it writes is computed by
:mod:`strikecast.evaluation.leaderboard` (the seed-42 leaderboard that
reproduces ``golden/results/<family>/leaderboard.csv``) and
:mod:`strikecast.evaluation.seeds` (the multi-seed ``leaderboard_ci.csv`` of
plan §7.1). This module only decides *which* experiment, *which* seed and
*where*, and degrades to a minimal seed-42 leaderboard of its own when those
modules are not importable, so the pipeline is usable before they land.

Nothing here touches a training pipeline: both analyses consume the
``PredictionSet``s and metric views already in the run store (§7).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strikecast.pipeline.context import get_spec, resolve_store_root
from strikecast.store import RunStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    import pandas as pd

    from strikecast.config.schema import ExperimentConfig

__all__ = ["report"]

logger = logging.getLogger(__name__)

#: The stages a report covers.
STAGES: tuple[str, ...] = ("cv", "test")


def _stochastic_map(cfg: ExperimentConfig) -> dict[str, bool]:
    """``{model: spec.stochastic}``: which models the seed sweep applies to (§5.4)."""
    out: dict[str, bool] = {}
    for name in cfg.model_names:
        try:
            out[name] = bool(get_spec(name, cfg.name).stochastic)
        except (KeyError, ImportError) as exc:  # a family whose deps are absent
            logger.debug("no spec for %s: %s", name, exc)
    return out


def report(
    cfg: ExperimentConfig,
    *,
    store: RunStore | None = None,
    seed: int = 42,
    eval_seeds: Sequence[int] | None = None,
    stages: Sequence[str] = STAGES,
) -> dict[str, Path]:
    """Write ``report/leaderboard.csv`` and, when possible, ``leaderboard_ci.csv``.

    Returns the paths written, keyed by table name.
    """
    store = store if store is not None else RunStore(resolve_store_root(cfg))
    out_dir = store.report_dir(cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

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
        written["leaderboard"] = path
        return written

    # Collect first and short-circuit on nothing: `build_leaderboard` sorts by
    # `split`, and an empty frame has no columns to sort by, so an empty store
    # raises `KeyError: 'split'` (flagged, not fixed -- evaluation/ is another
    # stream's file).
    rows = collect_metric_rows(
        store.root, experiment=cfg.name, stages=stages, seeds=[int(seed)]
    )
    if not rows:
        logger.warning(
            "no %s metrics under %s for seed %d; nothing to report",
            cfg.name,
            store.root,
            seed,
        )
        return written
    frame = build_leaderboard_from_rows(rows)
    written["leaderboard"] = write_leaderboard(frame, out_dir / "leaderboard.csv")

    seeds = [int(s) for s in (eval_seeds if eval_seeds is not None else cfg.seeds.eval_seeds)]
    try:
        from strikecast.evaluation.seeds import leaderboard_ci  # noqa: PLC0415
    except ImportError as exc:
        logger.info("evaluation.seeds is not available (%s); no CI table", exc)
        return written

    ci = leaderboard_ci(
        store.root,
        experiment=cfg.name,
        stages=stages,
        eval_seeds=seeds,
        stochastic=_stochastic_map(cfg),
        out_path=out_dir / "leaderboard_ci.csv",
    )
    written["leaderboard_ci"] = out_dir / "leaderboard_ci.csv"
    logger.info(
        "report: %d leaderboard rows, %d CI rows -> %s", len(frame), len(ci), out_dir
    )
    return written


def _fallback_leaderboard(
    store: RunStore, experiment: str, seed: int, stages: Sequence[str]
) -> pd.DataFrame:
    """One row per ``(model, paradigm, stage)`` from the stored ``global.json``.

    The minimum the plan asks of ``strikecast report`` before
    :mod:`strikecast.evaluation.leaderboard` exists: the seed-42 leaderboard,
    sorted by ``MASE_mean`` as every legacy leaderboard cell is.
    """
    import pandas as pd  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    exp_dir = store.experiment_dir(experiment)
    if exp_dir.is_dir():
        for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            if model_dir.name in {"shared", "tuning", "report"}:
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
                            "model": model_dir.name,
                            "paradigm": paradigm_dir.name,
                            "stage": stage,
                            "seed": int(seed),
                            **{k: v for k, v in payload.items()},
                        }
                    )
    frame = pd.DataFrame(rows)
    if not frame.empty and "MASE_mean" in frame.columns:
        frame = frame.sort_values("MASE_mean").reset_index(drop=True)
    return frame
