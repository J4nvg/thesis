"""Legacy-mode fold-subset verification against the golden predictions (audit D9).

Decision D9: the port is verified WITHOUT doubling the compute by running, for
every model of the thesis job matrix, only the first ``windows`` retrain
windows (``windows x 7`` folds) of its reported stage in **legacy mode** and
comparing those fold predictions with the thesis' own stored predictions.

Legacy mode is ``legacy=<experiment>`` (``configs/legacy/``): the buggy
``expdecay7`` window transform (``series.window.expdecay: legacy_alpha``) and
the thesis' cached feature selection (count/diff; the hurdle heads are selected
once, deterministically, on the legacy columns by :func:`prep`), plus the
thesis' tuned hyper-parameters imported into a SEPARATE store root
(``scripts/import_golden_params.py``: converted studies + ``_regression_GBDT
.ipynb`` cell 56; Chronos-2 from ``golden/checkpoints/chronos2_best``).
Legacy and publication runs never share a store.

What is compared, per ``(experiment, model, paradigm)``
-------------------------------------------------------
The run's ``PredictionSet.legacy_frame()`` against
``golden/results/<dir>/predictions_long_<stage>_<paradigm>_<model>[_tuned].parquet``
(``results/`` as a fallback), restricted to the folds that ran. ALWAYS exact:
the ``(region, fold, horizon, date)`` keys and ``y_true`` -- they come from the
data and the schedule, so any difference is a real bug (status ``FAIL``).
``y_pred`` is held to the level of :func:`tolerance_for`:

``E``  bit-level / closed form (naives exact, linear 1e-6, count CatBoost 1e-6:
       audit B1 reproduced golden fold 0 to 1.8e-15);
``F``  a documented family tolerance (LightGBM 5.0 and ARIMA 0.5: F124, the
       ceilings of ``tests/golden/test_pipeline_equality.py``; Chronos-2
       zero-shot 1e-2: F134);
``R``  report only -- no stored evidence of reproducibility (GPU training, the
       RNNs F9, XGBoost after the per-horizon fix B3/B4, the Chronos-2
       fine-tune, the hurdle whose thesis selection is unrecoverable B9). The
       deviation is reported; ``SUSPECT`` flags a relative MAE difference above
       :data:`SUSPECT_REL_MAE` (e.g. a log-space bug like B7 would show here).

Outputs under ``<store-root>/_verification/``: one ``results/<case>.json`` per
case (written as each finishes, so concurrent SLURM jobs never share a file)
and :func:`write_report`'s ``verification.csv`` + ``verification.md``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Case",
    "SUSPECT_REL_MAE",
    "cases_for",
    "compare_frames",
    "golden_path",
    "prep",
    "run_cases",
    "tolerance_for",
    "write_report",
]

REPO = Path(__file__).resolve().parents[3]
GOLDEN_ROOTS = (REPO / "golden" / "results", REPO / "results")
VERIFY_DIR = "_verification"
RETRAIN_STRIDE = 7
#: `R`-level cases whose MAE differs from golden by more than this are SUSPECT.
SUSPECT_REL_MAE = 0.25
KEYS = ["region", "fold", "horizon", "date"]

#: experiment -> golden results directory.
GOLDEN_DIR = {"diff": "diff", "chronos2": "chronos2", "hurdle": "finalhurdle"}
#: The stage each family is verified on: the reported one (test), except the
#: hurdle, whose test stage needs calibrators from a full CV; its CV stage
#: exercises the same heads and the stored golden CV frames exist.
VERIFY_STAGE = {"hurdle": "cv"}
#: hurdle channel -> golden file stem (``final_hurdle.ipynb`` cells 48-49).
HURDLE_CHANNELS = {
    "prob": "classifier_probs",
    "count": "regressor_preds",
    "hurdle": "hurdle_preds",
}

_E = ("E", "exact or closed form")
_LGBM = ("F", 5.0, "F124: LightGBM trees are not portable across platforms")
_ARIMA = ("F", 0.5, "F124: statsmodels MLE optimum depends on the BLAS")


@dataclass(frozen=True)
class Case:
    experiment: str
    model: str
    paradigm: str
    stage: str
    family: str
    kind: str
    resource: str
    golden: str | None

    @property
    def id(self) -> str:
        return f"{self.experiment}__{self.model}__{self.paradigm}__{self.stage}"


@dataclass
class Result:
    experiment: str
    model: str
    paradigm: str
    stage: str
    level: str
    tolerance: float | None
    status: str
    folds: int = 0
    rows: int = 0
    max_abs_dy: float | None = None
    mean_abs_dy: float | None = None
    mae_run: float | None = None
    mae_golden: float | None = None
    rel_mae_diff: float | None = None
    seconds: float | None = None
    golden: str | None = None
    note: str = ""
    channels: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# cases and tolerances
# --------------------------------------------------------------------------- #
def golden_dir_name(experiment: str, family: str) -> str:
    if experiment == "count":
        return "lstm" if family in ("lstm", "gru") else "gbdt"
    return GOLDEN_DIR.get(experiment, experiment)


def golden_path(
    experiment: str, model: str, paradigm: str, stage: str, family: str,
    roots: Sequence[Path] = GOLDEN_ROOTS,
) -> Path | None:
    """The stored thesis predictions for one case, or ``None``."""
    directory = golden_dir_name(experiment, family)
    if experiment == "chronos2":
        names = [f"predictions_long_{model}.parquet"]
    elif experiment == "hurdle":
        names = [f"{paradigm}_{stage}_{HURDLE_CHANNELS['hurdle']}.parquet"]
    else:
        names = [
            f"predictions_long_{stage}_{paradigm}_{model}_tuned.parquet",
            f"predictions_long_{stage}_{paradigm}_{model}.parquet",
        ]
    for root in roots:
        for name in names:
            path = Path(root) / directory / name
            if path.is_file():
                return path
    return None


def tolerance_for(experiment: str, model: str, family: str, kind: str) -> tuple[str, float | None, str]:
    """``(level, max |dy_pred| tolerance or None, why)`` for one case."""
    if kind == "naive":
        return "E", 0.0, "no fit at all"
    if family == "linear":
        return "E", 1e-6, "closed-form solve (golden level E)"
    if family == "arima":
        return _ARIMA
    if family == "lightgbm":
        return _LGBM
    if experiment == "count" and family == "catboost":
        return "E", 1e-6, "B1: cell-56 params reproduce golden fold 0 to 1.8e-15 (CPU CatBoost)"
    if model == "chronos2_zero_shot":
        return "F", 1e-2, "F134: transformer float non-determinism, max |d| ~5e-3"
    if family in ("lstm", "gru"):
        return "R", None, "F9: GPU training is not bit-reproducible"
    if family == "xgboost":
        return "R", None, "B3/B4: one model per horizon now; no stored evidence yet"
    if family == "catboost":
        return "R", None, "diff CatBoost ran on GPU (F9)"
    if kind == "composite":
        return "R", None, "B9: the thesis' hurdle selection is unrecoverable"
    if kind == "chronos":
        return "R", None, "fine-tune on GPU is not bit-reproducible"
    return "R", None, "no stored evidence"


def cases_for(
    experiment: str,
    *,
    resource: str | None = None,
    models: Sequence[str] | None = None,
    overrides: Sequence[str] = (),
    config_dir: str | Path | None = None,
) -> list[Case]:
    """Every thesis-matrix ``(model, paradigm)`` of one experiment with golden predictions."""
    from strikecast.config.loader import load_experiment  # noqa: PLC0415
    from strikecast.pipeline.context import get_spec  # noqa: PLC0415

    make_jobs = _make_jobs()
    cfg = load_experiment(experiment, list(overrides), config_dir=config_dir)
    stage = VERIFY_STAGE.get(experiment, "test")
    out: list[Case] = []
    for model in cfg.model_names:
        if models and model not in models:
            continue
        spec = make_jobs.lookup_spec(get_spec, model, experiment)
        res = make_jobs.resource_class(cfg, model, spec)
        if resource and res != resource:
            continue
        for paradigm in make_jobs.paradigms_for(cfg, model):
            path = golden_path(experiment, model, paradigm, stage, spec.family)
            out.append(
                Case(experiment, model, paradigm, stage, spec.family, spec.kind, res,
                     None if path is None else str(path))
            )
    return out


def _make_jobs():
    import importlib.util  # noqa: PLC0415

    path = REPO / "scripts" / "slurm" / "make_jobs.py"
    if "make_jobs" in sys.modules:
        return sys.modules["make_jobs"]
    spec = importlib.util.spec_from_file_location("make_jobs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["make_jobs"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def compare_frames(got, want, *, max_folds: int | None, level: str, tol: float | None) -> dict[str, Any]:
    """Compare one run frame with one golden frame (legacy long format).

    Both have ``region, fold, horizon, date, y_true, y_pred`` (``y_prob`` is
    accepted for the golden probability frames). Rows are aligned on the keys
    after restricting the golden frame to the folds that ran. Keys and
    ``y_true`` must match exactly; ``y_pred`` is scored against ``tol``.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    want = want.rename(columns={"y_prob": "y_pred"})
    if max_folds is not None:
        want = want[want["fold"] < max_folds]
    got = got.sort_values(["region", "fold", "horizon"], kind="stable").reset_index(drop=True)
    want = want.sort_values(["region", "fold", "horizon"], kind="stable").reset_index(drop=True)
    out: dict[str, Any] = {"rows": int(len(got)), "folds": int(got["fold"].nunique()) if len(got) else 0}
    if len(got) != len(want):
        out.update(status="FAIL", note=f"{len(got)} rows, golden has {len(want)} (schedule bug)")
        return out
    for column in KEYS:
        left, right = got[column], want[column]
        if column == "date":
            left, right = pd.to_datetime(left), pd.to_datetime(right)
        if not (left.astype(str).to_numpy() == right.astype(str).to_numpy()).all():
            out.update(status="FAIL", note=f"key column {column!r} differs from golden")
            return out
    y_got = got["y_true"].to_numpy(float)
    y_want = want["y_true"].to_numpy(float)
    if not np.array_equal(y_got, y_want, equal_nan=True):
        out.update(
            status="FAIL",
            note=f"y_true differs (max {float(np.nanmax(np.abs(y_got - y_want))):.3g}): data bug",
        )
        return out
    dy = np.abs(got["y_pred"].to_numpy(float) - want["y_pred"].to_numpy(float))
    mae_run = float(np.nanmean(np.abs(got["y_pred"].to_numpy(float) - y_want)))
    mae_golden = float(np.nanmean(np.abs(want["y_pred"].to_numpy(float) - y_want)))
    rel = abs(mae_run - mae_golden) / mae_golden if mae_golden else float("inf")
    out.update(
        max_abs_dy=float(np.nanmax(dy)) if len(dy) else 0.0,
        mean_abs_dy=float(np.nanmean(dy)) if len(dy) else 0.0,
        mae_run=mae_run,
        mae_golden=mae_golden,
        rel_mae_diff=float(rel),
    )
    if tol is None:
        out["status"] = "SUSPECT" if rel > SUSPECT_REL_MAE else "REPORT"
    else:
        out["status"] = "PASS" if out["max_abs_dy"] <= tol else "FAIL"
    return out


# --------------------------------------------------------------------------- #
# prep: tuned params and the hurdle's legacy selection, in the verify store
# --------------------------------------------------------------------------- #
def prep(store_root: str | Path, *, experiments: Iterable[str] = ("count", "diff", "hurdle", "chronos2")) -> None:
    """Import the thesis' params into ``store_root`` and select the hurdle heads.

    Idempotent. Runs ONCE (the ``verify_prep`` SLURM job) so the concurrent
    ``verify`` jobs never write the same files.
    """
    store_root = Path(store_root).resolve()
    store_root.mkdir(parents=True, exist_ok=True)
    script = REPO / "scripts" / "import_golden_params.py"
    result = subprocess.run(
        [sys.executable, str(script), "--store-root", str(store_root)],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        logger.warning("import_golden_params exited %d; affected cases will report "
                       "MissingTunedParams", result.returncode)
    experiments = list(experiments)
    if "chronos2" in experiments:
        _import_chronos_params(store_root)
    if "hurdle" in experiments:
        from strikecast.config.loader import load_experiment  # noqa: PLC0415
        from strikecast.pipeline import data_stage  # noqa: PLC0415
        from strikecast.store import RunStore  # noqa: PLC0415

        cfg = load_experiment("hurdle", _legacy_overrides("hurdle", store_root))
        for head, found in data_stage.select_features(cfg, RunStore(store_root)).items():
            print(f"hurdle/{head}: {found.source} past={len(found.past_keep)} "
                  f"future={len(found.future_keep)} {found.path}")


def _import_chronos_params(store_root: Path) -> None:
    """The thesis' Chronos-2 fine-tune params -> the store (if absent).

    Stream 2's ``scripts/import_golden_chronos.py`` (best_params + trials.csv,
    cross-checked) when it exists; otherwise the sidecar
    ``golden/checkpoints/chronos2_best/best_params.json`` alone.
    """
    source = REPO / "golden" / "checkpoints" / "chronos2_best" / "best_params.json"
    target = store_root / "chronos2" / "tuning" / "chronos2_fine_tuned" / "best_params.json"
    script = REPO / "scripts" / "import_golden_chronos.py"
    if not target.is_file() and script.is_file():
        done = subprocess.run(
            [sys.executable, str(script), "--store-root", str(store_root)],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        print(done.stdout, done.stderr, sep="")
    if target.is_file() or not source.is_file():
        return
    payload = json.loads(source.read_text(encoding="utf-8"))
    best = payload.get("best_params", payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "variant": "chronos2_fine_tuned",
                "source_dir": "golden/checkpoints/chronos2_best",
                "source_file": "best_params.json",
                "best_params": {
                    "fine_tune_lr": float(best["fine_tune_lr"]),
                    "fine_tune_steps": int(best["fine_tune_steps"]),
                },
                "best_value": None,
                "n_trials": payload.get("n_trials"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"chronos2_fine_tuned: imported {source} -> {target}")


def _legacy_overrides(experiment: str, store_root: Path, paradigm: str | None = None) -> list[str]:
    out = [f"legacy={experiment}", "tracking=noop", f"++store.root={store_root}",
           "hydra.job.chdir=false"]
    if paradigm is not None:
        out.append(f"paradigm={paradigm}")
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def results_dir(store_root: str | Path) -> Path:
    return Path(store_root).resolve() / VERIFY_DIR / "results"


def run_cases(
    experiment: str,
    *,
    store_root: str | Path,
    windows: int = 2,
    resource: str | None = None,
    models: Sequence[str] | None = None,
    force: bool = False,
) -> list[Result]:
    """Run and compare every case of one experiment (optionally one resource class)."""
    from strikecast.config.loader import load_experiment  # noqa: PLC0415
    from strikecast.pipeline import data_stage, run_stage  # noqa: PLC0415
    from strikecast.store import RunKey, RunStore  # noqa: PLC0415

    store_root = Path(store_root).resolve()
    store = RunStore(store_root)
    out_dir = results_dir(store_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_folds = int(windows) * RETRAIN_STRIDE
    base = _legacy_overrides(experiment, store_root)
    cases = cases_for(experiment, resource=resource, models=models, overrides=base)
    results: list[Result] = []
    by_paradigm: dict[str, list[Case]] = {}
    for case in cases:
        by_paradigm.setdefault(case.paradigm, []).append(case)

    for paradigm, group in by_paradigm.items():
        cfg = load_experiment(experiment, _legacy_overrides(experiment, store_root, paradigm))
        data = None
        for case in group:
            level, tol, why = tolerance_for(case.experiment, case.model, case.family, case.kind)
            result = Result(
                case.experiment, case.model, case.paradigm, case.stage, level, tol, "ERROR",
                golden=case.golden, note=why,
            )
            started = time.monotonic()
            try:
                if case.golden is None:
                    result.status = "SKIP"
                    result.note = "no golden predictions for this case"
                else:
                    if data is None:
                        data = data_stage.prepare_data(cfg, store)
                    run_stage.run_stage(
                        cfg, case.model, paradigm, 42, case.stage, data,
                        store=store, force=force, max_folds=max_folds,
                    )
                    key = RunKey(experiment, case.model, paradigm, 42)
                    _compare_case(result, store, key, case, max_folds, level, tol)
            except Exception as exc:  # recorded, never fatal: one case must not hide the rest
                result.status = "ERROR"
                result.note = f"{type(exc).__name__}: {exc}"[:500]
                logger.error("%s: %s\n%s", case.id, exc, traceback.format_exc())
            result.seconds = round(time.monotonic() - started, 1)
            _write_json(out_dir / f"{case.id}.json", asdict(result))
            print(f"{result.status:<8} {case.id}  max|dy|={_fmt(result.max_abs_dy)}  "
                  f"tol={_fmt(tol)}  {result.note}")
            results.append(result)
    return results


def _compare_case(result: Result, store, key, case: Case, max_folds: int, level: str, tol) -> None:
    import pandas as pd  # noqa: PLC0415

    predictions = store.load_predictions(key, case.stage, legacy_order=True)
    if case.kind == "composite":
        frame = predictions.frame
        worst: dict[str, Any] | None = None
        for channel, stem in HURDLE_CHANNELS.items():
            path = Path(case.golden).with_name(f"{case.paradigm}_{case.stage}_{stem}.parquet")
            if not path.is_file():
                continue
            got = frame[frame["channel"] == channel].loc[:, [*KEYS, "y_true", "y_pred"]]
            compared = compare_frames(
                got, pd.read_parquet(path), max_folds=max_folds, level=level, tol=tol
            )
            result.channels[channel] = compared
            if worst is None or compared.get("status") == "FAIL" or channel == "hurdle":
                worst = compared
        compared = worst or {"status": "SKIP", "note": "no golden channel files"}
    else:
        compared = compare_frames(
            predictions.legacy_frame(), pd.read_parquet(case.golden),
            max_folds=max_folds, level=level, tol=tol,
        )
    for name in ("rows", "folds", "max_abs_dy", "mean_abs_dy", "mae_run", "mae_golden",
                 "rel_mae_diff"):
        if name in compared:
            setattr(result, name, compared[name])
    result.status = compared.get("status", "ERROR")
    if compared.get("note"):
        result.note = compared["note"]
    if any(c.get("status") == "FAIL" for c in result.channels.values()):
        result.status = "FAIL"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3g}"
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
STATUS_ORDER = ("FAIL", "ERROR", "SUSPECT", "REPORT", "PASS", "SKIP")


def load_results(store_root: str | Path) -> list[dict[str, Any]]:
    directory = results_dir(store_root)
    rows = []
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return rows


def write_report(store_root: str | Path) -> tuple[Path, Path, dict[str, int]]:
    """``verification.csv`` + ``verification.md`` from every result JSON."""
    import pandas as pd  # noqa: PLC0415

    rows = load_results(store_root)
    out = Path(store_root).resolve() / VERIFY_DIR
    out.mkdir(parents=True, exist_ok=True)
    columns = ["experiment", "model", "paradigm", "stage", "level", "tolerance", "status",
               "folds", "rows", "max_abs_dy", "mean_abs_dy", "mae_run", "mae_golden",
               "rel_mae_diff", "seconds", "note", "golden"]
    frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    if len(frame):
        frame["_order"] = frame["status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
        frame = frame.sort_values(["_order", "experiment", "model", "paradigm"]).drop(columns="_order")
    csv_path = out / "verification.csv"
    frame.to_csv(csv_path, index=False)
    counts = {s: int((frame["status"] == s).sum()) for s in STATUS_ORDER} if len(frame) else {}

    lines = [
        "# Legacy-mode verification (audit 2026-09-26 D9)",
        "",
        f"Store `{Path(store_root).resolve()}`; first retrain windows of every thesis-matrix "
        "model, legacy mode (`legacy=<experiment>`), compared with the thesis' stored "
        "predictions. Levels: E exact/closed form, F family tolerance, R report only.",
        "",
        "Summary: " + ", ".join(f"{s} {n}" for s, n in counts.items() if n) if counts else "No results.",
        "",
        "| status | experiment | model | paradigm | stage | level | tol | folds | max abs dy | "
        "mean abs dy | MAE run | MAE golden | rel dMAE | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            "| " + " | ".join(
                _fmt(row.get(c)) if c not in ("note",) else str(row.get(c) or "").replace("|", "/")
                for c in ("status", "experiment", "model", "paradigm", "stage", "level",
                          "tolerance", "folds", "max_abs_dy", "mean_abs_dy", "mae_run",
                          "mae_golden", "rel_mae_diff", "note")
            ) + " |"
        )
    md_path = out / "verification.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path, counts


def clean(store_root: str | Path) -> None:
    """Remove the verification results (not the runs)."""
    shutil.rmtree(Path(store_root).resolve() / VERIFY_DIR, ignore_errors=True)
