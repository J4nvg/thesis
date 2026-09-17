#!/usr/bin/env python3
"""Convert legacy pickled artefacts under golden/ into durable formats.

Outputs land in <golden>/converted/ and a golden/MANIFEST.json is written with
size+sha256 for every file under golden/.

Usage:
    uv run python scripts/convert_legacy_artifacts.py [--golden golden] [--force]
    uv run python scripts/convert_legacy_artifacts.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

TUNE_DIRS = ("checkpoints_tune", "checkpoints_tune_diff")
CV_DIRS = ("checkpoints", "checkpoints_diff", "checkpoints_log")
FEATURE_DIR = "features"


# ---------------------------------------------------------------- utilities
def jsonable(obj):
    """Best-effort conversion of arbitrary values to JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    for attr in ("item", "isoformat"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return jsonable(fn())
            except Exception:
                pass
    return str(obj)


def describe_exc(exc: BaseException) -> dict:
    """Describe a failure, naming the missing module/attribute when that is the cause."""
    info = {"error_type": type(exc).__name__, "error": str(exc)}
    if isinstance(exc, ModuleNotFoundError):
        info["missing_module"] = exc.name
        info["cause"] = f"missing module: {exc.name}"
    elif isinstance(exc, ImportError):
        info["cause"] = f"import failure: {exc}"
    elif isinstance(exc, AttributeError):
        info["cause"] = f"missing attribute: {exc}"
    return info


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ts_frame(series, region: str) -> pd.DataFrame:
    """darts TimeSeries -> tidy DataFrame with date + region columns."""
    df = series.to_dataframe()
    df = df.reset_index()
    df.columns = [str(c) for c in df.columns]
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df.insert(1, "region", region)
    return df


def concat_regions(series_list, region_names) -> pd.DataFrame:
    return pd.concat(
        [ts_frame(s, r) for s, r in zip(series_list, region_names)],
        ignore_index=True,
    )


# ------------------------------------------------------------ job discovery
def discover(golden: Path) -> dict:
    """Map every legacy pickle to the converted outputs it is expected to produce."""
    conv = golden / "converted"
    jobs = {"tuning": [], "feature_sets": [], "cv_predictions": []}

    for sub in TUNE_DIRS:
        for pkl in sorted((golden / sub).glob("*_best.pkl")):
            variant = pkl.name[: -len("_best.pkl")]
            out = conv / "tuning" / sub / variant
            jobs["tuning"].append(
                {
                    "source": pkl,
                    "variant": variant,
                    "source_dir": sub,
                    "out_dir": out,
                    "expected": [out / "best_params.json", out / "trials.csv"],
                }
            )

    for pkl in sorted((golden / FEATURE_DIR).glob("*_saved_sets.pkl")):
        name = pkl.name[: -len("_saved_sets.pkl")]
        out = conv / "feature_sets"
        jobs["feature_sets"].append(
            {
                "source": pkl,
                "name": name,
                "out_dir": out / name,
                "json": out / f"{name}.json",
                "expected": [
                    out / f"{name}.json",
                    out / name / "target_full.parquet",
                    out / name / "past_covs.parquet",
                    out / name / "future_covs.parquet",
                ],
            }
        )

    for sub in CV_DIRS:
        for pkl in sorted((golden / sub).glob("*_tuned.pkl")):
            name = pkl.name[: -len(".pkl")]
            out = conv / "cv_predictions" / sub
            jobs["cv_predictions"].append(
                {
                    "source": pkl,
                    "name": name,
                    "source_dir": sub,
                    "parquet": out / f"{name}.parquet",
                    "json": out / f"{name}.json",
                    "expected": [out / f"{name}.parquet", out / f"{name}.json"],
                }
            )
    return jobs


# ------------------------------------------------------------- converters
def convert_tuning(job, force: bool) -> str:
    out = job["out_dir"]
    params_path, trials_path = job["expected"]
    if not force and params_path.exists() and trials_path.exists():
        return "skipped"

    best_params, study = load_pickle(job["source"])
    out.mkdir(parents=True, exist_ok=True)

    trials = list(study.trials)

    def count(state_name):
        return sum(1 for t in trials if t.state.name == state_name)

    try:
        best_value = jsonable(study.best_value)
    except Exception as exc:  # multi-objective or no completed trial
        best_value = None
        print(f"    note: best_value unavailable ({type(exc).__name__}: {exc})")

    try:
        direction = [d.name for d in study.directions]
        direction = direction[0] if len(direction) == 1 else direction
    except Exception:
        direction = str(getattr(study, "direction", None))

    meta = {
        "variant": job["variant"],
        "source_dir": job["source_dir"],
        "source_file": job["source"].name,
        "best_params": jsonable(best_params),
        "best_value": best_value,
        "n_trials": len(trials),
        "n_pruned": count("PRUNED"),
        "n_complete": count("COMPLETE"),
        "n_failed": count("FAIL"),
        "sampler": type(study.sampler).__name__,
        "pruner": type(study.pruner).__name__,
        "direction": direction,
        "study_name": getattr(study, "study_name", None),
    }
    params_path.write_text(json.dumps(meta, indent=2, sort_keys=False) + "\n")
    study.trials_dataframe().to_csv(trials_path, index=False)
    return "converted"


def convert_feature_set(job, force: bool) -> str:
    if not force and all(p.exists() for p in job["expected"]):
        return "skipped"

    obj = load_pickle(job["source"])
    (region_names, train_target, val_target, test_target,
     full_past_covs, full_fut_covs, target_for_cv,
     train_val_end, cv_start_val) = obj

    out_dir = job["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    region_names = [str(r) for r in region_names]
    static = train_target[0].static_covariates
    static_cols = [] if static is None else [str(c) for c in static.columns]

    meta = {
        "name": job["name"],
        "source_file": job["source"].name,
        "n_regions": len(region_names),
        "region_names": region_names,
        "past_covariate_components": [str(c) for c in full_past_covs[0].components],
        "future_covariate_components": [str(c) for c in full_fut_covs[0].components],
        "target_components": [str(c) for c in train_target[0].components],
        "n_past_covariates": len(full_past_covs[0].components),
        "n_future_covariates": len(full_fut_covs[0].components),
        "TRAIN_VAL_END": jsonable(train_val_end),
        "CV_START_VAL": jsonable(cv_start_val),
        "static_covariate_columns": static_cols,
        "region0": {
            "region": region_names[0],
            "train_len": len(train_target[0]),
            "val_len": len(val_target[0]),
            "test_len": len(test_target[0]),
            "cv_view_len": len(target_for_cv[0]),
            "past_covs_len": len(full_past_covs[0]),
            "future_covs_len": len(full_fut_covs[0]),
            "train_start": jsonable(train_target[0].start_time()),
            "train_end": jsonable(train_target[0].end_time()),
            "val_start": jsonable(val_target[0].start_time()),
            "val_end": jsonable(val_target[0].end_time()),
            "test_start": jsonable(test_target[0].start_time()),
            "test_end": jsonable(test_target[0].end_time()),
            "cv_view_start": jsonable(target_for_cv[0].start_time()),
            "cv_view_end": jsonable(target_for_cv[0].end_time()),
        },
    }
    job["json"].parent.mkdir(parents=True, exist_ok=True)
    job["json"].write_text(json.dumps(meta, indent=2) + "\n")

    target_full = [
        tr.append(vl).append(te)
        for tr, vl, te in zip(train_target, val_target, test_target)
    ]
    concat_regions(target_full, region_names).to_parquet(
        out_dir / "target_full.parquet", index=False)
    concat_regions(full_past_covs, region_names).to_parquet(
        out_dir / "past_covs.parquet", index=False)
    concat_regions(full_fut_covs, region_names).to_parquet(
        out_dir / "future_covs.parquet", index=False)
    return "converted"


def convert_cv(job, force: bool) -> str:
    if not force and job["parquet"].exists() and job["json"].exists():
        return "skipped"

    long_df, fold_preds = load_pickle(job["source"])
    job["parquet"].parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(job["parquet"], index=False)

    n_regions = len(fold_preds)
    fold_counts = sorted({len(fp) for fp in fold_preds})
    meta = {
        "name": job["name"],
        "source_dir": job["source_dir"],
        "source_file": job["source"].name,
        "n_rows": int(len(long_df)),
        "columns": [str(c) for c in long_df.columns],
        "n_regions": n_regions,
        "n_folds": fold_counts[0] if len(fold_counts) == 1 else fold_counts,
        "fold_counts_per_region": [len(fp) for fp in fold_preds],
        "note": "fold_preds (per-region lists of darts TimeSeries) not serialised; "
                "long_df carries the predictions.",
    }
    job["json"].write_text(json.dumps(meta, indent=2) + "\n")
    return "converted"


# ---------------------------------------------------------------- manifest
def write_manifest(golden: Path, counts: dict, failures: list) -> dict:
    files = []
    manifest_path = golden / "MANIFEST.json"
    for path in sorted(p for p in golden.rglob("*") if p.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": str(path.relative_to(golden)),
                "size": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "thesis-final"],
            cwd=golden.parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:
        commit = None
        print(f"  warning: could not resolve tag thesis-final ({exc})")

    manifest = {
        "summary": {
            "git_tag": "thesis-final",
            "git_commit": commit,
            "n_files": len(files),
            "total_bytes": sum(f["size"] for f in files),
            "counts": counts,
            "failures": failures,
        },
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ------------------------------------------------------------------- modes
def run_convert(golden: Path, force: bool) -> int:
    jobs = discover(golden)
    handlers = {
        "tuning": convert_tuning,
        "feature_sets": convert_feature_set,
        "cv_predictions": convert_cv,
    }
    counts = {}
    failures = []
    for category, items in jobs.items():
        converted = skipped = 0
        print(f"\n== {category} ({len(items)} pickles)")
        for job in items:
            rel = str(job["source"].relative_to(golden))
            try:
                status = handlers[category](job, force)
            except Exception as exc:
                info = describe_exc(exc)
                info.update({"category": category, "source": rel,
                             "traceback": traceback.format_exc(limit=5)})
                failures.append(info)
                print(f"  FAIL {rel}: {info['error_type']}: {info['error']}")
                continue
            if status == "converted":
                converted += 1
                print(f"  ok   {rel}")
            else:
                skipped += 1
                print(f"  skip {rel} (outputs exist)")
        counts[category] = {
            "n_sources": len(items),
            "converted": converted,
            "skipped": skipped,
            "failed": sum(1 for f in failures if f["category"] == category),
        }

    # Directories deliberately left as opaque bytes (AutoGluon not installable).
    opaque = [
        str(p.relative_to(golden))
        for p in sorted((golden / "checkpoints").glob("chronos2_*"))
        if p.is_dir()
    ]
    counts["opaque_dirs_hashed_only"] = opaque

    conv_dir = golden / "converted"
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / "conversion_log.json").write_text(
        json.dumps({"counts": counts, "failures": failures}, indent=2) + "\n"
    )

    manifest = write_manifest(golden, counts, failures)
    print(f"\nmanifest: {manifest['summary']['n_files']} files, "
          f"{manifest['summary']['total_bytes'] / 1e6:.1f} MB")
    return 1 if failures else 0


def run_check(golden: Path) -> int:
    jobs = discover(golden)
    missing = []
    total = 0
    for category, items in jobs.items():
        for job in items:
            for exp in job["expected"]:
                total += 1
                if not exp.exists():
                    missing.append(str(exp.relative_to(golden)))
    if not (golden / "MANIFEST.json").exists():
        missing.append("MANIFEST.json")
        total += 1
    print(f"checked {total} expected outputs; missing {len(missing)}")
    for m in missing:
        print(f"  MISSING {m}")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", default="golden", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="verify expected converted outputs exist; exit 1 if any missing")
    ap.add_argument("--force", action="store_true",
                    help="re-convert even when outputs already exist")
    args = ap.parse_args()

    golden = args.golden.resolve()
    if not golden.is_dir():
        print(f"error: {golden} is not a directory", file=sys.stderr)
        return 2
    return run_check(golden) if args.check else run_convert(golden, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
