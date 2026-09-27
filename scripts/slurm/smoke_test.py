#!/usr/bin/env python3
"""Smoke-test a freshly built environment and stamp it (run by ``setup.sbatch``).

    .venv/bin/python scripts/slurm/smoke_test.py main
    envs/autogluon/.venv/bin/python scripts/slurm/smoke_test.py autogluon

``main``: CUDA is available and a tiny op runs on the GPU; LightGBM, XGBoost
and CatBoost import and fit a toy problem on the CPU, and XGBoost (``cuda``) and
CatBoost (``GPU``) also on the GPU -- the diff family runs them there (plan §1
"Device policy"); darts, torch and the strikecast model registry import.

``autogluon``: the same CUDA check, AutoGluon imports, and a tiny Chronos-2
zero-shot fit/predict, which downloads the Chronos-2 weights into ``$HF_HOME``
(shared ``/home``) so no job has to; ``strikecast.models.chronos`` imports.

On success the stamp ``<venv>/.strikecast-setup.json`` records the lock file's
sha256, the host and the versions; ``submit_all.py`` resubmits the setup job
when the stamp is missing or the lock changed. Any failure exits non-zero, so
every job that depends on the setup job is cancelled rather than run in a
broken environment. ``--no-gpu`` skips the GPU parts (laptop).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCKS = {"main": REPO / "uv.lock", "autogluon": REPO / "envs" / "autogluon" / "uv.lock"}
VENVS = {"main": REPO / ".venv", "autogluon": REPO / "envs" / "autogluon" / ".venv"}
STAMP = ".strikecast-setup.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda(gpu: bool) -> dict:
    import torch

    info = {"torch": torch.__version__, "cuda_build": torch.version.cuda}
    if not gpu:
        return info
    if not torch.cuda.is_available():
        raise SystemExit("smoke test: torch.cuda.is_available() is False on a GPU job")
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    value = float((x * 2).sum().item())
    if value != 1023 * 1024:
        raise SystemExit(f"smoke test: a tiny CUDA op returned {value}")
    info["device"] = torch.cuda.get_device_name(0)
    return info


def _toy():
    import numpy as np

    rng = np.random.default_rng(0)
    features = rng.normal(size=(256, 5))
    target = np.maximum(features[:, 0] * 2 + rng.normal(size=256), 0)
    return features, target


def check_main(gpu: bool) -> dict:
    info = {"cuda": _cuda(gpu)}
    features, target = _toy()

    import catboost
    import lightgbm
    import xgboost

    lightgbm.LGBMRegressor(n_estimators=5, verbose=-1).fit(features, target)
    xgboost.XGBRegressor(n_estimators=5, device="cpu").fit(features, target)
    catboost.CatBoostRegressor(iterations=5, verbose=False, task_type="CPU").fit(features, target)
    if gpu:
        xgboost.XGBRegressor(n_estimators=5, device="cuda").fit(features, target)
        catboost.CatBoostRegressor(iterations=5, verbose=False, task_type="GPU").fit(
            features, target
        )
    info["versions"] = {
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
    }

    import darts

    import strikecast.models.registry  # noqa: F401  (imports every model family)

    info["versions"]["darts"] = darts.__version__
    return info


def check_autogluon(gpu: bool) -> dict:
    info = {"cuda": _cuda(gpu)}
    import numpy as np
    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    import strikecast.models.chronos  # noqa: F401

    frame = pd.DataFrame(
        {
            "item_id": ["a"] * 80,
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="D"),
            "target": np.random.default_rng(0).poisson(2.0, 80).astype(float),
        }
    )
    data = TimeSeriesDataFrame.from_data_frame(frame)
    with tempfile.TemporaryDirectory() as tmp:
        predictor = TimeSeriesPredictor(prediction_length=7, path=tmp, verbosity=0)
        predictor.fit(data, hyperparameters={"Chronos2": {}}, enable_ensemble=False)
        forecast = predictor.predict(data)
    if len(forecast) != 7:
        raise SystemExit(f"smoke test: Chronos-2 returned {len(forecast)} rows, expected 7")
    import autogluon.timeseries as agts

    info["versions"] = {"autogluon.timeseries": getattr(agts, "__version__", "?")}
    info["hf_home"] = os.environ.get("HF_HOME")
    return info


def write_stamp(target: str, info: dict) -> Path:
    stamp = VENVS[target] / STAMP
    payload = {
        "target": target,
        "lock_sha256": sha256(LOCKS[target]),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        **info,
    }
    stamp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=sorted(VENVS))
    parser.add_argument("--no-gpu", action="store_true", help="skip the GPU checks (laptop)")
    parser.add_argument("--no-stamp", action="store_true", help="do not write the stamp")
    args = parser.parse_args(argv)

    gpu = not args.no_gpu
    info = check_main(gpu) if args.target == "main" else check_autogluon(gpu)
    print(json.dumps(info, indent=2))
    if not args.no_stamp:
        print(f"stamp: {write_stamp(args.target, info)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
