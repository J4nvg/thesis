#!/usr/bin/env python3
"""Import the thesis's Chronos-2 fine-tune study into the run store (audit C15).

``scripts/import_golden_params.py`` copies the converted GBDT/RNN studies from
``golden/converted/tuning/``. The Chronos-2 study was never converted: its only
records are

* ``golden/checkpoints/chronos2_best/best_params.json`` -- the sidecar
  ``_chronos2.py:672-682`` writes next to the refit predictor
  (``{"best_params": {fine_tune_lr, fine_tune_steps}, "best_val_mase": null,
  "n_trials": 12, ...}``; ``best_val_mase`` is ``null`` because the Optuna
  pickle already existed when the sidecar was written), and
* ``golden/results/chronos2/optuna_trials.csv`` -- ``study.trials_dataframe(
  attrs=("number", "value", "params", "state"))`` of the same 12-trial study.

This script writes them in the format every other study uses, to

    <store-root>/chronos2/tuning/chronos2_fine_tuned/{best_params.json,trials.csv}

so ``strikecast.pipeline.run_stage.resolve_params`` (and therefore
``strikecast run experiment=chronos2 model=chronos2_fine_tuned``) picks the
thesis's winner up exactly like a freshly tuned study:
``fine_tune_lr = 8.721349828452045e-05``, ``fine_tune_steps = 1500``.

``best_params.json`` carries :data:`strikecast.tuning.BEST_PARAMS_KEYS`:
``best_params`` is copied verbatim from the sidecar; ``best_value`` is the
winning trial's value from ``optuna_trials.csv`` (the sidecar has ``null``);
the script refuses if the CSV's best trial does not carry the sidecar's params.
``sampler``/``pruner`` are what the legacy ``create_study`` got
(``TPESampler(seed=42)``, Optuna's default ``MedianPruner``, never consulted).
``trials.csv`` is ``optuna_trials.csv`` copied byte for byte (it has the five
``attrs`` columns, not the full ``trials_dataframe()`` header).

D4 (publication runs re-tune every model): this import is the THESIS
reproduction (legacy store root); the publication store gets a fresh study from
``strikecast tune experiment=chronos2 model=chronos2_fine_tuned``.

Idempotent, like ``import_golden_params.py``: identical files are left alone,
different ones are only overwritten with ``--force`` (exit code 1 otherwise).

Usage::

    python scripts/import_golden_chronos.py --dry-run
    python scripts/import_golden_chronos.py --store-root runs_legacy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

EXPERIMENT = "chronos2"
MODEL = "chronos2_fine_tuned"
SIDECAR = Path("checkpoints") / "chronos2_best" / "best_params.json"
TRIALS = Path("results") / "chronos2" / "optuna_trials.csv"
PARAM_KEYS = ("fine_tune_lr", "fine_tune_steps")

DEFAULT_GOLDEN = Path("golden")
DEFAULT_STORE_ROOT = Path("runs")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_trials(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_payload(golden: Path) -> tuple[dict, bytes]:
    """``(best_params.json payload, trials.csv bytes)`` from the golden files."""
    sidecar = json.loads((golden / SIDECAR).read_text(encoding="utf-8"))
    best = sidecar.get("best_params")
    if not isinstance(best, dict) or set(best) != set(PARAM_KEYS):
        raise ValueError(f"{golden / SIDECAR}: best_params must hold exactly {PARAM_KEYS}")

    trials_path = golden / TRIALS
    rows = read_trials(trials_path)
    complete = [r for r in rows if r.get("state") == "COMPLETE" and r.get("value")]
    if not complete:
        raise ValueError(f"{trials_path}: no COMPLETE trial")
    winner = min(complete, key=lambda r: float(r["value"]))
    got_lr = float(winner["params_fine_tune_lr"])
    got_steps = int(float(winner["params_fine_tune_steps"]))
    if not (
        math.isclose(got_lr, float(best["fine_tune_lr"]), rel_tol=0, abs_tol=0)
        and got_steps == int(best["fine_tune_steps"])
    ):
        raise ValueError(
            f"the best trial of {trials_path} (lr={got_lr!r}, steps={got_steps}) does not "
            f"carry the sidecar's params {best}"
        )
    if int(sidecar.get("n_trials", len(rows))) != len(rows):
        raise ValueError(
            f"sidecar records n_trials={sidecar.get('n_trials')} but {trials_path} has "
            f"{len(rows)} rows"
        )

    def count(state: str) -> int:
        return sum(1 for r in rows if r.get("state") == state)

    payload = {
        "variant": MODEL,
        "source_dir": str(SIDECAR.parent),
        "source_file": SIDECAR.name,
        "best_params": {"fine_tune_lr": best["fine_tune_lr"], "fine_tune_steps": best["fine_tune_steps"]},
        "best_value": float(winner["value"]),
        "n_trials": len(rows),
        "n_pruned": count("PRUNED"),
        "n_complete": count("COMPLETE"),
        "n_failed": count("FAIL"),
        "sampler": "TPESampler",
        "pruner": "MedianPruner",
        "direction": "MINIMIZE",
        "study_name": None,
    }
    return payload, trials_path.read_bytes()


def _write(target: Path, data: bytes, *, dry_run: bool, force: bool) -> tuple[str, str | None]:
    if target.is_file():
        if _digest(target.read_bytes()) == _digest(data):
            return "unchanged", None
        if not force:
            return "conflict", f"{target} exists with different contents; re-run with --force"
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
    return "copied", None


def import_chronos(
    golden: Path, store_root: Path, *, dry_run: bool = False, force: bool = False
) -> tuple[str, list[str]]:
    """Write both files. Returns ``(status, messages)``; status as in the GBDT importer."""
    payload, trials = build_payload(golden)
    destination = store_root / EXPERIMENT / "tuning" / MODEL
    text = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    statuses: list[str] = []
    messages: list[str] = []
    for name, data in (("best_params.json", text), ("trials.csv", trials)):
        status, message = _write(destination / name, data, dry_run=dry_run, force=force)
        statuses.append(status)
        if message:
            messages.append(message)
    if "conflict" in statuses:
        return "conflict", messages
    return ("copied" if "copied" in statuses else "unchanged"), messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, type=Path, help="golden/ directory")
    parser.add_argument(
        "--store-root", default=DEFAULT_STORE_ROOT, type=Path, help="run store root (default: runs)"
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite differing files")
    args = parser.parse_args(argv)

    try:
        status, messages = import_chronos(
            args.golden, args.store_root, dry_run=args.dry_run, force=args.force
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for message in messages:
        print(message)
    where = args.store_root / EXPERIMENT / "tuning" / MODEL
    print(f"{EXPERIMENT}/{MODEL}: {status}{' (dry run)' if args.dry_run else ''} -> {where}")
    return 1 if status == "conflict" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
