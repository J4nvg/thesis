#!/usr/bin/env python3
"""Legacy-mode fold-subset verification of the port (audit 2026-09-26 D9).

For every model of the thesis job matrix: run the first ``--windows`` retrain
windows (x 7 folds) in legacy mode (``legacy=<experiment>``: legacy expdecay7,
the thesis' feature-selection caches, the thesis' tuned params) into a
SEPARATE store root, and compare the fold predictions with the golden ones.
See :mod:`strikecast.verification.legacy` for levels and tolerances.

    # once per store: import the thesis params, select the hurdle heads (legacy)
    .venv/bin/python scripts/verify_legacy.py prep --store-root runs_verify
    # one family (optionally one resource class, or some models)
    .venv/bin/python scripts/verify_legacy.py run --experiment=diff --resource=cpu \\
        --store-root runs_verify
    # the pass/fail table: runs_verify/_verification/verification.{csv,md}
    .venv/bin/python scripts/verify_legacy.py report --store-root runs_verify

    # laptop, one cheap model, one window (seconds):
    PYTHONHASHSEED=0 .venv/bin/python scripts/verify_legacy.py all --experiment=diff \\
        --models=linear --windows=1 --store-root runs_verify

``submit_all.py`` runs ``prep`` -> ``run`` per (experiment, resource) ->
``report`` as SLURM jobs (the Chronos-2 ``run`` in ``envs/autogluon``).
Exit status of ``run``/``all``/``report``: 1 when any case FAILed or ERRORed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # a checkout without an installed package
    sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify_legacy.py", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prep", "run", "report", "all"))
    parser.add_argument("--store-root", default="runs_verify")
    parser.add_argument("--experiment", action="append", default=None,
                        help="count, diff, hurdle, chronos2 (repeatable; default: all four)")
    parser.add_argument("--resource", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--models", default=None, help="comma-separated registry names")
    parser.add_argument("--windows", type=int, default=2, help="retrain windows (7 folds each)")
    parser.add_argument("--force", action="store_true", help="recompute complete stages")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("verify_legacy.py: set PYTHONHASHSEED=0 (F16)", file=sys.stderr)
        return 2

    from strikecast.verification import legacy  # noqa: PLC0415

    experiments = args.experiment or ["count", "diff", "hurdle", "chronos2"]
    models = [m for m in args.models.split(",") if m] if args.models else None
    failed = False
    if args.command in ("prep", "all"):
        legacy.prep(args.store_root, experiments=experiments)
    if args.command in ("run", "all"):
        for experiment in experiments:
            results = legacy.run_cases(
                experiment, store_root=args.store_root, windows=args.windows,
                resource=args.resource, models=models, force=args.force,
            )
            failed |= any(r.status in ("FAIL", "ERROR") for r in results)
    if args.command in ("report", "all"):
        csv_path, md_path, counts = legacy.write_report(args.store_root)
        print(f"{csv_path}\n{md_path}\n" + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
        failed |= bool(counts.get("FAIL") or counts.get("ERROR"))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
