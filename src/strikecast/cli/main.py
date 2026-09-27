"""``strikecast``: the Hydra entry point (plan §5.6, Appendix C).

Five subcommands::

    strikecast tune     experiment=count  model=catboost_tweedie
    strikecast run      experiment=count  model=lightgbm_poisson paradigm=global \
                        stage=cv,test seed=42
    strikecast run -m   experiment=count  model=lightgbm_poisson,xgboost_tweedie \
                        paradigm=global,activity,local stage=test seed=1,2,3,4,5
    strikecast evaluate experiment=count  stage=test          # metrics only
    strikecast report   experiment=count                      # §7 tables
    strikecast verify   --golden golden/                      # §6 comparisons
    strikecast featsel  experiment=count   # feature selection ONCE (needs PYTHONHASHSEED=0)
    strikecast importance experiment=count model=catboost_tweedie paradigm=activity seed=42

Argument grammar
----------------
Everything after the subcommand is ``key=value``. Four keys are **selectors**
consumed here and never passed to Hydra, because they choose what to *run*
rather than what the experiment *is*, and each accepts a comma-separated list:

``model``   registry names, default: every model in the experiment YAML
``paradigm`` global | activity | local, default: the experiment's paradigms
``stage``   cv | test, default: both
``seed``    evaluation seeds, default: ``seeds.eval_seeds``

``experiment=<name>`` names the primary config. Every other ``key=value`` is a
Hydra override and is passed through untouched (``tracking=noop``,
``seeds.eval_seeds=[1,2]``, ``store.root=/scratch/runs``, ...). When a single
paradigm is selected and ``configs/paradigm/<name>.yaml`` exists, it is ALSO
composed as a group override, so the snapshot in the run directory names the
paradigm the run actually used.

``-m`` / ``--multirun`` is accepted and is a no-op beyond documenting intent:
the selectors already expand to the full product, in this process. The Hydra
launcher path (submitit / a SLURM array over ``jobs.txt``) is P6.

Hydra lives ONLY here and in :mod:`strikecast.config.loader` (§5.6). Composition
runs with ``hydra.job.chdir=false`` and the run-store root is resolved to an
absolute path before anything is written (§9, "Hydra working-directory and
launcher quirks").
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from strikecast.config.schema import ExperimentConfig

__all__ = ["main"]

logger = logging.getLogger(__name__)

COMMANDS: tuple[str, ...] = (
    "run", "tune", "evaluate", "report", "verify", "featsel", "importance",
)

#: Selectors consumed by the CLI; never forwarded to Hydra.
SELECTORS: tuple[str, ...] = ("model", "paradigm", "stage", "seed")

#: What ``hydra.job.chdir`` must be (§9). Written out so a reader of the log
#: sees it rather than having to know Hydra's default.
CHDIR_OVERRIDE = "hydra.job.chdir=false"

#: The golden levels of §6, for ``strikecast verify``.
GOLDEN_LEVELS: tuple[tuple[str, str, str], ...] = (
    ("A", "panel", "build_panel vs get_engineered_features on the real parquet"),
    ("B", "series bundle", "values, time index and static covariates of every list"),
    ("C", "schedule", "fold t0 indices and retrain flags per family"),
    ("D", "loop equivalence", "engine vs every legacy runner on a synthetic panel"),
    ("E", "predictions", "naive/linear/ARIMA/count GBDTs vs results/*.parquet"),
    ("F", "metrics", "diff-branch GPU GBDTs and the RNNs vs results/global_*.json"),
    ("G", "feature selection", "inputs bit-identical; selection loaded from cache"),
    ("H", "hurdle and calibration", "vs results/finalhurdle/"),
    ("I", "thesis tables", "Table 4 and Appendix E recomputed from the run store"),
)

#: Which golden levels run in CI on every push (§6).
CI_LEVELS: frozenset[str] = frozenset({"A", "B", "C", "D", "G"})


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strikecast",
        description="Reproducible drone-strike forecasting experiments.",
        allow_abbrev=False,
    )
    parser.add_argument("command", choices=COMMANDS, help="what to do")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="key=value selectors and Hydra overrides (see the module docstring)",
    )
    parser.add_argument(
        "-m",
        "--multirun",
        action="store_true",
        help="documented intent; the selectors already expand the full product",
    )
    parser.add_argument("--config-dir", default=None, help="the configs/ tree to compose from")
    parser.add_argument("--store-root", default=None, help="override store.root")
    parser.add_argument("--golden", default="golden", help="verify: the golden/ directory")
    parser.add_argument("--force", action="store_true", help="recompute complete stages")
    parser.add_argument("--n-trials", type=int, default=None, help="tune: total trial budget")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="log one line every N folds (0 = off)",
    )
    parser.add_argument(
        "--allow-default-params",
        action="store_true",
        help="run: let a tunable model without best_params.json run on the spec "
        "defaults (otherwise an error, audit C17); the --benchmark pilot uses it",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="run: only the first N folds of each stage (benchmark / legacy "
        "verification); part of the stage identity",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="report: bootstrap replicates for the pairwise CIs (§7.1 default: 1000)",
    )
    parser.add_argument(
        "--no-comparisons",
        action="store_true",
        help="report: leaderboards only, no pairwise/CD/family tables",
    )
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="importance: GBDT gain only, skip the (expensive) permutation half",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="importance: joblib workers for sklearn permutation_importance "
        "(legacy -1; it never changes a value)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG"
    )
    return parser


def split_overrides(
    tokens: Sequence[str],
) -> tuple[str | None, dict[str, list[str]], list[str]]:
    """``(experiment, selectors, hydra overrides)``.

    A selector's value is split on commas; everything else is passed through
    verbatim, including values that contain commas (``seeds.eval_seeds=[1,2]``).
    """
    experiment: str | None = None
    selectors: dict[str, list[str]] = {}
    passthrough: list[str] = []
    for token in tokens:
        if "=" not in token:
            raise SystemExit(f"not a key=value override: {token!r}")
        key, value = token.split("=", 1)
        key = key.strip()
        if key == "experiment":
            experiment = value
        elif key in SELECTORS:
            selectors[key] = [v for v in (p.strip() for p in value.split(",")) if v]
        else:
            passthrough.append(token)
    return experiment, selectors, passthrough


def _load(
    experiment: str | None,
    overrides: list[str],
    selectors: dict[str, list[str]],
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Compose one experiment into a validated :class:`ExperimentConfig`."""
    from strikecast.config.loader import default_config_dir, load_experiment  # noqa: PLC0415

    if not experiment:
        raise SystemExit("experiment=<name> is required (e.g. experiment=count)")

    config_dir = Path(args.config_dir) if args.config_dir else default_config_dir()
    composed = list(overrides)

    if args.store_root:
        # `++` sets the key whether or not the YAML already has a `store:`
        # block, so `--store-root` works against every experiment file.
        composed.append(f"++store.root={args.store_root}")
    if not any(o.startswith("hydra.job.chdir") for o in composed):
        composed.append(CHDIR_OVERRIDE)

    # A single selected paradigm is also a Hydra group, so the snapshot in the
    # run directory names the paradigm the run actually used. Several paradigms
    # stay a CLI-side product. An experiment whose defaults list has no
    # `paradigm` entry cannot take the override; it then stays a selector only.
    paradigms = selectors.get("paradigm") or []
    group = None
    if (
        len(paradigms) == 1
        and (config_dir / "paradigm" / f"{paradigms[0]}.yaml").is_file()
        and not any(o.startswith("paradigm=") for o in composed)
    ):
        group = f"paradigm={paradigms[0]}"

    if group is None:
        return load_experiment(experiment, composed, config_dir=config_dir)
    try:
        return load_experiment(experiment, [*composed, group], config_dir=config_dir)
    except Exception as exc:  # noqa: BLE001 - Hydra raises its own exception type
        logger.info("%s is not composable (%s); using it as a selector only", group, exc)
        return load_experiment(experiment, composed, config_dir=config_dir)


def _store(cfg: ExperimentConfig, args: argparse.Namespace):
    from strikecast.pipeline.context import resolve_store_root  # noqa: PLC0415
    from strikecast.store import RunStore  # noqa: PLC0415

    return RunStore(resolve_store_root(cfg, args.store_root))


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def _cmd_run(cfg: ExperimentConfig, selectors: dict[str, list[str]], args) -> int:
    from strikecast.pipeline import data_stage, run_stage  # noqa: PLC0415

    store = _store(cfg, args)
    data = data_stage.prepare_data(cfg, store)
    outcomes = run_stage.run_experiment(
        cfg,
        data=data,
        models=selectors.get("model"),
        paradigms=selectors.get("paradigm"),
        seeds=[int(s) for s in selectors["seed"]] if "seed" in selectors else None,
        stages=tuple(selectors.get("stage", ("cv", "test"))),
        store=store,
        force=args.force,
        progress_every=args.progress_every,
        allow_default_params=args.allow_default_params,
        max_folds=args.max_folds,
    )
    ran = [o for o in outcomes if not o.skipped]
    for outcome in outcomes:
        print(
            f"{'skip' if outcome.skipped else 'done'}  {outcome.label}"
            f"  folds={outcome.n_folds} rows={outcome.n_rows}"
        )
    print(f"{len(ran)} stage(s) run, {len(outcomes) - len(ran)} skipped; store={store.root}")
    return 0


def _cmd_tune(cfg: ExperimentConfig, selectors: dict[str, list[str]], args) -> int:
    from strikecast.pipeline import data_stage, tune_stage  # noqa: PLC0415

    store = _store(cfg, args)
    data = data_stage.prepare_data(cfg, store)
    outcomes = tune_stage.tune_experiment(
        cfg,
        data=data,
        models=selectors.get("model"),
        store=store,
        n_trials=args.n_trials,
        force=args.force,
    )
    for outcome in outcomes:
        state = f"skip ({outcome.reason})" if outcome.skipped else f"best={outcome.best_value}"
        print(f"{outcome.model}: {state}")
    return 0


def _cmd_evaluate(cfg: ExperimentConfig, selectors: dict[str, list[str]], args) -> int:
    """Recompute the metric views from the predictions already in the store.

    No model is built and no fold is re-predicted: this is the cheap half of
    ``run``, for when a metric changes or a stage's metrics were lost.
    """
    from strikecast.pipeline import data_stage  # noqa: PLC0415
    from strikecast.pipeline.run_stage import _evaluate  # noqa: PLC0415
    from strikecast.store import RunKey  # noqa: PLC0415

    store = _store(cfg, args)
    data = data_stage.prepare_data(cfg, store)
    stages = tuple(selectors.get("stage", ("cv", "test")))
    models = selectors.get("model") or cfg.model_names
    paradigms = selectors.get("paradigm") or [str(p) for p in cfg.paradigm_names]
    seeds = [int(s) for s in selectors.get("seed", [])] or list(cfg.seeds.eval_seeds)

    n = 0
    for model in models:
        for paradigm in paradigms:
            for seed in seeds:
                key = RunKey(cfg.name, model, paradigm, int(seed))
                for stage in stages:
                    if not store.part_paths(key, stage):
                        continue
                    preds = store.load_predictions(key, stage, legacy_order=True)
                    views = _evaluate(cfg, data, cfg.stage(stage), paradigm, preds)
                    store.write_metrics(key, stage, views)
                    print(f"evaluated {key.relative()}/{stage}  rows={len(preds)}")
                    n += 1
    print(f"{n} stage(s) re-evaluated")
    return 0


def _cmd_report(cfg: ExperimentConfig, selectors: dict[str, list[str]], args) -> int:
    """``strikecast report``: every plan §7 table for one experiment.

    The selectors keep their meaning: ``seed=`` is the evaluation-seed list of
    §7.1 and its FIRST entry is also the single seed the leaderboard and the
    pairwise comparison are built from (the leaderboard has no seed column, §7's
    Q2), ``model=`` and ``paradigm=`` restrict which runs are compared, and
    ``stage=`` restricts which stages are read.
    """
    from strikecast.pipeline import report_stage  # noqa: PLC0415

    store = _store(cfg, args)
    seeds = [int(s) for s in selectors.get("seed", [])] or None
    written = report_stage.report(
        cfg,
        store=store,
        seed=seeds[0] if seeds else 42,
        eval_seeds=seeds,
        stages=tuple(selectors.get("stage", ("cv", "test"))),
        models=selectors.get("model") or None,
        paradigms=selectors.get("paradigm") or None,
        n_boot=args.n_boot,
        comparisons=not args.no_comparisons,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


def _cmd_verify(args) -> int:
    """List the golden levels and which of them have material on disk.

    A stub, on purpose: levels A-D and G are the ``tests/golden`` and
    ``tests/equivalence`` suites and run under pytest today, and E/F/H/I need
    the real data plus the stored best params. This reports what exists so the
    next step is obvious.
    """
    golden = Path(args.golden).expanduser().resolve()
    print(f"golden material: {golden}{'' if golden.is_dir() else '  (MISSING)'}")
    present = {
        "E": golden / "results",
        "F": golden / "results",
        "G": golden / "converted" / "feature_sets",
        "H": golden / "results" / "finalhurdle",
        "I": golden / "results",
    }
    for level, what, how in GOLDEN_LEVELS:
        where = present.get(level)
        if level in CI_LEVELS:
            state = "pytest (tests/golden, tests/equivalence)"
        elif where is not None and where.is_dir():
            state = f"on disk: {where}"
        else:
            state = "material MISSING"
        print(f"  {level}  {what:<24}  {state}")
        print(f"     {how}")
    print(
        "\nLevels A-D and G run with: "
        ".venv/bin/python -m pytest tests/golden tests/equivalence -q -p no:cacheprovider"
    )
    print("Levels E, F, H and I are on-demand comparisons and are not implemented yet.")
    return 0


def _cmd_featsel(cfg: ExperimentConfig, selectors: dict[str, list[str]], args) -> int:
    """``strikecast featsel experiment=<name>``: the feature selection, ONCE.

    Computes (or, when already cached, loads and provenance-checks) the
    selection of every head -- one for count/diff, ``regressor`` +
    ``classifier`` for the hurdle, one per key for damage -- and writes it to
    ``<store>/<experiment>/shared/feature_selection.<hash>.json``. Later
    cv/tune/test jobs with ``feature_selection.require_cached`` load it and
    refuse to re-select (audit 2026-09-26 A12/D2). Needs ``PYTHONHASHSEED=0``.
    ``--force`` recomputes. Pass the same overrides (``legacy=<name>``, ...)
    and ``--store-root`` as the jobs that will read it.
    """
    from strikecast.pipeline import data_stage  # noqa: PLC0415

    store = _store(cfg, args)
    sets = data_stage.select_features(cfg, store, force=args.force)
    for head, found in sets.items():
        print(
            f"{cfg.name}{'/' + head if head else ''}: {found.source}  "
            f"past={len(found.past_keep)} future={len(found.future_keep)}  {found.path}"
        )
    return 0


def _cmd_importance(cfg: ExperimentConfig, selectors: dict[str, list[str]], args) -> int:
    """``strikecast importance``: feature importance of fitted models (audit C14).

    GBDT gain + permutation per horizon (``_regression_GBDT.ipynb`` cells
    57-58) or Chronos-2 permutation (``_chronos2.py:686``), written next to the
    run (``<run>/importance/``) and collected per seed into
    ``<store>/<experiment>/report/importance/seed=<s>/``. Without ``model=`` /
    ``paradigm=`` it runs what the thesis computed
    (``importance_stage.DEFAULT_JOBS``). ``seed=`` defaults to the TUNING seed:
    the thesis computed importances once, not per evaluation seed. Chronos-2
    needs ``envs/autogluon`` and a completed test stage of the same seed.
    """
    from strikecast.pipeline import data_stage, importance_stage  # noqa: PLC0415

    store = _store(cfg, args)
    seeds = [int(s) for s in selectors.get("seed", [])] or [int(cfg.seeds.tuning_seed)]
    if "model" in selectors:
        jobs = {
            m: tuple(selectors.get("paradigm") or [str(p) for p in cfg.paradigm_names])
            for m in selectors["model"]
        }
    else:
        jobs = importance_stage.default_jobs(cfg.name)
        if "paradigm" in selectors:
            jobs = {
                m: tuple(p for p in ps if p in selectors["paradigm"]) for m, ps in jobs.items()
            }
    if not jobs:
        raise SystemExit(
            f"no default importance jobs for experiment {cfg.name!r}; pass model=... paradigm=..."
        )
    data = data_stage.prepare_data(cfg, store)
    n_run = 0
    for seed in seeds:
        for model, paradigms in jobs.items():
            for paradigm in paradigms:
                outcome = importance_stage.run_importance(
                    cfg,
                    model,
                    paradigm,
                    seed,
                    data,
                    store=store,
                    force=args.force,
                    permutation=not args.no_permutation,
                    n_jobs=args.n_jobs,
                )
                n_run += not outcome.skipped
                state = "skip" if outcome.skipped else f"done {outcome.seconds:.0f}s"
                print(f"{state}  {outcome.label}  rows={outcome.n_rows}  {outcome.path}")
        # Every COMPLETED importance run of the experiment, not just this
        # invocation's: separate per-model jobs must not overwrite each other.
        written = importance_stage.collect_importance(store, cfg, seed)
        for name, path in written.items():
            print(f"{name}: {path}")
    print(f"{n_run} importance run(s) computed; store={store.root}")
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )
    return _run_interruptible(args)


def _run_interruptible(args: argparse.Namespace) -> int:
    """Dispatch under the SIGTERM/SIGUSR1 handlers of audit C21.

    A signal (SLURM wall clock, requeue, ``scancel``) marks every stage this
    process started ``interrupted`` instead of leaving it ``running``, and the
    process exits with ``EXIT_INTERRUPTED`` (99). ``tune`` exits at once so the
    running Optuna trial is not counted (see ``strikecast.pipeline.interrupt``).
    """
    from strikecast.pipeline import interrupt  # noqa: PLC0415

    previous = interrupt.install(args.command)
    try:
        return _dispatch(args)
    except interrupt.StageInterrupted as exc:
        from strikecast.store.run_store import interrupt_active_stages  # noqa: PLC0415

        marked = interrupt_active_stages(str(exc))
        print(f"{exc}: {len(marked)} stage(s) marked interrupted", file=sys.stderr)
        return interrupt.EXIT_INTERRUPTED
    finally:
        interrupt.restore(previous)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "verify":
        return _cmd_verify(args)

    experiment, selectors, overrides = split_overrides(args.overrides)
    cfg = _load(experiment, overrides, selectors, args)

    if args.command == "run":
        return _cmd_run(cfg, selectors, args)
    if args.command == "tune":
        return _cmd_tune(cfg, selectors, args)
    if args.command == "evaluate":
        return _cmd_evaluate(cfg, selectors, args)
    if args.command == "report":
        return _cmd_report(cfg, selectors, args)
    if args.command == "featsel":
        return _cmd_featsel(cfg, selectors, args)
    if args.command == "importance":
        return _cmd_importance(cfg, selectors, args)
    raise SystemExit(f"unknown command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
