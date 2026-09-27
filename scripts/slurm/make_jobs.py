#!/usr/bin/env python3
"""Expand an experiment sweep into a `jobs.txt` matrix for a SLURM array.

Plan §5.6: a plain SLURM array reading a `jobs.txt` matrix replaces the four
`tune_*.sh` scripts. The submitit alternative Hydra offers is NOT wired up (see
`scripts/slurm/run.sbatch`): the CLI expands its selectors in process, so a
Hydra launcher would silently run the sweep locally. Jobs declare their resource
class (`cpu`, `gpu`) from the model spec.

One line of `jobs.txt` is one array task and is exactly the argument string of
one `strikecast` invocation::

    run experiment=count model=lightgbm_poisson paradigm=global stage=test seed=1 tracking=wandb_online

`scripts/slurm/run.sbatch` reads line `$SLURM_ARRAY_TASK_ID` and executes it, so
a line is readable, reproducible and can be run by hand on the laptop.

What this generator knows that a shell loop does not
----------------------------------------------------
* **The lineup** comes from the composed experiment config (`models:` in
  `configs/experiment/<name>.yaml`) and every name is resolved in the model
  registry, so a typo fails here rather than in task 217 of an array.
* **§5.4's seed rules**: the `cv` stage is single-seed (the tuning seed) and a
  `stochastic=False` model (naives, linear, ARIMA) runs once, under the first
  evaluation seed, instead of being swept.
* **The resource class** is `gpu` when the resolved device for that model is
  not the CPU, else `cpu`. The config wins over `ModelSpec.device`, because
  `ExperimentConfig.device_for` is what the pipeline itself builds the model
  with (§1 "Device policy"). A SLURM array is homogeneous, so the two classes
  are written to two files and submitted as two arrays.
* **The thesis-faithful matrix** (audit 2026-09-26 C16/D8). Without a
  `paradigm=` selector each model runs in the paradigms its experiment entry
  lists (`models: - {name: ..., paradigms: [...]}`): count GBDTs x 3, count RNNs
  x Global/Activity, diff tuned models x 3, diff baselines Global only, the
  hurdle composite x 3 and its components never standalone. Without an explicit
  `stage=` the cv stage is emitted only where a later stage NEEDS it (`--cv
  needed`: a composite with calibration fits its calibrators on the CV rows);
  `--cv all` restores cv for every model, and `stage=cv` asks for it
  explicitly. A stage the experiment does not define (Chronos-2 has no cv) is
  never emitted.
* **Work already done is skipped**: a stage that the run store reports
  `complete` is not re-queued, and a model whose `tuning/<model>/best_params.json`
  exists is not re-tuned (Appendix C). The check is by run key and stage name,
  NOT by the full stage identity (recomputing the stage hash would mean
  building the panel, the bundle and the feature selection for every line),
  but it never trusts a stage that (a) ran a tunable model on the spec
  defaults, (b) finished before the model's `best_params.json` was written, or
  (c) ran only its first `max_folds` folds (audit C17). Pass `--force` to emit
  everything; the pipeline's own identity check still skips a complete stage
  at run time, so a stale line costs a process start.
* **The environment**: Chronos-2 lines run in `envs/autogluon` (`Job.env`).

Usage
-----
::

    # the thesis matrix of one experiment (per-model paradigms, cv only where needed)
    .venv/bin/python scripts/slurm/make_jobs.py experiment=count seed=42 --out -

    # the count family's test stage over three paradigms and five seeds
    uv run python scripts/slurm/make_jobs.py experiment=count stage=test \
        paradigm=global,activity,local seed=42,1,2,3,4 --out jobs

    # everything that still needs tuning
    uv run python scripts/slurm/make_jobs.py experiment=count stage=tune --out tune_jobs

    # a single model, and a Hydra override forwarded to every line
    uv run python scripts/slurm/make_jobs.py experiment=diff model=catboost_mse \
        stage=cv,test tracking=wandb_offline --out jobs

`--out jobs` writes `jobs.cpu.txt` and/or `jobs.gpu.txt` (only the non-empty
ones) and prints the `sbatch` command for each. `--out -` prints the lines to
stdout instead, one per line, prefixed with their resource class.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

    from strikecast.config.schema import ExperimentConfig

#: Selectors the CLI consumes; everything else on the command line is a Hydra
#: override and is forwarded to every job line verbatim.
SELECTORS: tuple[str, ...] = ("experiment", "model", "paradigm", "stage", "seed")

#: The stages a job line can ask for. `tune` is a different subcommand, not a
#: backtest stage, which is why it is handled separately below.
STAGES: tuple[str, ...] = ("tune", "cv", "test")

RESOURCES: tuple[str, ...] = ("cpu", "gpu")

#: `--cv` policies: every model, or only where a later stage needs the CV rows.
CV_POLICIES: tuple[str, ...] = ("all", "needed")


@dataclass(frozen=True)
class Job:
    """One array task: a resource class and the `strikecast` arguments.

    ``experiment``/``paradigm``/``seed``/``family``/``kind``/``env`` are metadata
    for `submit_all.py` (dependencies, time classes, the Python environment);
    they are not part of the command line.
    """

    resource: str
    command: str
    model: str
    stage: str
    experiment: str = ""
    paradigm: str | None = None
    seed: int | None = None
    family: str = ""
    kind: str = ""
    env: str = "main"
    tunable: bool = False

    def __str__(self) -> str:
        return self.command


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def split_args(argv: Sequence[str]) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Split `key=value` arguments into selectors, overrides and flags.

    Returns `(selectors, overrides, flags)` where `selectors` maps a selector
    name to its comma-split values, `overrides` are the remaining `key=value`
    strings (forwarded to Hydra, and therefore onto every job line) and `flags`
    is everything that is not a `key=value` pair (parsed by argparse).
    """
    selectors: dict[str, list[str]] = {}
    overrides: list[str] = []
    flags: list[str] = []
    for arg in argv:
        if "=" not in arg or arg.startswith("-"):
            flags.append(arg)
            continue
        key, _, value = arg.partition("=")
        if key in SELECTORS:
            selectors[key] = [v for v in value.split(",") if v]
        else:
            overrides.append(arg)
    return selectors, overrides, flags


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_jobs.py",
        description="Expand experiment x model x paradigm x seed x stage into jobs.txt",
        allow_abbrev=False,
    )
    parser.add_argument("--out", default="jobs", help="basename, or - for stdout")
    parser.add_argument("--store-root", default=None, help="run store root (default: config)")
    parser.add_argument("--config-dir", default=None, help="the configs/ tree to compose from")
    parser.add_argument(
        "--resource",
        choices=RESOURCES,
        default=None,
        help="emit only jobs of this resource class",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="emit complete stages and already tuned models too",
    )
    parser.add_argument(
        "--cv",
        choices=CV_POLICIES,
        default=None,
        help="cv lines for every model (all) or only where a later stage needs them "
        "(needed; the default unless stage= names cv explicitly)",
    )
    return parser


# --------------------------------------------------------------------------- #
# the matrix
# --------------------------------------------------------------------------- #
def resource_class(cfg: ExperimentConfig, model: str, spec: object) -> str:
    """`gpu` when this model runs off the CPU, else `cpu` (§5.6).

    The config is asked first (`ExperimentConfig.device_for`, which is what the
    pipeline builds the model with) and `ModelSpec.device` is the fallback for
    a model the experiment YAML does not pin.
    """
    device = None
    try:
        device = cfg.device_for(model)
    except Exception:  # a model that is not in this experiment's `models:`
        device = None
    if not device:
        device = str(getattr(spec, "device", "cpu"))
    return "cpu" if str(device).lower() in {"", "cpu", "none"} else "gpu"


def seeds_for(stage: str, cfg: ExperimentConfig, seeds: Sequence[int], stochastic: bool) -> list[int]:
    """§5.4: cv is single-seed; a deterministic model is not swept."""
    if stage == "cv":
        return [int(cfg.seeds.tuning_seed)]
    if not stochastic:
        return [int(seeds[0])] if seeds else []
    return [int(s) for s in seeds]


def paradigms_for(cfg: ExperimentConfig, model: str) -> list[str]:
    """The thesis-faithful paradigms of one model (audit C16/D8).

    The experiment entry's ``paradigms`` list when it has one (``[]`` = not a
    standalone job), else the experiment's own paradigms.
    """
    listed = None
    entry_for = getattr(cfg, "model_entry", None)
    if entry_for is not None:
        try:
            listed = getattr(entry_for(model), "paradigms", None)
        except KeyError:
            listed = None
    if listed is not None:
        return [str(p) for p in listed]
    return [str(p) for p in getattr(cfg, "paradigm_names", ["global"])]


def cv_needed(cfg: ExperimentConfig, spec: object) -> bool:
    """True when a later stage reads this model's CV rows.

    Only the composites with calibration do: the hurdle's test stage applies
    calibrators fit on the CV predictions of the tuning seed (audit C18/D8).
    Every other model's CV stage feeds no thesis table (audit B §matrix).
    """
    calibration = getattr(cfg, "calibration", None)
    enabled = bool(getattr(calibration, "enabled", False))
    return getattr(spec, "kind", "") == "composite" and enabled


def env_for(cfg: ExperimentConfig, spec: object) -> str:
    """``autogluon`` for the Chronos-2 family (its own venv, C15), else ``main``."""
    variant = getattr(getattr(cfg, "data", None), "panel_variant", None)
    return "autogluon" if getattr(spec, "kind", "") == "chronos" or variant == "chronos" else "main"


def _has_stage(cfg: ExperimentConfig, stage: str) -> bool:
    stages = getattr(cfg, "stages", None)
    return True if not stages else stage in stages


def lookup_spec(get_spec, model: str, experiment: str):
    """``get_spec`` with the Chronos-2 specs imported on demand.

    ``strikecast.models.registry`` does not import ``strikecast.models.chronos``
    (it would be the only family whose module belongs to another venv), so the
    first lookup of a ``chronos2_*`` name imports it; the module keeps its
    AutoGluon imports inside functions, so this works in the main env.
    """
    try:
        return get_spec(model, experiment)
    except KeyError:
        import importlib  # noqa: PLC0415

        try:
            importlib.import_module("strikecast.models.chronos")
        except ImportError:
            raise KeyError(model) from None
        return get_spec(model, experiment)


def build_jobs(
    cfg: ExperimentConfig,
    *,
    models: Sequence[str],
    paradigms: Sequence[str] | None,
    stages: Sequence[str],
    seeds: Sequence[int],
    overrides: Sequence[str] = (),
    store: object | None = None,
    force: bool = False,
    get_spec=None,
    cv: str = "all",
) -> list[Job]:
    """The full matrix for one experiment, minus what is already done.

    ``paradigms=None`` is the thesis matrix: each model in its own
    :func:`paradigms_for`; a list applies to every model. ``cv="needed"``
    keeps the cv stage only for models :func:`cv_needed` says a later stage
    reads.
    """
    from strikecast.pipeline.context import get_spec as _default_get_spec  # noqa: PLC0415
    from strikecast.store import RunKey  # noqa: PLC0415

    get_spec = get_spec or _default_get_spec
    if cv not in CV_POLICIES:
        raise ValueError(f"cv={cv!r}; expected one of {CV_POLICIES}")
    tail = " ".join(overrides)
    jobs: list[Job] = []
    for model in models:
        spec = lookup_spec(get_spec, model, cfg.name)
        resource = resource_class(cfg, model, spec)
        meta = {
            "experiment": str(cfg.name),
            "family": str(getattr(spec, "family", "")),
            "kind": str(getattr(spec, "kind", "")),
            "env": env_for(cfg, spec),
            "tunable": bool(getattr(spec, "tunable", False)),
        }
        model_paradigms = list(paradigms) if paradigms is not None else paradigms_for(cfg, model)
        for stage in stages:
            if stage == "tune":
                if not getattr(spec, "tunable", False):
                    continue
                if paradigms is None and not model_paradigms:
                    continue  # a component: tuned (if ever) inside its composite
                if not force and store is not None and _is_tuned(store, cfg.name, model):
                    continue
                parts = [f"tune experiment={cfg.name}", f"model={model}"]
                jobs.append(Job(resource, _line(parts, tail), model, stage, **meta))
                continue
            if not _has_stage(cfg, stage):
                continue
            if stage == "cv" and cv == "needed" and not cv_needed(cfg, spec):
                continue
            for paradigm in model_paradigms:
                for seed in seeds_for(stage, cfg, seeds, bool(getattr(spec, "stochastic", True))):
                    key = RunKey(cfg.name, model, str(paradigm), int(seed))
                    if not force and store is not None and stage_done(store, key, stage, spec):
                        continue
                    parts = [
                        f"run experiment={cfg.name}",
                        f"model={model}",
                        f"paradigm={paradigm}",
                        f"stage={stage}",
                        f"seed={seed}",
                    ]
                    jobs.append(
                        Job(
                            resource,
                            _line(parts, tail),
                            model,
                            stage,
                            paradigm=str(paradigm),
                            seed=int(seed),
                            **meta,
                        )
                    )
    return jobs


def stage_done(store: object, key: object, stage: str, spec: object) -> bool:
    """``complete`` in the store AND trustworthy (audit C17).

    Not done when the stage ran a tunable model on the spec defaults, when it
    finished before ``best_params.json`` was (re)written, or when it ran only
    its first ``max_folds`` folds. The full stage identity is the pipeline's
    own check at run time; this is the cheap generator-side approximation.
    """
    state = store.read_state(key, stage)  # type: ignore[attr-defined]
    if state is None or state.status != "complete":
        return False
    if getattr(state, "max_folds", None) is not None:
        return False
    if getattr(spec, "tunable", False):
        if getattr(state, "params_source", None) == "defaults":
            return False
        best = Path(store.tuning_dir(key.experiment, key.model)) / "best_params.json"  # type: ignore[attr-defined]
        finished = _parse_time(getattr(state, "finished", None))
        if best.is_file() and finished is not None and best.stat().st_mtime > finished:
            return False
    return True


def _parse_time(stamp: str | None) -> float | None:
    if not stamp:
        return None
    import datetime as _dt  # noqa: PLC0415

    try:
        return _dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _line(parts: Sequence[str], tail: str) -> str:
    line = " ".join(parts)
    return f"{line} {tail}".strip() if tail else line


def _is_tuned(store: object, experiment: str, model: str) -> bool:
    directory = store.tuning_dir(experiment, model)  # type: ignore[attr-defined]
    return (Path(directory) / "best_params.json").is_file()


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def write_jobs(jobs: Iterable[Job], out: str | Path) -> dict[str, Path]:
    """Write one file per resource class; returns `{resource: path}`.

    A file is written only when that class has jobs, and it always ends with a
    newline: the array templates index it with `sed -n "${i}p"`, so a missing
    final newline would silently drop the last task.
    """
    by_resource: dict[str, list[Job]] = {}
    for job in jobs:
        by_resource.setdefault(job.resource, []).append(job)
    written: dict[str, Path] = {}
    base = Path(out)
    for resource, group in sorted(by_resource.items()):
        path = base.with_name(f"{base.name}.{resource}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{job.command}\n" for job in group), encoding="utf-8")
        written[resource] = path
    return written


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selectors, overrides, flags = split_args(argv)
    args = _parser().parse_args(flags)

    if "experiment" not in selectors:
        print("make_jobs.py: experiment=<name> is required", file=sys.stderr)
        return 2

    from strikecast.config.loader import load_experiment  # noqa: PLC0415
    from strikecast.store import RunStore  # noqa: PLC0415

    stages = selectors.get("stage") or ["cv", "test"]
    # An explicit `stage=cv` means cv for every selected model; the default
    # stage list is the thesis matrix, where cv runs only where it is needed.
    cv_policy = args.cv or ("all" if "stage" in selectors else "needed")
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"make_jobs.py: unknown stage(s) {unknown}; known: {list(STAGES)}", file=sys.stderr)
        return 2

    all_jobs: list[Job] = []
    for experiment in selectors["experiment"]:
        cfg = load_experiment(experiment, overrides, config_dir=args.config_dir)
        root = args.store_root or cfg.store.root
        store = RunStore(Path(root).expanduser().resolve())
        all_jobs.extend(
            build_jobs(
                cfg,
                models=selectors.get("model") or cfg.model_names,
                paradigms=selectors.get("paradigm") or None,
                stages=stages,
                seeds=[int(s) for s in selectors.get("seed", [])] or cfg.seeds.eval_seeds,
                overrides=overrides,
                store=store,
                force=args.force,
                cv=cv_policy,
            )
        )

    if args.resource:
        all_jobs = [j for j in all_jobs if j.resource == args.resource]

    if not all_jobs:
        print("nothing to do: every stage is complete (use --force to re-emit)", file=sys.stderr)
        return 0

    if args.out == "-":
        for job in all_jobs:
            print(f"{job.resource}\t{job.command}")
        return 0

    written = write_jobs(all_jobs, args.out)
    template = {"tune": "tune.sbatch"}.get(stages[0], "run.sbatch") if len(stages) == 1 else None
    for resource, path in written.items():
        n = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        script = template or ("tune.sbatch" if "tune" in stages else "run.sbatch")
        print(f"{path}: {n} {resource.upper()} job(s)")
        # C19/C27: GPU tasks need the GRES (and, on this cluster, disable-binding
        # to get more than 4 CPUs); the log directory must exist before sbatch.
        gres = " --gres=gpu:1 --gres-flags=disable-binding -c 8" if resource == "gpu" else " -c 48"
        print(
            f"  mkdir -p logs/slurm && sbatch --array=1-{n} --partition=GPU{gres} "
            f"scripts/slurm/{script} {path}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
