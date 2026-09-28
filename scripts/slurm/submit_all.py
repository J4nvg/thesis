#!/usr/bin/env python3
"""Submit the whole thesis re-run to SLURM as one DAG of jobs (audit 2026-09-26 C30).

Run on the LOGIN node from the repository root; it only calls ``sbatch``,
``squeue`` and ``git``, never computes::

    python3 scripts/slurm/submit_all.py --setup-only        # once: build the envs (SLURM jobs)
    python3 scripts/slurm/submit_all.py --dry-run           # print the DAG and every sbatch line
    python3 scripts/slurm/submit_all.py --benchmark         # pilot: 2 retrain windows / 2 trials
    python3 scripts/slurm/submit_all.py timings             # pilot -> configs/cluster/time_table.json
    python3 scripts/slurm/submit_all.py                     # the full seed-42 DAG
    python3 scripts/slurm/submit_all.py status              # what is pending/running/done/failed
    python3 scripts/slurm/submit_all.py --seeds 1,2,3,4 --stages test   # more seeds, later

The DAG (one SLURM job per box, per-model dependency chains)::

    setup (GPU node: uv, .venv, CUDA/GBDT smoke test)
      ├─ setup_ag (envs/autogluon + Chronos-2 weights)             [chronos2 only]
      ├─ featsel:<exp> (CPU, once per family; all heads)             count, diff, hurdle
      │    └─ tune:<exp>:<model> (resumable: requeued on the wall clock)
      │         ├─ cv:<exp>:<model>:<paradigm>   (only where needed: hurdle calibrators)
      │         └─ test:<exp>:<model>:<paradigm>:s<seed>   (afterok: its own tune [+ cv])
      │              └─ importance:<exp>:<model>:<paradigm>   (thesis FI jobs)
      ├─ report:<exp>      (afterany: every job of that experiment)
      │    └─ figures      (afterany: every report; placeholder until `strikecast figures`)
      └─ verify_prep → verify:<exp>:<resource> → verify_report   (legacy-mode, own store)

A model's cv/test start as soon as ITS tune is done (``afterok``), not when
every tune is; report/figures use ``afterany`` so a partial result is still
reported (``status`` shows what failed). Jobs whose ``afterok`` dependency
fails are cancelled (``--kill-on-invalid-dep=yes``).

What is decided where
---------------------
* **The matrix** is ``make_jobs.build_jobs`` over each experiment's config:
  the per-model ``paradigms`` lists (C16/D8), cv only where needed (``--cv``),
  already complete/tuned work skipped. Damage is not in the default experiment
  list (not reported in the thesis, B19).
* **Resources** come from the model's device: GPU jobs
  ``-p GPU --gres=gpu:1 --gres-flags=disable-binding -c 8`` (``gres.conf`` binds
  both GPUs to cores 0-1, so ``-c > 4`` needs ``disable-binding``); CPU jobs
  ``-p GPU -c 16`` without gres (the configs pin 4-12 threads). Two GPU jobs plus
  three CPU jobs fit a 64-CPU node.
* **Wall time** from :data:`TIME_TABLE` (evidence in each entry), overridden by
  ``configs/cluster/time_table.json`` when present (written by ``timings`` from
  the ``--benchmark`` pilot) or ``--time-table``. A job class whose time exceeds
  ``--long-threshold`` (30 h) goes to ``GPUExtended`` unless it is resumable:
  tune jobs stay on ``GPU`` (36 h), get ``--requeue --signal=B:USR1@900`` and
  are requeued by ``job.sbatch`` before the limit, continuing the Optuna study.
* **Environment**: ``.venv`` / ``envs/autogluon/.venv`` built ONCE by the setup
  jobs; every other job runs the prebuilt interpreter (``job.sbatch``), never
  ``uv sync`` (C24). ``PYTHONHASHSEED=0`` everywhere.
* **Stores**: publication runs in ``runs_publication/``, the pilot in
  ``runs_benchmark/``, the legacy verification in ``runs_verify/`` -- legacy and
  publication runs must never share a store (tuned params, feature caches).
* **Safety**: refuses a dirty or unpushed checkout (``--allow-dirty``); creates
  ``logs/`` first (C27); writes ``logs/submissions/<timestamp>.json`` (commit,
  options, config hashes, every job id and sbatch line).

This file is standard-library only at import time. Building the matrix needs
``strikecast`` (Hydra configs + model registry): when the current interpreter
cannot import it, the script re-executes itself with ``.venv/bin/python``.
``--setup-only``, ``status`` and ``timings`` never need it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SLURM_DIR = Path(__file__).resolve().parent
JOB_SCRIPT = "scripts/slurm/job.sbatch"
SETUP_SCRIPT = "scripts/slurm/setup.sbatch"
CLI = ["-m", "strikecast.cli.main"]

DEFAULT_EXPERIMENTS = ("count", "diff", "hurdle", "chronos2")
#: Families whose data stage runs a feature selection (chronos2 has none, F28).
FEATSEL_EXPERIMENTS = frozenset({"count", "diff", "hurdle", "damage"})
ALL_STAGES = (
    "setup", "featsel", "tune", "cv", "test", "importance", "report", "figures", "verify",
)
DEFAULT_STORE = "runs_publication"
BENCHMARK_STORE = "runs_benchmark"
VERIFY_STORE = "runs_verify"
LOG_ROOT = "logs/slurm"
MANIFEST_DIR = "logs/submissions"
TIME_TABLE_FILE = "configs/cluster/time_table.json"

PARTITION = "GPU"
LONG_PARTITION = "GPUExtended"
PARTITION_LIMIT_H = 36.0
LONG_THRESHOLD_H = 30.0
SIGNAL_LEAD_S = 900
MAX_REQUEUES = 8

#: Full-stage fold counts (audit B14: 79 folds / 12 retrains in CV, 164 / 24 in
#: test), used to extrapolate the pilot's seconds per fold.
FULL_FOLDS = {"cv": 79, "test": 164}
#: Retrain stride of the shared schedule: one "window" = 7 folds.
RETRAIN_STRIDE = 7

#: SLURM resources per class. `gpu` also gets `--gres-flags=disable-binding`.
RESOURCES: dict[str, dict[str, Any]] = {
    "gpu": {"cpus": 8, "mem": "40G", "gres": "gpu:1"},
    # 16 CPUs: the GBDT families pin 4-12 threads (count 12, diff/hurdle 4), so
    # 16 leaves headroom and lets ~3 CPU jobs share a node with 2 GPU jobs.
    "cpu": {"cpus": 16, "mem": "48G", "gres": None},
    "cpu_small": {"cpus": 8, "mem": "32G", "gres": None},
}

FAMILY_GROUP = {
    "lightgbm": "gbdt", "xgboost": "gbdt", "catboost": "gbdt",
    "lstm": "rnn", "gru": "rnn",
    "linear": "baseline", "arima": "baseline", "naive": "baseline",
    "chronos": "chronos",
    "hurdle": "composite", "damage": "composite",
}

# --------------------------------------------------------------------------- #
# wall-time table: hours per job class, most specific key wins
# (stage/family/resource/paradigm > stage/family/resource > stage/group/resource/
#  paradigm > stage/group/resource > stage/group > stage). Evidence: audit C
# "Runtime estimates per job type": (a) legacy Optuna studies on GPU4EDU A40s,
# (b) the AutoGluon Chronos log, (c) laptop part mtimes (contended), (d) code
# structure (one tuning trial = one 79-fold CV backtest). Replace with the pilot
# (`--benchmark` then `timings`) before the real submission.
# --------------------------------------------------------------------------- #
TIME_TABLE: dict[str, dict[str, Any]] = {
    "setup": {"hours": 2, "evidence": "uv sync of the locked env (torch+CUDA wheels, ~5 GB) + smoke test"},
    "setup_ag": {"hours": 2, "evidence": "uv sync of envs/autogluon + Chronos-2 weights download"},
    "featsel": {"hours": 4, "evidence": "one LightGBM fit per head on the train split, CPU (minutes); margin"},
    "tune/lightgbm/cpu": {"hours": 8, "evidence": "(a) count LightGBM-P study 2.14 h, trial max 5.6 min; x3 margin"},
    "tune/xgboost/cpu": {"hours": 24, "evidence": "(a) diff XGBoost 2.12 h on GPU; CPU hist 3-5x slower -> 6-12 h, unverified"},
    "tune/catboost/cpu": {"hours": 36, "evidence": "(a) diff CatBoost 8.44 h on GPU, trial max 50.9 min; CPU 17-40 h -> requeue"},
    "tune/xgboost/gpu": {"hours": 8, "evidence": "(a) diff XGBoost study 2.12 h"},
    "tune/catboost/gpu": {"hours": 20, "evidence": "(a) diff CatBoost study 8.44 h; x2 margin"},
    "tune/rnn/gpu": {"hours": 6, "evidence": "(a) 24 RNN studies 0.28-1.49 h each"},
    "tune/chronos/gpu": {"hours": 10, "evidence": "(b) one fit at 1500 steps = 636 s on an A40; 12 trials of 200-3000 steps -> 1.5-4 h"},
    "tune": {"hours": 24, "evidence": "fallback"},
    "cv/gbdt/cpu": {"hours": 4, "evidence": "(a) a trial at best params 1-5 min; (c) count LGBM-P CV ~40 min on a laptop"},
    "cv/gbdt/cpu/activity": {"hours": 6, "evidence": "3 groups, ~1-1.5x global"},
    "cv/gbdt/cpu/local": {"hours": 12, "evidence": "20 regions; no evidence (C20), margin"},
    "test/gbdt/cpu": {"hours": 8, "evidence": "(c) count LGBM-P test 2 h 40 min on a contended laptop; 164 folds / 24 retrains"},
    "test/gbdt/cpu/activity": {"hours": 12, "evidence": "3 groups, ~1-1.5x global"},
    "test/gbdt/cpu/local": {"hours": 24, "evidence": "20 regions x 24 retrains, persisted only at the end (C20); no evidence"},
    "cv/gbdt/gpu": {"hours": 4, "evidence": "(a) diff XGB/CatBoost trial 2.5/4.4 min at best params"},
    "test/gbdt/gpu": {"hours": 8, "evidence": "~2x CV (164 vs 79 folds)"},
    "test/gbdt/gpu/activity": {"hours": 12, "evidence": "3 groups"},
    "test/gbdt/gpu/local": {"hours": 24, "evidence": "20 regions; no evidence"},
    "cv/rnn/gpu": {"hours": 4, "evidence": "(a) RNN trial 0.3-4 min"},
    "test/rnn/gpu": {"hours": 4, "evidence": "(a) RNN trial 0.3-4 min; 2-15 min per stage"},
    "test/rnn/gpu/activity": {"hours": 6, "evidence": "3 groups"},
    "test/rnn/gpu/local": {"hours": 16, "evidence": "20 regions (diff RNNs only)"},
    "cv/rnn/gpu/local": {"hours": 12, "evidence": "20 regions"},
    "cv/baseline": {"hours": 4, "evidence": "(c) ARIMA CV 14 min, naive/linear seconds"},
    "test/baseline": {"hours": 4, "evidence": "(c) ARIMA test 47 min on a laptop"},
    "cv/composite/cpu": {"hours": 8, "evidence": "hurdle: SPE + CatBoost per retrain; thesis ran it on an i7 laptop; no timing"},
    "cv/composite/cpu/activity": {"hours": 12, "evidence": "3 groups"},
    "cv/composite/cpu/local": {"hours": 24, "evidence": "20 regions; no evidence"},
    "test/composite/cpu": {"hours": 12, "evidence": "~2x CV"},
    "test/composite/cpu/activity": {"hours": 16, "evidence": "3 groups"},
    "test/composite/cpu/local": {"hours": 28, "evidence": "20 regions; no evidence -- first pilot candidate"},
    "test/chronos/gpu": {"hours": 4, "evidence": "(b) zero-shot fit 100 s, ~2 s/fold; fine-tune 636 s at 1500 steps"},
    "importance": {"hours": 16, "evidence": "C14: GBDT permutation 5 repeats x 7 horizons x ~500 features, 2-10 h"},
    "report": {"hours": 2, "evidence": "diff report ~4 s; pairwise work grows with runs^2 (count ~1275 pairs)"},
    "figures": {"hours": 2, "evidence": "C1: < 1 h"},
    "verify_prep": {"hours": 4, "evidence": "golden param import + hurdle legacy featsel (two LightGBM fits)"},
    "verify": {"hours": 8, "evidence": "first 2 retrain windows (14 folds) of every thesis-matrix model of one family"},
    "verify_report": {"hours": 1, "evidence": "reads JSON results"},
}


# --------------------------------------------------------------------------- #
# the DAG
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """One SLURM job of the DAG (before and after submission)."""

    name: str
    stage: str
    resource: str
    env: str
    argv: list[str]
    time_keys: list[str]
    script: str = JOB_SCRIPT
    after_ok: list[str] = field(default_factory=list)
    after_any: list[str] = field(default_factory=list)
    requeue: bool = False
    optional: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    # resolved by `plan_resources`
    hours: float = 0.0
    partition: str = PARTITION
    time: str = ""
    evidence: str = ""
    # filled at submission
    job_id: str | None = None
    log: str = ""
    sbatch: list[str] = field(default_factory=list)


@dataclass
class Options:
    seeds: list[int] = field(default_factory=lambda: [42])
    experiments: list[str] = field(default_factory=lambda: list(DEFAULT_EXPERIMENTS))
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    models: list[str] | None = None
    #: only these (experiment, model, paradigm) runs (``--runs`` / ``--top``); None = all
    runs: set[tuple[str, str, str]] | None = None
    cv: str = "needed"
    store_root: str = DEFAULT_STORE
    verify_store_root: str = VERIFY_STORE
    tracking: str | None = None
    tracking_overrides: list[str] = field(default_factory=list)
    benchmark: bool = False
    benchmark_windows: int = 2
    benchmark_trials: int = 2
    verify_windows: int = 2
    force: bool = False
    #: build the matrix as if the store were empty, without forcing reruns (``--adopt``)
    ignore_store: bool = False
    setup: str = "auto"  # auto | always | never
    long_threshold: float = LONG_THRESHOLD_H
    max_requeues: int = MAX_REQUEUES
    signal_lead_s: int = SIGNAL_LEAD_S


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "__", name)


def run_node_name(job: Any) -> str:
    if job.stage == "tune":
        return f"tune:{job.experiment}:{job.model}"
    if job.stage == "cv":
        return f"cv:{job.experiment}:{job.model}:{job.paradigm}"
    return f"{job.stage}:{job.experiment}:{job.model}:{job.paradigm}:s{job.seed}"


def time_keys(stage: str, family: str = "", resource: str = "", paradigm: str = "") -> list[str]:
    group = FAMILY_GROUP.get(family, family)
    keys = []
    if family:
        keys += [f"{stage}/{family}/{resource}/{paradigm}", f"{stage}/{family}/{resource}"]
    if group:
        keys += [
            f"{stage}/{group}/{resource}/{paradigm}",
            f"{stage}/{group}/{resource}",
            f"{stage}/{group}",
        ]
    keys.append(stage)
    seen: list[str] = []
    for key in keys:
        key = key.replace("//", "/").rstrip("/")
        if key and key not in seen:
            seen.append(key)
    return seen


def _tracking_overrides(args: argparse.Namespace) -> list[str]:
    """Hydra overrides for the W&B coordinates, appended to every job line."""
    out: list[str] = []
    if args.wandb_project:
        out.append(f"tracking.project={args.wandb_project}")
    if args.wandb_entity:
        out.append(f"tracking.entity={args.wandb_entity}")
    if args.wandb_tag:
        out.append("tracking.tags=[" + ",".join(args.wandb_tag) + "]")
    return out


def _store_args(root: str) -> list[str]:
    return ["--store-root", root]


def build_dag(
    opts: Options,
    matrices: dict[str, list[Any]],
    *,
    importance_jobs: dict[str, dict[str, tuple[str, ...]]] | None = None,
    verify_groups: dict[str, list[str]] | None = None,
    need_setup: dict[str, bool] | None = None,
    queued: dict[str, str] | None = None,
) -> list[Node]:
    """The DAG as a topologically ordered list of :class:`Node`.

    ``matrices``: experiment -> ``make_jobs.Job`` list (tune/cv/test, already
    filtered for work that is done). ``importance_jobs``: experiment ->
    ``{model: paradigms}``. ``verify_groups``: experiment -> resource classes
    that have at least one model (``cpu``/``gpu``). ``need_setup``: env ->
    whether its setup job must run. ``queued``: node name -> job id of a job
    from an EARLIER submission that is still in the queue (dependencies of a
    later ``--seeds`` submission).
    """
    stages = set(opts.stages)
    need_setup = need_setup or {}
    queued = dict(queued or {})
    nodes: list[Node] = []
    by_name: dict[str, Node] = {}

    def add(node: Node) -> Node:
        if node.name in by_name:
            raise ValueError(f"duplicate DAG node {node.name!r}")
        if node.name in queued:
            # Still pending/running from an earlier submission (2026-09-27: a
            # resubmit after one failure re-emitted 110 live jobs): never submit
            # it twice; dependants wait on the queued job id via ``dep``.
            return node
        nodes.append(node)
        by_name[node.name] = node
        return node

    def dep(name: str) -> list[str]:
        """``name`` when it is in this DAG or still queued from an earlier one."""
        return [name] if name in by_name or name in queued else []

    tracking = [f"tracking={opts.tracking}"] if opts.tracking else []
    tracking += opts.tracking_overrides
    store = _store_args(opts.store_root)
    experiments = [e for e in opts.experiments if e in matrices or e in (importance_jobs or {})]
    uses_ag = any(
        getattr(j, "env", "main") == "autogluon" for jobs in matrices.values() for j in jobs
    ) or "chronos2" in opts.experiments

    # -- setup ---------------------------------------------------------------
    if "setup" in stages and need_setup.get("main"):
        add(Node("setup", "setup", "gpu", "main", ["main"], ["setup"], script=SETUP_SCRIPT))
    if "setup" in stages and need_setup.get("autogluon") and uses_ag:
        add(
            Node(
                "setup_ag", "setup", "gpu", "autogluon", ["autogluon"], ["setup_ag"],
                script=SETUP_SCRIPT, after_ok=dep("setup"),
            )
        )

    def env_ready(env: str) -> list[str]:
        return dep("setup") + (dep("setup_ag") if env == "autogluon" else [])

    # -- featsel ---------------------------------------------------------------
    for exp in experiments:
        if "featsel" in stages and exp in FEATSEL_EXPERIMENTS and matrices.get(exp):
            add(
                Node(
                    f"featsel:{exp}", "featsel", "cpu", "main",
                    [*CLI, "featsel", f"experiment={exp}", *tracking, *store, "-v"],
                    ["featsel"], after_ok=env_ready("main"), meta={"experiment": exp},
                )
            )

    run_flags: list[str] = []
    tune_flags: list[str] = []
    if opts.benchmark:
        # No --allow-default-params: the pilot's cv/test wait for their own 2-trial tune
        # and use its params, i.e. the exact code path of the real run. (Default params
        # pick the legacy default RNN trainer preset, precision="32-true", which the
        # thesis never used and which fails on float64 series.)
        run_flags = ["--max-folds", str(opts.benchmark_windows * RETRAIN_STRIDE)]
        tune_flags = ["--n-trials", str(opts.benchmark_trials)]

    # -- tune / cv / test --------------------------------------------------------
    for exp in experiments:
        jobs = matrices.get(exp, [])
        for stage in ("tune", "cv", "test"):
            for job in (j for j in jobs if j.stage == stage):
                name = run_node_name(job)
                upstream = env_ready(job.env) + dep(f"featsel:{exp}")
                after_ok = list(upstream)
                if stage in ("cv", "test"):
                    after_ok += dep(f"tune:{exp}:{job.model}")
                if stage == "test":  # a composite's calibrators come from its CV (C18)
                    after_ok += dep(f"cv:{exp}:{job.model}:{job.paradigm}")
                flags = tune_flags if stage == "tune" else run_flags
                argv = [*CLI, *shlex.split(job.command), *flags, *store, "-v"]
                add(
                    Node(
                        name, stage, job.resource, job.env, argv,
                        time_keys(stage, job.family, job.resource, job.paradigm or ""),
                        after_ok=_unique(after_ok),
                        requeue=stage == "tune",
                        meta={
                            "experiment": exp, "model": job.model, "paradigm": job.paradigm,
                            "seed": job.seed, "family": job.family,
                        },
                    )
                )

    # -- importance --------------------------------------------------------------
    if "importance" in stages and not opts.benchmark:
        seed = 42  # the tuning seed: the thesis computed importances once (Stream 2)
        for exp in experiments:
            for model, paradigms in (importance_jobs or {}).get(exp, {}).items():
                info = next((j for j in matrices.get(exp, []) if j.model == model), None)
                resource = getattr(info, "resource", "gpu" if exp == "chronos2" else "cpu")
                env = getattr(info, "env", "autogluon" if exp == "chronos2" else "main")
                family = getattr(info, "family", "")
                for paradigm in paradigms:
                    after_ok = env_ready(env) + dep(f"featsel:{exp}") + dep(f"tune:{exp}:{model}")
                    after_ok += dep(f"test:{exp}:{model}:{paradigm}:s{seed}")
                    add(
                        Node(
                            f"importance:{exp}:{model}:{paradigm}", "importance", resource, env,
                            [
                                *CLI, "importance", f"experiment={exp}", f"model={model}",
                                f"paradigm={paradigm}", f"seed={seed}", *tracking, *store, "-v",
                            ],
                            time_keys("importance", family, resource, paradigm),
                            after_ok=_unique(after_ok),
                            meta={"experiment": exp, "model": model, "paradigm": paradigm},
                        )
                    )

    # -- report / figures ----------------------------------------------------------
    reports: list[str] = []
    if "report" in stages and not opts.benchmark:
        seeds = _report_seeds(opts.seeds)
        for exp in experiments:
            members = [n.name for n in nodes if n.meta.get("experiment") == exp]
            members += [q for q in queued if q.split(":")[1:2] == [exp] and q not in members]
            add(
                Node(
                    f"report:{exp}", "report", "cpu_small", "main",
                    [*CLI, "report", f"experiment={exp}", f"seed={seeds}", *tracking, *store, "-v"],
                    ["report"], after_ok=env_ready("main"), after_any=_unique(members),
                    meta={"experiment": exp},
                )
            )
            reports.append(f"report:{exp}")
    if "figures" in stages and not opts.benchmark and reports:
        add(
            Node(
                "figures", "figures", "cpu_small", "main",
                [*CLI, "figures", *store, "-v"], ["figures"],
                after_ok=env_ready("main"), after_any=reports, optional=True,
            )
        )

    # -- legacy verification (own store, independent of the publication chain) ----
    if "verify" in stages and not opts.benchmark and verify_groups:
        vstore = _store_args(opts.verify_store_root)
        windows = ["--windows", str(opts.verify_windows)]
        verify = [*("scripts/verify_legacy.py",)]
        add(
            Node(
                "verify_prep", "verify", "cpu", "main",
                [*verify, "prep", *vstore], ["verify_prep"], after_ok=env_ready("main"),
            )
        )
        members = []
        for exp, resources in verify_groups.items():
            for resource in resources:
                env = "autogluon" if exp == "chronos2" else "main"
                name = f"verify:{exp}:{resource}"
                add(
                    Node(
                        name, "verify", resource, env,
                        [*verify, "run", f"--experiment={exp}", f"--resource={resource}",
                         *windows, *vstore],
                        ["verify"], after_ok=env_ready(env) + ["verify_prep"],
                        meta={"experiment": f"verify/{exp}"},
                    )
                )
                members.append(name)
        add(
            Node(
                "verify_report", "verify", "cpu_small", "main",
                [*verify, "report", *vstore], ["verify_report"],
                after_ok=env_ready("main"), after_any=members,
            )
        )
    return nodes


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def leaderboard_runs(path: Path, top: int) -> set[tuple[str, str, str]]:
    """The first ``top`` rows of ``strikecast figures``' ``master_leaderboard.csv``
    (seed 42, SkillScore order) as run-store ``(experiment, model, paradigm)`` keys.

    The leaderboard uses the thesis' display families and paradigms; this undoes
    them (``gbdt``/``lstm`` -> count, ``finalhurdle`` -> hurdle, Chronos-2 is shown
    as ``local`` but runs under ``global``), as ``reporting.sources.StoreSource`` does.
    """
    if not path.is_file():
        raise SystemExit(f"--top needs {path}; run the figures stage first")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:top]
    runs = set()
    for row in rows:
        family, paradigm, model = row["Modelname"], row["paradigm"], row["model"]
        experiment = {"gbdt": "count", "lstm": "count", "finalhurdle": "hurdle"}.get(family, family)
        if experiment == "chronos2":
            paradigm = "global"
        runs.add((experiment, model, paradigm))
    return runs


def parse_runs(spec: str) -> set[tuple[str, str, str]]:
    """``count:xgboost_tweedie:global,hurdle:hurdle:global`` -> run keys."""
    runs = set()
    for item in (s for s in spec.split(",") if s):
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--runs entry {item!r} is not experiment:model:paradigm")
        runs.add((parts[0], parts[1], parts[2]))
    return runs


def _report_seeds(seeds: list[int]) -> str:
    """42 first (the leaderboard seed of §7), then the others of this submission."""
    ordered = [42] + [s for s in seeds if s != 42]
    return ",".join(str(s) for s in ordered)


# --------------------------------------------------------------------------- #
# resources and wall time
# --------------------------------------------------------------------------- #
def load_time_table(path: Path | None) -> dict[str, dict[str, Any]]:
    table = {k: dict(v) for k, v in TIME_TABLE.items()}
    if path is not None and path.is_file():
        override = json.loads(path.read_text(encoding="utf-8"))
        for key, value in override.items():
            if key.startswith("_"):
                continue
            entry = value if isinstance(value, dict) else {"hours": float(value)}
            table[key] = {**table.get(key, {}), **entry}
    return table


def format_time(hours: float) -> str:
    minutes = max(1, int(math.ceil(hours * 60)))
    days, rest = divmod(minutes, 24 * 60)
    h, m = divmod(rest, 60)
    return f"{days}-{h:02d}:{m:02d}:00" if days else f"{h:02d}:{m:02d}:00"


def plan_resources(nodes: list[Node], table: dict[str, dict[str, Any]], opts: Options) -> None:
    """Resolve hours, partition and ``--time`` per node (in place).

    Resumable (requeue) jobs stay on ``GPU`` with at most the partition limit;
    any other job whose hours exceed ``opts.long_threshold`` goes to
    ``GPUExtended``. The benchmark pilot asks for a fraction of the time.
    """
    for node in nodes:
        entry = next((table[k] for k in node.time_keys if k in table), {"hours": 24})
        hours = float(entry.get("hours", 24))
        evidence = str(entry.get("evidence", ""))
        if opts.benchmark and node.stage in ("tune", "cv", "test"):
            hours = max(1.0, min(hours, 6.0))
            evidence = f"benchmark pilot cap; {evidence}"
        if node.requeue:
            node.partition = PARTITION
            hours = min(hours, PARTITION_LIMIT_H)
        elif hours > opts.long_threshold:
            node.partition = LONG_PARTITION
        else:
            node.partition = PARTITION
            hours = min(hours, PARTITION_LIMIT_H)
        node.hours = hours
        node.time = format_time(hours)
        node.evidence = evidence


def sbatch_args(node: Node, ids: dict[str, str], opts: Options) -> list[str]:
    """The full ``sbatch`` command line of one node."""
    res = RESOURCES[node.resource]
    node.log = f"{LOG_ROOT}/{node.stage}/{sanitize(node.name)}-%j.out"
    args = [
        "sbatch", "--parsable",
        f"--job-name=sc:{node.name}",
        f"--output={node.log}",
        f"--partition={node.partition}",
        f"--time={node.time}",
        f"--cpus-per-task={res['cpus']}",
        f"--mem={res['mem']}",
    ]
    if res["gres"]:
        args += [f"--gres={res['gres']}", "--gres-flags=disable-binding"]
    if node.requeue:
        args += ["--requeue", "--open-mode=append", f"--signal=B:USR1@{opts.signal_lead_s}"]
    deps = []
    ok = [ids.get(n, f"<{n}>") for n in node.after_ok]
    anyof = [ids.get(n, f"<{n}>") for n in node.after_any]
    if ok:
        deps.append("afterok:" + ":".join(ok))
    if anyof:
        deps.append("afterany:" + ":".join(anyof))
    if deps:
        args.append("--dependency=" + ",".join(deps))
        if ok:
            args.append("--kill-on-invalid-dep=yes")
    export = [f"STRIKECAST_MAX_REQUEUES={opts.max_requeues}"]
    if node.optional:
        export.append("STRIKECAST_OPTIONAL=1")
    args.append("--export=ALL," + ",".join(export))
    if node.script == JOB_SCRIPT:
        args += [node.script, node.env, *node.argv]
    else:
        args += [node.script, *node.argv]
    return args


# --------------------------------------------------------------------------- #
# git, setup stamps, queue
# --------------------------------------------------------------------------- #
IGNORED_DIRTY = ("logs/", "runs_", "runs/", "wandb/", "configs/cluster/time_table.json")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def git_state() -> dict[str, Any]:
    status = [
        line for line in _git("status", "--porcelain").splitlines()
        if not any(line[3:].startswith(p) for p in IGNORED_DIRTY)
    ]
    commit = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream:
        ahead = _git("rev-list", "--count", f"{upstream}..HEAD")
        unpushed = bool(ahead) and ahead != "0"
    else:
        unpushed = not _git("branch", "-r", "--contains", "HEAD")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "upstream": upstream or None,
        "dirty": bool(status),
        "dirty_files": status[:50],
        "unpushed": unpushed,
    }


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def setup_needed(env: str) -> bool:
    """True when the env's stamp is missing or was built from another lock."""
    venv = REPO / (".venv" if env == "main" else "envs/autogluon/.venv")
    lock = REPO / ("uv.lock" if env == "main" else "envs/autogluon/uv.lock")
    stamp = venv / ".strikecast-setup.json"
    if not stamp.is_file():
        return True
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8")).get("lock_sha256")
    except (OSError, ValueError):
        return True
    return recorded != _sha256(lock)


def squeue_ids() -> dict[str, dict[str, str]] | None:
    """``job id -> {state, reason, elapsed, node}`` of the user's jobs, or None."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%T|%r|%M|%N"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    jobs: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 5:
            jobs[parts[0]] = {
                "state": parts[1], "reason": parts[2], "elapsed": parts[3], "node": parts[4],
            }
    return jobs


def squeue_names() -> dict[str, str] | None:
    """``node name -> job id`` of the user's queued/running ``sc:<node>`` jobs, or None."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    found: dict[str, str] = {}
    for line in out.splitlines():
        job_id, _, name = line.partition("|")
        if name.startswith("sc:"):
            found[name[3:]] = job_id
    return found


def adopt(nodes: list[Node], opts: Options, live: dict[str, str]) -> list[Node]:
    """Attach already-submitted job ids to ``nodes`` (after an interrupted submit).

    A node's id is its live ``sc:<name>`` job, else the newest log
    ``logs/slurm/<stage>/<name>-<id>.out`` (a job that already ran). Nodes with
    neither were never submitted and are left out, and listed.
    """
    ids: dict[str, str] = {}
    kept: list[Node] = []
    for node in nodes:
        job_id = live.get(node.name)
        if job_id is None:
            logs = (REPO / LOG_ROOT / node.stage).glob(f"{sanitize(node.name)}-*.out")
            numbers = [p.stem.rsplit("-", 1)[1] for p in logs]
            numbers = [n for n in numbers if n.isdigit()]
            job_id = max(numbers, key=int) if numbers else None
        if job_id is None:
            print(f"  not submitted, left out: {node.name}")
            continue
        node.sbatch = sbatch_args(node, ids, opts)
        node.job_id = job_id
        ids[node.name] = job_id
        kept.append(node)
    return kept


def manifests(directory: Path | None = None) -> list[Path]:
    directory = directory or REPO / MANIFEST_DIR
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def queued_from_earlier(store_root: str) -> dict[str, str]:
    """Node name -> job id for jobs of earlier submissions still in the queue."""
    live = squeue_ids()
    if not live:
        return {}
    found: dict[str, str] = {}
    for path in manifests():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("options", {}).get("store_root") != store_root:
            continue
        for job in payload.get("jobs", []):
            if job.get("job_id") in live:
                found[job["name"]] = job["job_id"]
    return found


# --------------------------------------------------------------------------- #
# the matrix (needs strikecast)
# --------------------------------------------------------------------------- #
def _load_make_jobs():
    spec = importlib.util.spec_from_file_location("make_jobs", SLURM_DIR / "make_jobs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("make_jobs", module)
    spec.loader.exec_module(module)
    return module


def build_matrices(opts: Options) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """experiment -> make_jobs.Job list, and per-experiment info for the manifest."""
    from strikecast.config.loader import load_experiment
    from strikecast.pipeline.context import get_spec
    from strikecast.store import RunStore

    make_jobs = _load_make_jobs()
    root = (REPO / opts.store_root).resolve()
    store = RunStore(root)
    stages = [s for s in ("tune", "cv", "test") if s in opts.stages]
    if "test" in stages and opts.cv == "needed" and "cv" not in stages:
        stages.insert(stages.index("test"), "cv")  # a composite's test needs its CV
    overrides = [f"tracking={opts.tracking}"] if opts.tracking else []
    overrides += opts.tracking_overrides
    matrices: dict[str, list[Any]] = {}
    info: dict[str, Any] = {}
    for exp in opts.experiments:
        cfg = load_experiment(exp, overrides)
        models = [m for m in cfg.model_names if not opts.models or m in opts.models]
        if opts.runs is not None:
            models = [m for m in models if any(r[:2] == (exp, m) for r in opts.runs)]
        jobs = make_jobs.build_jobs(
            cfg,
            models=models,
            paradigms=None,
            stages=stages,
            seeds=opts.seeds,
            overrides=overrides,
            store=None if opts.force or opts.ignore_store else store,
            force=opts.force,
            get_spec=get_spec,
            cv=opts.cv,
        )
        if opts.runs is not None:
            jobs = [j for j in jobs if j.stage == "tune" or (exp, j.model, j.paradigm) in opts.runs]
        # A deterministic model runs once, under the canonical seed (§5.4); a
        # later `--seeds 1,2` submission must not re-run it under seed 1.
        canonical = int(cfg.seeds.eval_seeds[0]) if cfg.seeds.eval_seeds else 42
        kept = []
        for job in jobs:
            spec = make_jobs.lookup_spec(get_spec, job.model, exp)
            deterministic = not getattr(spec, "stochastic", True)
            if job.stage == "test" and deterministic and job.seed != canonical:
                continue
            kept.append(job)
        matrices[exp] = kept
        info[exp] = {"config_sha256": cfg.resolved_hash(), "models": models}
    return matrices, info


def importance_table(experiments: list[str]) -> dict[str, dict[str, tuple[str, ...]]]:
    try:
        from strikecast.pipeline.importance_stage import default_jobs
    except ImportError:  # the importance stage (Stream 2) is not in this checkout
        return {}
    return {exp: default_jobs(exp) for exp in experiments if default_jobs(exp)}


def verification_pending(
    groups: dict[str, list[str]], store_root: Path
) -> dict[str, list[str]]:
    """Drop (experiment, resource) groups whose every case already has a result.

    A case is finished once ``_verification/results/<case>.json`` exists with any
    status but ERROR (a FAIL is a finished comparison; re-running it changes
    nothing). Without this a later resubmit re-queued finished verification jobs.
    """
    from strikecast.verification import legacy  # noqa: PLC0415

    results = legacy.results_dir(store_root)
    pending: dict[str, list[str]] = {}
    for exp, resources in groups.items():
        for resource in resources:
            cases = legacy.cases_for(exp, resource=resource)
            done = True
            for case in cases:
                path = results / f"{case.id}.json"
                try:
                    status = json.loads(path.read_text(encoding="utf-8")).get("status")
                except (OSError, ValueError):
                    status = None
                if status in (None, "ERROR"):
                    done = False
                    break
            if not done:
                pending.setdefault(exp, []).append(resource)
    return pending


def verify_table(matrices: dict[str, list[Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for exp, jobs in matrices.items():
        resources = sorted({j.resource for j in jobs if j.stage == "test"})
        if resources:
            groups[exp] = resources
    return groups


def _ensure_strikecast(argv: list[str]) -> None:
    """Re-exec under .venv/bin/python when this interpreter lacks strikecast."""
    try:
        import strikecast  # noqa: F401
        import strikecast.config.loader  # noqa: F401
        return
    except ImportError:
        pass
    venv_python = REPO / ".venv" / "bin" / "python"
    if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *argv], env)
    raise SystemExit(
        "submit_all.py: `strikecast` is not importable and .venv does not exist yet.\n"
        "Build the environments first (SLURM jobs, nothing runs on the login node):\n"
        "    python3 scripts/slurm/submit_all.py --setup-only\n"
        "then rerun this command once `status` shows the setup job(s) done."
    )


# --------------------------------------------------------------------------- #
# printing and submitting
# --------------------------------------------------------------------------- #
def summarize(nodes: list[Node]) -> list[str]:
    counts: dict[tuple[str, str, str], int] = {}
    for node in nodes:
        key = (node.stage, node.resource, node.partition)
        counts[key] = counts.get(key, 0) + 1
    lines = [f"{'stage':<12} {'class':<10} {'partition':<12} {'jobs':>5}"]
    for (stage, resource, partition), n in sorted(
        counts.items(), key=lambda kv: (ALL_STAGES.index(kv[0][0]) if kv[0][0] in ALL_STAGES else 99, kv[0])
    ):
        lines.append(f"{stage:<12} {resource:<10} {partition:<12} {n:>5}")
    lines.append(f"{'total':<36} {len(nodes):>5}")
    return lines


def submit(nodes: list[Node], opts: Options, *, dry_run: bool, queued: dict[str, str]) -> dict[str, str]:
    ids: dict[str, str] = dict(queued)
    if not dry_run:  # C27: SLURM opens --output before the job runs
        for node in nodes:
            (REPO / LOG_ROOT / node.stage).mkdir(parents=True, exist_ok=True)
    for node in nodes:
        node.sbatch = sbatch_args(node, ids, opts)
        if dry_run:
            print(shlex.join(node.sbatch))
            ids[node.name] = f"<{node.name}>"
            continue
        result = subprocess.run(node.sbatch, cwd=REPO, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                f"sbatch failed for {node.name}: {result.stderr.strip()}\n"
                f"  {shlex.join(node.sbatch)}\n"
                f"{len([n for n in nodes if n.job_id])} job(s) were already submitted; "
                "see the partial manifest and `scancel` them if needed."
            )
        node.job_id = result.stdout.strip().split(";")[0]
        ids[node.name] = node.job_id
        print(f"{node.job_id:>10}  {node.name}  [{node.partition} {node.time} {node.resource}]")
    return ids


def write_manifest(
    nodes: list[Node], opts: Options, git: dict[str, Any], info: dict[str, Any], argv: list[str]
) -> Path:
    # timezone.utc, not datetime.UTC: this file runs on the login node's system python3.
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
    directory = REPO / MANIFEST_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp}{'-benchmark' if opts.benchmark else ''}.json"
    payload = {
        "created": stamp,
        "argv": argv,
        "git": git,
        "options": asdict(opts),
        "experiments": info,
        "jobs": [
            {
                "name": n.name, "stage": n.stage, "job_id": n.job_id, "resource": n.resource,
                "env": n.env, "partition": n.partition, "time": n.time, "hours": n.hours,
                "evidence": n.evidence, "log": n.log, "after_ok": n.after_ok,
                "after_any": n.after_any, "requeue": n.requeue, "meta": n.meta,
                "sbatch": n.sbatch,
            }
            for n in nodes
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


# --------------------------------------------------------------------------- #
# status (no sacct on this cluster)
# --------------------------------------------------------------------------- #
EXIT_RE = re.compile(r"^STRIKECAST_EXIT (\S+)\s*$")


def _log_path(job: dict[str, Any]) -> Path | None:
    if not job.get("log") or not job.get("job_id"):
        return None
    return REPO / job["log"].replace("%j", str(job["job_id"]))


def _tail(path: Path | None, n: int) -> list[str]:
    if path is None or not path.is_file():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 64 * 1024))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    return lines[-n:]


def _store_state(store_root: Path, job: dict[str, Any]) -> tuple[str | None, str | None]:
    """``(status, error)`` from the run store for a run job, else ``(None, None)``."""
    meta = job.get("meta") or {}
    if job.get("stage") not in ("cv", "test") or not meta.get("model"):
        return None, None
    seed = meta.get("seed")
    if job["stage"] == "cv" or seed is None:
        seed = 42
    path = (
        store_root / meta["experiment"] / meta["model"] / str(meta.get("paradigm"))
        / f"seed={seed}" / "state.json"
    )
    try:
        state = json.loads(path.read_text(encoding="utf-8")).get(job["stage"])
    except (OSError, ValueError):
        return None, None
    if not state:
        return None, None
    return state.get("status"), state.get("error")


def job_status(
    job: dict[str, Any], live: dict[str, dict[str, str]] | None, store_root: Path
) -> tuple[str, str]:
    """``(state, detail)`` for one manifest job.

    In the queue: squeue's state (``pending`` + reason, e.g. ``Dependency``).
    Gone from the queue: the ``STRIKECAST_EXIT`` line of its log (0 -> done,
    99 -> interrupted, other -> failed; none -> killed: wall clock, OOM or node
    failure), refined by the stage's ``state.json`` for run jobs.
    """
    job_id = job.get("job_id")
    if not job_id:
        return "not-submitted", ""
    if live is not None and job_id in live:
        entry = live[job_id]
        state = entry["state"].lower()
        detail = entry["reason"] if state == "pending" else f"{entry['elapsed']} on {entry['node']}"
        return state, detail
    exit_code = None
    for line in reversed(_tail(_log_path(job), 50)):
        match = EXIT_RE.match(line)
        if match:
            exit_code = match.group(1)
            break
    status, error = _store_state(store_root, job)
    if exit_code == "0":
        if status in (None, "complete") or job.get("stage") not in ("cv", "test"):
            return "done", ""
        return "failed", f"exit 0 but state.json says {status}"
    if exit_code == "requeued":
        return "requeued", "waiting to be re-queued"
    if exit_code == "99":
        return "interrupted", error or "signal"
    if exit_code is not None:
        return "failed", error or f"exit {exit_code}"
    if status == "complete":
        return "done", "(no exit line)"
    if live is None:
        return "unknown", "squeue unavailable"
    return "failed", error or "no exit line: killed (wall clock / OOM / node) or not started"


def _manifest_store(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("options", {}).get("store_root")
    except (OSError, ValueError):
        return None


def merged_rows(
    paths: list[Path], live: dict[str, dict[str, str]] | None, store_root: Path
) -> list[tuple[dict[str, Any], str, str]]:
    """One row per job name across manifests (oldest first): the copy that is in
    the queue wins, then one that finished, then the newest. A job cancelled as a
    duplicate therefore never hides the original that is still running or done."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for p in paths:
        try:
            jobs = json.loads(p.read_text(encoding="utf-8")).get("jobs", [])
        except (OSError, ValueError):
            continue
        for job in jobs:
            candidates.setdefault(job["name"], []).append(job)
    rows = []
    for jobs in candidates.values():
        judged = [(job, *job_status(job, live, store_root)) for job in jobs]
        in_queue = [r for r in judged if live is not None and r[0].get("job_id") in live]
        done = [r for r in judged if r[1] == "done"]
        rows.append((in_queue or done or judged)[-1])
    return rows


def cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="submit_all.py status")
    parser.add_argument("--manifest", type=Path, default=None, help="default: the latest one")
    parser.add_argument("--tail", type=int, default=15, help="log lines per failed job")
    parser.add_argument("--all", action="store_true", help="list every job, not only unfinished")
    args = parser.parse_args(argv)

    found = [args.manifest] if args.manifest else manifests()
    if not found:
        print(f"no submission manifest under {MANIFEST_DIR}/", file=sys.stderr)
        return 1
    path = found[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    store_name = payload.get("options", {}).get("store_root", DEFAULT_STORE)
    store_root = (REPO / store_name).resolve()
    live = squeue_ids()
    # Every submission to the same store, oldest first (a resubmit after a failure
    # adds only the missing jobs, so one manifest alone is not the whole picture).
    siblings = [path] if args.manifest else [
        p for p in found
        if _manifest_store(p) == store_name
    ]
    print(f"manifest {path.relative_to(REPO) if path.is_relative_to(REPO) else path}"
          f"{f' (+{len(siblings) - 1} earlier for this store)' if len(siblings) > 1 else ''}  "
          f"commit {str(payload.get('git', {}).get('commit'))[:10]}  store {store_root}")
    if live is None:
        print("(squeue unavailable: states come from the logs and the run store only)")

    rows = merged_rows(siblings, live, store_root)

    counts: dict[str, dict[str, int]] = {}
    for job, state, _ in rows:
        counts.setdefault(job["stage"], {}).setdefault(state, 0)
        counts[job["stage"]][state] += 1
    states = sorted({s for c in counts.values() for s in c})
    print(f"\n{'stage':<12}" + "".join(f"{s:>14}" for s in states))
    for stage in sorted(counts, key=lambda s: ALL_STAGES.index(s) if s in ALL_STAGES else 99):
        print(f"{stage:<12}" + "".join(f"{counts[stage].get(s, 0):>14}" for s in states))

    print()
    for job, state, detail in rows:
        if args.all or state not in ("done",):
            print(f"{state:<12} {str(job.get('job_id')):>10}  {job['name']}  {detail}")
    failed = [(job, state) for job, state, _ in rows if state in ("failed", "interrupted")]
    for job, state in failed:
        log = _log_path(job)
        print(f"\n--- {state}: {job['name']} ({log}) ---")
        for line in _tail(log, args.tail):
            print(f"    {line}")
    return 0


# --------------------------------------------------------------------------- #
# timings: the pilot's seconds per fold / trial -> a time table
# --------------------------------------------------------------------------- #
def _trial_seconds(trials_csv: Path) -> list[float]:
    out: list[float] = []
    try:
        with trials_csv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("state", "COMPLETE") not in ("COMPLETE", "TrialState.COMPLETE"):
                    continue
                duration = row.get("duration")
                if duration:
                    out.append(_parse_duration(duration))
                    continue
                start, end = row.get("datetime_start"), row.get("datetime_complete")
                if start and end:
                    out.append(
                        (_dt.datetime.fromisoformat(end) - _dt.datetime.fromisoformat(start))
                        .total_seconds()
                    )
    except (OSError, ValueError):
        return []
    return [s for s in out if s > 0]


def _parse_duration(text: str) -> float:
    """``0 days 00:02:24.123`` / ``00:02:24`` / seconds -> seconds."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    days = 0
    match = re.match(r"(\d+) days? (.*)", text)
    if match:
        days, text = int(match.group(1)), match.group(2)
    h, m, s = text.split(":")
    return days * 86400 + int(h) * 3600 + int(m) * 60 + float(s)


def collect_timings(store_root: Path, manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per pilot job with measured seconds per fold (run) or trial (tune)."""
    rows: list[dict[str, Any]] = []
    jobs = (manifest or {}).get("jobs", [])
    by_run = {
        (j["meta"].get("experiment"), j["meta"].get("model"), j["meta"].get("paradigm"), j["stage"]): j
        for j in jobs if j.get("meta")
    }
    for state_path in sorted(store_root.glob("*/*/*/seed=*/state.json")):
        exp, model, paradigm = state_path.parts[-5], state_path.parts[-4], state_path.parts[-3]
        try:
            states = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for stage, state in states.items():
            if stage not in FULL_FOLDS:
                continue
            attempts = [a for a in state.get("attempts", []) if a.get("outcome") == "complete"]
            if not attempts or not state.get("folds_done"):
                continue
            seconds = float(attempts[-1].get("seconds") or 0)
            folds = int(state.get("folds_done") or attempts[-1].get("folds_done") or 0)
            if seconds <= 0 or folds <= 0:
                continue
            job = by_run.get((exp, model, paradigm, stage), {})
            rows.append({
                "stage": stage, "experiment": exp, "model": model, "paradigm": paradigm,
                "family": job.get("meta", {}).get("family", ""), "resource": job.get("resource", ""),
                "seconds": seconds, "units": folds, "per_unit": seconds / folds,
                "full_units": FULL_FOLDS[stage],
            })
    for trials in sorted(store_root.glob("*/tuning/*/trials.csv")):
        exp, model = trials.parts[-4], trials.parts[-2]
        durations = _trial_seconds(trials)
        if not durations:
            continue
        job = next((j for j in jobs if j.get("name") == f"tune:{exp}:{model}"), {})
        n_trials = 12 if exp == "chronos2" else 50
        rows.append({
            "stage": "tune", "experiment": exp, "model": model, "paradigm": "",
            "family": job.get("meta", {}).get("family", ""), "resource": job.get("resource", ""),
            "seconds": sum(durations), "units": len(durations),
            "per_unit": max(durations), "full_units": n_trials,
        })
    return rows


def timings_table(rows: list[dict[str, Any]], safety: float = 1.5) -> dict[str, dict[str, Any]]:
    """Rows -> ``{time key: {hours, evidence}}`` at the most specific key (max over models)."""
    table: dict[str, dict[str, Any]] = {}
    today = _dt.date.today().isoformat()
    for row in rows:
        key = time_keys(row["stage"], row["family"], row["resource"], row["paradigm"])[0]
        hours = row["per_unit"] * row["full_units"] * safety / 3600.0
        hours = max(0.5, math.ceil(hours * 4) / 4)
        unit = "trial (max)" if row["stage"] == "tune" else "fold"
        evidence = (
            f"pilot {today}: {row['experiment']}/{row['model']} {row['per_unit']:.1f} s/{unit} "
            f"x {row['full_units']} x {safety}"
        )
        if key not in table or hours > table[key]["hours"]:
            table[key] = {"hours": hours, "evidence": evidence}
    return table


def cmd_timings(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="submit_all.py timings")
    parser.add_argument("--store-root", default=BENCHMARK_STORE)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="the pilot's manifest (default: the latest *-benchmark.json)")
    parser.add_argument("--out", type=Path, default=REPO / TIME_TABLE_FILE)
    parser.add_argument("--safety", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    manifest_path = args.manifest or next(
        (p for p in reversed(manifests()) if p.name.endswith("-benchmark.json")), None
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None
    rows = collect_timings((REPO / args.store_root).resolve(), manifest)
    if not rows:
        print(f"no completed pilot stages or trials under {args.store_root}", file=sys.stderr)
        return 1
    print(f"{'stage':<6} {'experiment/model/paradigm':<48} {'s/unit':>9} {'units':>6} {'full h':>7}")
    for row in sorted(rows, key=lambda r: (r["stage"], r["experiment"], r["model"], r["paradigm"])):
        full = row["per_unit"] * row["full_units"] / 3600
        label = f"{row['experiment']}/{row['model']}/{row['paradigm']}".rstrip("/")
        print(f"{row['stage']:<6} {label:<48} {row['per_unit']:>9.1f} {row['units']:>6} {full:>7.2f}")
    table = timings_table(rows, args.safety)
    payload = {"_comment": f"generated by `submit_all.py timings` from {args.store_root}", **table}
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(table)} entries to {args.out}; submit_all.py reads it automatically "
          "(commit it: it is the evidence for the --time requests)")
    return 0


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="submit_all.py",
        description="Submit the thesis re-run DAG to SLURM (subcommands: status, timings).",
    )
    p.add_argument("--seeds", default="42", help="evaluation seeds of the test stage (default 42)")
    p.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help=f"subset of {','.join(ALL_STAGES)}")
    p.add_argument("--models", default=None, help="only these registry names")
    p.add_argument("--top", type=int, default=None,
                   help="only the top-N runs of <store>/_figures/master_leaderboard.csv "
                        "(e.g. a seed sweep of the top 20)")
    p.add_argument("--runs", default=None,
                   help="only these runs, experiment:model:paradigm[,...]; added to --top")
    p.add_argument("--cv", choices=("needed", "all"), default="needed",
                   help="cv stage only where a later stage needs it (hurdle) or for every model")
    p.add_argument("--store-root", default=None,
                   help=f"default {DEFAULT_STORE} ({BENCHMARK_STORE} with --benchmark)")
    p.add_argument("--verify-store-root", default=VERIFY_STORE)
    p.add_argument("--tracking", default=None,
                   help="tracking group override (default: the config's wandb_online, "
                        "strict: false; noop for --benchmark)")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project for every job (tracking.project=...; default: the config's)")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (user or team) for every job (tracking.entity=...)")
    p.add_argument("--wandb-tag", action="append", default=[],
                   help="extra W&B tag for every job; repeatable (tracking.tags=[...])")
    p.add_argument("--time-table", type=Path, default=None,
                   help=f"JSON time overrides (default {TIME_TABLE_FILE} when present)")
    p.add_argument("--long-threshold", type=float, default=LONG_THRESHOLD_H,
                   help="hours above which a non-resumable job goes to GPUExtended")
    p.add_argument("--max-requeues", type=int, default=MAX_REQUEUES)
    p.add_argument("--setup-only", action="store_true", help="submit only the environment build")
    p.add_argument("--setup", choices=("auto", "always", "never"), default="auto",
                   help="auto: submit setup when a venv stamp is missing or stale")
    p.add_argument("--no-verify", action="store_true", help="leave the legacy verification out")
    p.add_argument("--verify-only", "--verify", dest="verify_only", action="store_true",
                   help="only the legacy verification (it is also part of the default DAG)")
    p.add_argument("--verify-windows", type=int, default=2)
    p.add_argument("--benchmark", action="store_true",
                   help="pilot: 2 retrain windows per run, 2 trials per study, own store")
    p.add_argument("--benchmark-windows", type=int, default=2)
    p.add_argument("--benchmark-trials", type=int, default=2)
    p.add_argument("--force", action="store_true", help="re-emit complete stages / tuned models")
    p.add_argument("--dry-run", action="store_true", help="print the DAG and sbatch lines only")
    p.add_argument("--adopt", action="store_true",
                   help="write the manifest for jobs an interrupted submit already queued "
                        "(same arguments as that submit); submits nothing")
    p.add_argument("--allow-dirty", action="store_true",
                   help="submit from a dirty or unpushed checkout")
    return p


def options_from(args: argparse.Namespace) -> Options:
    stages = [s for s in args.stages.split(",") if s]
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {list(ALL_STAGES)}")
    if args.setup_only:
        stages = ["setup"]
    elif args.verify_only:
        stages = ["setup", "verify"]
    elif args.no_verify:
        stages = [s for s in stages if s != "verify"]
    if args.benchmark:
        stages = [s for s in stages if s in ("setup", "featsel", "tune", "cv", "test")]
    store_root = args.store_root or (BENCHMARK_STORE if args.benchmark else DEFAULT_STORE)
    runs = None
    if args.top is not None or args.runs:
        runs = parse_runs(args.runs or "")
        if args.top is not None:
            runs |= leaderboard_runs(REPO / store_root / "_figures" / "master_leaderboard.csv",
                                     args.top)
    experiments = [e for e in args.experiments.split(",") if e]
    if runs is not None:
        experiments = [e for e in experiments if any(r[0] == e for r in runs)]
    return Options(
        seeds=[int(s) for s in args.seeds.split(",") if s],
        experiments=experiments,
        stages=stages,
        models=[m for m in args.models.split(",") if m] if args.models else None,
        runs=runs,
        cv=args.cv,
        store_root=store_root,
        verify_store_root=args.verify_store_root,
        tracking=args.tracking or ("noop" if args.benchmark else None),
        tracking_overrides=_tracking_overrides(args),
        benchmark=args.benchmark,
        benchmark_windows=args.benchmark_windows,
        benchmark_trials=args.benchmark_trials,
        verify_windows=args.verify_windows,
        force=args.force,
        setup="always" if args.setup_only else args.setup,
        long_threshold=args.long_threshold,
        max_requeues=args.max_requeues,
    )


def cmd_submit(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    opts = options_from(args)
    if args.adopt:  # rebuild the full DAG of that submit, finished jobs included
        opts.ignore_store = True

    git = git_state()
    if (git["dirty"] or git["unpushed"]) and not args.allow_dirty:
        what = "dirty" if git["dirty"] else "not pushed"
        message = (
            f"the checkout is {what} (commit {git['commit'][:10]}); the manifest could not name "
            "the code that ran. Commit and push, or pass --allow-dirty."
        )
        if git["dirty"]:
            message += "\n  " + "\n  ".join(git["dirty_files"][:10])
        if not args.dry_run:
            raise SystemExit(f"submit_all.py: refusing to submit: {message}")
        print(f"WARNING (dry run): {message}\n")

    need_setup = {
        "main": opts.setup == "always" or (opts.setup == "auto" and setup_needed("main")),
        "autogluon": opts.setup == "always" or (opts.setup == "auto" and setup_needed("autogluon")),
    }
    matrices: dict[str, list[Any]] = {}
    info: dict[str, Any] = {}
    importance: dict[str, dict[str, tuple[str, ...]]] = {}
    verify: dict[str, list[str]] = {}
    needs_matrix = any(s in opts.stages for s in ("featsel", "tune", "cv", "test", "importance",
                                                     "report", "figures", "verify"))
    if needs_matrix:
        _ensure_strikecast(argv)
        matrices, info = build_matrices(opts)
        if "importance" in opts.stages:
            importance = importance_table(opts.experiments)
        if "verify" in opts.stages and not opts.benchmark:
            verify = verify_table(build_matrices(
                Options(**{**asdict(opts), "force": True, "stages": ["test"], "seeds": [42]})
            )[0])
            if not opts.force:
                verify = verification_pending(verify, REPO / opts.verify_store_root)

    # read-only (squeue), so the dry run shows exactly what a real submit would add
    queued = queued_from_earlier(opts.store_root)
    nodes = build_dag(
        opts, matrices, importance_jobs=importance, verify_groups=verify,
        need_setup=need_setup, queued=queued,
    )
    table = load_time_table(args.time_table or REPO / TIME_TABLE_FILE)
    plan_resources(nodes, table, opts)
    if not nodes:
        print("nothing to submit: every requested stage is complete (use --force to re-emit)")
        return 0

    print(f"commit {git['commit'][:10]} ({git['branch']}), store {opts.store_root}, "
          f"seeds {opts.seeds}, experiments {opts.experiments}"
          f"{' [BENCHMARK]' if opts.benchmark else ''}\n")
    print("\n".join(summarize(nodes)))
    print()
    if args.adopt:
        live = squeue_names()
        if live is None:
            raise SystemExit("--adopt needs squeue (run it on the login node)")
        nodes = adopt(nodes, opts, live)
        if not nodes:
            print("nothing to adopt: no queued job or log matches this DAG")
            return 0
        manifest = write_manifest(nodes, opts, git, info, argv)
        print(f"\nadopted {len(nodes)} job(s); manifest {manifest.relative_to(REPO)}")
        return 0
    try:
        submit(nodes, opts, dry_run=args.dry_run, queued=queued)
    except (KeyboardInterrupt, SystemExit):
        if not args.dry_run and any(n.job_id for n in nodes):
            partial = write_manifest([n for n in nodes if n.job_id], opts, git, info, argv)
            print(f"\ninterrupted: partial manifest {partial.relative_to(REPO)}", file=sys.stderr)
        raise
    if args.dry_run:
        print(f"\n(dry run: {len(nodes)} job(s), nothing submitted)")
        return 0
    manifest = write_manifest(nodes, opts, git, info, argv)
    print(f"\n{len(nodes)} job(s) submitted; manifest {manifest.relative_to(REPO)}")
    print("follow with: python3 scripts/slurm/submit_all.py status")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "status":
        return cmd_status(argv[1:])
    if argv and argv[0] == "timings":
        return cmd_timings(argv[1:])
    if argv and argv[0] == "submit":
        argv = argv[1:]
    return cmd_submit(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
