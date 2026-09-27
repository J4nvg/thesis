#!/usr/bin/env python3
"""Merge one run store into another (plan §5.3, §8 P6: "run-store sync between
cluster scratch and the laptop").

    # look first -- nothing is written
    python scripts/sync_runs.py /scratch/$USER/runs ./runs --dry-run

    # then do it
    python scripts/sync_runs.py /scratch/$USER/runs ./runs

Why this is Python and not `rsync`
----------------------------------
`rsync` copies bytes and decides by timestamp. A run store is not a pile of
bytes: `state.json` says whether a stage is `complete`, and a *complete* stage
must never be replaced by a half-finished one just because the half-finished
one is newer. That is exactly what `rsync -au` from a scratch directory where a
job died would do. This script reads the two `state.json` files, decides per
stage, and only then copies. It also merges `env.json` and `state.json` instead
of letting one side's copy win wholesale, which no file-level tool can do.

Standard library only, so it runs on a login node with bare `python3` and needs
no `uv sync`.

Rules
-----
The unit of decision is one **stage** of one run directory
(`<experiment>/<model>/<paradigm>/seed=<s>/<stage>`), ranked by
:func:`stage_rank` = ``(status == "complete", folds_done)``:

* Source ranks **higher** -> the stage is *accepted*: its files replace the
  destination's, and destination prediction parts the source does not have are
  pruned, so the stage does not end up as a mix of two schedules.
* Otherwise -> *top-up only*: files missing in the destination are copied,
  nothing is overwritten and nothing is deleted. A `complete` destination stage
  therefore survives a scratch directory full of crashed restarts, which is the
  guarantee that matters.

`state.json` is merged stage by stage under the same rank rule. `env.json` is
merged key by key: the side that contributed the newer stages wins a contested
scalar, keys only one side has are kept, and `tracker_run_ids` is the union
rebuilt from the merged states -- the same reconstruction
`RunStore.record_tracker_run_id` does, so the W&B ids of both machines survive.

`shared/` artefacts are content-addressed (`panel.<hash>.parquet`), so they are
copied when missing and never overwritten. `tuning/`, `artifacts/`, `report/`
and `config.yaml` are copied when missing; when both sides have a *different*
file the destination is kept and the difference is reported as a **conflict**
(`--force` takes the source instead). An Optuna SQLite study cannot be merged
by a file copier, and pretending otherwise would lose trials.

Idempotence: running it twice changes nothing the second time. Two files are
"the same" when their sizes and SHA-256 digests match; above `--hash-limit`
(default 64 MiB) equal size is taken as equal, which is why the digest is
skipped there and not why it is trusted -- predictions are written as immutable
`part-<from>-<to>.parquet` files, never appended to.

Exit codes: 0, or 1 with `--strict` when anything was reported as a conflict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

#: Directories under `<root>/<experiment>/` that are not run directories (§5.3).
RESERVED: tuple[str, ...] = ("shared", "tuning", "report")

STATE_FILE = "state.json"
ENV_FILE = "env.json"
CONFIG_FILE = "config.yaml"

#: The run-directory file that is never a stage directory (§5.3).
RUN_ONLY_DIRS: tuple[str, ...] = ("artifacts",)

DEFAULT_HASH_LIMIT = 64 * 1024 * 1024

#: Verbs, in the order the summary prints them.
VERBS: tuple[str, ...] = ("copy", "replace", "merge", "prune", "keep", "conflict")


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Action:
    """One decision, reported before it is (maybe) carried out.

    ``path`` is relative to the destination root, so the report reads the same
    whichever machine it runs on.
    """

    verb: str
    path: str
    reason: str
    src: Path | None = None
    dst: Path | None = None
    payload: str | None = None

    def __str__(self) -> str:
        return f"{self.verb:<8} {self.path}  ({self.reason})"


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)

    def add(self, action: Action) -> None:
        self.actions.append(action)

    def of(self, verb: str) -> list[Action]:
        return [a for a in self.actions if a.verb == verb]

    @property
    def conflicts(self) -> list[Action]:
        return self.of("conflict")

    def counts(self) -> dict[str, int]:
        return {verb: len(self.of(verb)) for verb in VERBS if self.of(verb)}

    def report(self, *, verbose: bool = False) -> str:
        lines = [str(a) for a in self.actions if verbose or a.verb != "keep"]
        counts = self.counts()
        tally = ", ".join(f"{n} {verb}" for verb, n in counts.items()) or "nothing to do"
        lines.append(f"-- {tally}")
        return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any] | None:
    """``None`` when the file is absent or unreadable (a corrupt half-write)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def stage_rank(state: dict[str, Any] | None) -> tuple[int, int]:
    """How much a stage is worth: complete beats incomplete, then folds done.

    A missing state is ``(0, 0)``, which is why a stage the destination has
    never heard of is always taken from the source.
    """
    if not state:
        return (0, 0)
    complete = 1 if state.get("status") == "complete" else 0
    try:
        folds = int(state.get("folds_done") or 0)
    except (TypeError, ValueError):
        folds = 0
    return (complete, folds)


def same_file(a: Path, b: Path, *, hash_limit: int = DEFAULT_HASH_LIMIT) -> bool:
    """Byte-equality, with the digest skipped for files above ``hash_limit``."""
    try:
        sa, sb = a.stat().st_size, b.stat().st_size
    except OSError:
        return False
    if sa != sb:
        return False
    if sa > hash_limit:
        return True
    return _digest(a) == _digest(b)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - defensive
        return str(path)


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _plan_tree(
    plan: Plan,
    src_dir: Path,
    dst_dir: Path,
    dst_root: Path,
    *,
    reason: str,
    overwrite: bool,
    force: bool,
    hash_limit: int,
) -> None:
    """Plan a directory copy.

    ``overwrite`` says what happens when both sides have a *different* file:
    replace it (an accepted stage) or report a conflict and keep the
    destination (everything else, unless ``force``).
    """
    if not src_dir.is_dir():
        return
    for src in _iter_files(src_dir):
        dst = dst_dir / src.relative_to(src_dir)
        rel = _rel(dst_root, dst)
        if not dst.exists():
            plan.add(Action("copy", rel, reason, src, dst))
        elif same_file(src, dst, hash_limit=hash_limit):
            plan.add(Action("keep", rel, "identical", src, dst))
        elif overwrite or force:
            plan.add(Action("replace", rel, reason, src, dst))
        else:
            plan.add(Action("conflict", rel, f"{reason}; differs, keeping destination", src, dst))


def _stage_names(run_dir: Path, states: dict[str, Any]) -> list[str]:
    """Stages of a run: the `state.json` keys plus any stage directory on disk."""
    names = set(states)
    if run_dir.is_dir():
        names |= {
            child.name
            for child in run_dir.iterdir()
            # `.state.lock` (RunStore's NFS-safe lock directory) is never a stage.
            if child.is_dir() and child.name not in RUN_ONLY_DIRS and not child.name.startswith(".")
        }
    return sorted(names)


def plan_run(
    plan: Plan,
    src_run: Path,
    dst_run: Path,
    dst_root: Path,
    *,
    force: bool,
    hash_limit: int,
) -> None:
    """One `<model>/<paradigm>/seed=<s>` directory, stage by stage."""
    src_states = read_json(src_run / STATE_FILE) or {}
    dst_states = read_json(dst_run / STATE_FILE) or {}

    accepted: list[str] = []
    merged_states: dict[str, Any] = dict(dst_states)
    for stage in _stage_names(src_run, src_states):
        src_state = src_states.get(stage)
        dst_state = dst_states.get(stage)
        take = stage_rank(src_state) > stage_rank(dst_state)
        if take:
            accepted.append(stage)
            if src_state is not None:
                merged_states[stage] = src_state
        elif dst_state is None and src_state is not None:
            merged_states[stage] = src_state

        src_stage = src_run / stage
        if not src_stage.is_dir():
            continue
        reason = (
            f"source stage {_fmt_rank(src_state)} > destination {_fmt_rank(dst_state)}"
            if take
            else f"destination stage {_fmt_rank(dst_state)} kept; top-up only"
        )
        _plan_tree(
            plan,
            src_stage,
            dst_run / stage,
            dst_root,
            reason=reason,
            overwrite=take,
            force=force,
            hash_limit=hash_limit,
        )
        if take:
            _plan_prune(plan, src_stage, dst_run / stage, dst_root)

    # run-level files: config.yaml is a conflict when it differs (two different
    # runs would be sharing one run key), artifacts/ is copy-if-missing.
    _plan_tree(
        plan,
        src_run / "artifacts",
        dst_run / "artifacts",
        dst_root,
        reason="run artifacts",
        overwrite=False,
        force=force,
        hash_limit=hash_limit,
    )
    src_cfg, dst_cfg = src_run / CONFIG_FILE, dst_run / CONFIG_FILE
    if src_cfg.is_file():
        rel = _rel(dst_root, dst_cfg)
        if not dst_cfg.exists():
            plan.add(Action("copy", rel, "run config", src_cfg, dst_cfg))
        elif same_file(src_cfg, dst_cfg, hash_limit=hash_limit):
            plan.add(Action("keep", rel, "identical", src_cfg, dst_cfg))
        elif force:
            plan.add(Action("replace", rel, "run config (--force)", src_cfg, dst_cfg))
        else:
            plan.add(
                Action(
                    "conflict",
                    rel,
                    "two configs under one run key; keeping destination",
                    src_cfg,
                    dst_cfg,
                )
            )

    _plan_state_and_env(
        plan,
        src_run,
        dst_run,
        dst_root,
        merged_states=merged_states,
        dst_states=dst_states,
        accepted=accepted,
    )


def _fmt_rank(state: dict[str, Any] | None) -> str:
    if not state:
        return "[absent]"
    return f"[{state.get('status', '?')}, {stage_rank(state)[1]} folds]"


def _plan_prune(plan: Plan, src_stage: Path, dst_stage: Path, dst_root: Path) -> None:
    """Drop destination prediction parts the accepted source stage lacks.

    Without this, accepting a source stage over a longer but incomplete
    destination one would leave parts of two different schedules side by side
    and `load_predictions` would read both.
    """
    dst_preds = dst_stage / "predictions"
    if not dst_preds.is_dir():
        return
    src_preds = src_stage / "predictions"
    keep = {p.name for p in src_preds.glob("*.parquet")} if src_preds.is_dir() else set()
    for part in sorted(dst_preds.glob("*.parquet")):
        if part.name not in keep:
            plan.add(
                Action(
                    "prune",
                    _rel(dst_root, part),
                    "not in the accepted source stage",
                    None,
                    part,
                )
            )


def merge_env(
    src_env: dict[str, Any],
    dst_env: dict[str, Any],
    merged_states: dict[str, Any],
    *,
    source_wins: bool,
) -> dict[str, Any]:
    """Merge two `env.json` payloads.

    ``source_wins`` when the source contributed at least one accepted stage:
    the machine that produced the newer results is the one whose environment
    describes them. Contested scalars go to the winner; keys only one side has
    are kept either way. `tracker_run_ids` is never contested -- it is the
    union of both maps, then overwritten from the merged `state.json`, which is
    where `RunStore.record_tracker_run_id` reconstructs it from too.
    """
    loser, winner = (dst_env, src_env) if source_wins else (src_env, dst_env)
    merged: dict[str, Any] = {**loser, **winner}

    ids: dict[str, Any] = {}
    for env in (loser, winner):
        raw = env.get("tracker_run_ids")
        if isinstance(raw, dict):
            ids.update(raw)
    for stage, state in merged_states.items():
        run_id = (state or {}).get("tracker_run_id")
        if run_id:
            ids[stage] = run_id
    if ids:
        merged["tracker_run_ids"] = ids
        latest = max(
            (s for s in merged_states.values() if (s or {}).get("tracker_run_id")),
            key=lambda s: (s.get("finished") or s.get("started") or ""),
            default=None,
        )
        if latest is not None:
            merged["tracker_run_id"] = latest["tracker_run_id"]
    elif "tracker_run_ids" in merged:  # pragma: no cover - both sides empty maps
        del merged["tracker_run_ids"]
    return merged


def _plan_state_and_env(
    plan: Plan,
    src_run: Path,
    dst_run: Path,
    dst_root: Path,
    *,
    merged_states: dict[str, Any],
    dst_states: dict[str, Any],
    accepted: Sequence[str],
) -> None:
    src_state_file = src_run / STATE_FILE
    if merged_states and (merged_states != dst_states or not (dst_run / STATE_FILE).exists()):
        plan.add(
            Action(
                "merge",
                _rel(dst_root, dst_run / STATE_FILE),
                f"stages from source: {', '.join(accepted) or 'none'}",
                src_state_file,
                dst_run / STATE_FILE,
                payload=json.dumps(merged_states, indent=2, sort_keys=True) + "\n",
            )
        )
    elif merged_states:
        plan.add(Action("keep", _rel(dst_root, dst_run / STATE_FILE), "identical"))

    src_env = read_json(src_run / ENV_FILE)
    dst_env = read_json(dst_run / ENV_FILE)
    if src_env is None and dst_env is None:
        return
    merged_env = merge_env(
        src_env or {},
        dst_env or {},
        merged_states,
        source_wins=bool(accepted),
    )
    if merged_env != (dst_env or {}) or dst_env is None:
        plan.add(
            Action(
                "merge",
                _rel(dst_root, dst_run / ENV_FILE),
                f"{'source' if accepted else 'destination'} environment wins contested keys",
                src_run / ENV_FILE,
                dst_run / ENV_FILE,
                payload=json.dumps(merged_env, indent=2, sort_keys=True) + "\n",
            )
        )
    else:
        plan.add(Action("keep", _rel(dst_root, dst_run / ENV_FILE), "identical"))


def iter_run_dirs(experiment_dir: Path) -> Iterable[Path]:
    """`<model>/<paradigm>/seed=<s>` directories of one experiment (§5.3)."""
    if not experiment_dir.is_dir():
        return
    for model in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        if model.name in RESERVED:
            continue
        for paradigm in sorted(p for p in model.iterdir() if p.is_dir()):
            for seed in sorted(p for p in paradigm.iterdir() if p.is_dir()):
                if seed.name.startswith("seed="):
                    yield seed


def plan_sync(
    src_root: Path,
    dst_root: Path,
    *,
    experiments: Sequence[str] | None = None,
    force: bool = False,
    hash_limit: int = DEFAULT_HASH_LIMIT,
) -> Plan:
    """Decide everything; write nothing."""
    plan = Plan()
    if not src_root.is_dir():
        raise SystemExit(f"sync_runs: source store {src_root} does not exist")
    wanted = set(experiments or ())
    for exp in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if wanted and exp.name not in wanted:
            continue
        dst_exp = dst_root / exp.name
        # content-addressed: never overwritten, so a partial copy simply
        # completes on the next run.
        _plan_tree(
            plan,
            exp / "shared",
            dst_exp / "shared",
            dst_root,
            reason="shared artefact (content-addressed)",
            overwrite=False,
            force=force,
            hash_limit=hash_limit,
        )
        _plan_tree(
            plan,
            exp / "tuning",
            dst_exp / "tuning",
            dst_root,
            reason="tuning study (not mergeable file-wise)",
            overwrite=False,
            force=force,
            hash_limit=hash_limit,
        )
        _plan_tree(
            plan,
            exp / "report",
            dst_exp / "report",
            dst_root,
            reason="report aggregate (regenerable)",
            overwrite=False,
            force=force,
            hash_limit=hash_limit,
        )
        for src_run in iter_run_dirs(exp):
            dst_run = dst_exp / src_run.relative_to(exp)
            plan_run(plan, src_run, dst_run, dst_root, force=force, hash_limit=hash_limit)
    return plan


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def apply_plan(plan: Plan, *, dry_run: bool = False) -> int:
    """Carry out a plan. Returns the number of actions performed."""
    done = 0
    for action in plan.actions:
        if action.verb in ("keep", "conflict"):
            continue
        if dry_run:
            done += 1
            continue
        if action.verb in ("copy", "replace"):
            assert action.src is not None and action.dst is not None
            _atomic_copy(action.src, action.dst)
        elif action.verb == "merge":
            assert action.dst is not None and action.payload is not None
            _atomic_text(action.dst, action.payload)
        elif action.verb == "prune":
            assert action.dst is not None
            action.dst.unlink(missing_ok=True)
        done += 1
    return done


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sync_runs.py",
        description="Merge one strikecast run store into another (plan §5.3, §8 P6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # cluster scratch -> laptop, look first\n"
            "  python scripts/sync_runs.py /scratch/$USER/runs ./runs -n\n"
            "  # laptop -> cluster scratch, one experiment\n"
            "  python scripts/sync_runs.py ./runs /scratch/$USER/runs -e count\n"
        ),
    )
    parser.add_argument("source", type=Path, help="the run store to read")
    parser.add_argument("dest", type=Path, help="the run store to merge into")
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="report the plan, write nothing"
    )
    parser.add_argument(
        "-e",
        "--experiment",
        action="append",
        default=None,
        help="only this experiment (repeatable)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="also list the files left untouched"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="resolve conflicts (config.yaml, tuning, report) in favour of the source",
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if anything was reported as a conflict"
    )
    parser.add_argument(
        "--hash-limit",
        type=int,
        default=DEFAULT_HASH_LIMIT,
        help="above this size (bytes) equal size counts as equal content",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = plan_sync(
        args.source.expanduser().resolve(),
        args.dest.expanduser().resolve(),
        experiments=args.experiment,
        force=args.force,
        hash_limit=args.hash_limit,
    )
    print(plan.report(verbose=args.verbose))
    n = apply_plan(plan, dry_run=args.dry_run)
    verb = "would perform" if args.dry_run else "performed"
    print(f"sync_runs: {verb} {n} action(s) on {args.dest}")
    if plan.conflicts:
        print(
            f"sync_runs: {len(plan.conflicts)} conflict(s) left as they are; "
            "re-run with --force to prefer the source",
            file=sys.stderr,
        )
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
