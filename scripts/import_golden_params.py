#!/usr/bin/env python3
"""Import the thesis's tuned hyper-parameters into the new run store.

The legacy studies were pickled to ``checkpoints_tune{,_diff}/<variant>_best.pkl``
and converted once, in phase 0, to

    golden/converted/tuning/<source_dir>/<variant>/{best_params.json,trials.csv}

``strikecast.pipeline.run_stage.resolve_params`` looks for the same two files
under ``RunStore.tuning_dir(experiment, model)``, i.e.

    <store-root>/<experiment>/tuning/<model>/{best_params.json,trials.csv}

so importing is a copy plus the ``source_dir -> experiment`` mapping, which is
``strikecast.models.rnn.GOLDEN_TUNING_DIR`` inverted (``checkpoints_tune`` is
the count family, ``checkpoints_tune_diff`` the diff family). The converted
variant directory name IS the registry model name in both families, which this
script verifies against the registry when it can import it.

The copy is byte-for-byte: ``best_params.json`` already carries the
:data:`strikecast.tuning.BEST_PARAMS_KEYS` payload that ``load_best_params``
reads, and nothing is re-derived here. Level E/F depends on these numbers being
the thesis's own, so re-writing them would defeat the point.

Idempotent: a destination that already holds identical bytes is left alone; a
destination that differs is only overwritten with ``--force`` (otherwise it is
reported and the exit code is 1), so a freshly tuned study is never silently
clobbered by the golden one.

Second source: ``_regression_GBDT.ipynb`` cell 56 (audit 2026-09-26 B1)
---------------------------------------------------------------------
Only ``lightgbm_poisson`` of the six count GBDTs has a converted study. The
other five were tuned on Colab and their pickles are lost, but cell 56 of
``_regression_GBDT.ipynb`` (source cleared, output kept) printed
``best_params_by_variant.items()`` for ALL six at full float64 precision. The
output is parsed with :func:`ast.literal_eval` (never ``eval``), its
``lightgbm_poisson`` entry is required to equal the converted golden JSON
exactly (so the parse is proven on a known answer), and the five missing
variants are written as ``<store-root>/count/tuning/<variant>/best_params.json``
in the :data:`strikecast.tuning.BEST_PARAMS_KEYS` format -- the same place the
lightgbm_poisson import lands, which is what ``resolve_params`` reads. The
study statistics that did not survive (``best_value``, trial counts, study
name) are ``null``; ``source_file`` names the notebook cell. Count CatBoost-
Tweedie with these params reproduces golden fold 0 to 1.8e-15 (audit B1).

When ``golden/from_cluster/checkpoints_tune/lightgbm_tweedie_best.pkl``
(``(best_params, optuna.Study)``) exists, its ``best_params`` are cross-checked
against cell 56 and a mismatch is an error.

These params were tuned on the LEGACY ``expdecay7`` features (thesis mode,
``legacy=count``); the publication runs re-tune (decision D4) and must use a
different store root.

Usage::

    python scripts/import_golden_params.py --dry-run
    python scripts/import_golden_params.py
    python scripts/import_golden_params.py --experiment diff --model lightgbm
    python scripts/import_golden_params.py --store-root runs --force
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: ``golden/converted/tuning/<source_dir>`` -> experiment name.
#: The inverse of ``strikecast.models.rnn.GOLDEN_TUNING_DIR``.
SOURCE_DIR_TO_EXPERIMENT: dict[str, str] = {
    "checkpoints_tune": "count",
    "checkpoints_tune_diff": "diff",
}

#: What ``resolve_params`` / ``load_best_params`` need, and what else is worth
#: carrying over. ``best_params.json`` is required; ``trials.csv`` is optional
#: (only the reports read it).
REQUIRED_FILES = ("best_params.json",)
OPTIONAL_FILES = ("trials.csv",)

DEFAULT_GOLDEN = Path("golden")
DEFAULT_STORE_ROOT = Path("runs")

#: ``_regression_GBDT.ipynb`` cell 56: ``best_params_by_variant.items()``.
DEFAULT_NOTEBOOK = Path("_regression_GBDT.ipynb")
CELL56_INDEX = 56
#: Count GBDTs whose ONLY surviving params are cell 56 (audit B1).
CELL56_VARIANTS = (
    "lightgbm_tweedie",
    "xgboost_poisson",
    "xgboost_tweedie",
    "catboost_poisson",
    "catboost_tweedie",
)
#: The cell-56 entry that must equal the converted golden study exactly.
CELL56_ANCHOR = "lightgbm_poisson"
#: Where the cluster's lightgbm_tweedie pickle lands when it is copied over.
CLUSTER_PICKLE = Path("from_cluster") / "checkpoints_tune" / "lightgbm_tweedie_best.pkl"


@dataclass(frozen=True)
class Item:
    """One ``(experiment, model)`` import."""

    experiment: str
    model: str
    source: Path
    destination: Path

    @property
    def label(self) -> str:
        return f"{self.experiment}/{self.model}"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover(golden: Path) -> list[Item]:
    """Every convertible ``<source_dir>/<variant>`` under ``golden/converted/tuning``."""
    root = golden / "converted" / "tuning"
    items: list[Item] = []
    if not root.is_dir():
        return items
    for source_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        experiment = SOURCE_DIR_TO_EXPERIMENT.get(source_dir.name)
        if experiment is None:
            print(
                f"warning: {source_dir.name} is not a known tuning source directory; "
                f"known: {sorted(SOURCE_DIR_TO_EXPERIMENT)}",
                file=sys.stderr,
            )
            continue
        for variant in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            if not (variant / "best_params.json").is_file():
                continue
            items.append(
                Item(experiment, variant.name, variant, Path(experiment) / "tuning" / variant.name)
            )
    return items


def check_against_registry(items: list[Item]) -> list[str]:
    """Names the registry does not know, as warnings (never fatal).

    The registry pulls in lightgbm/xgboost/catboost/torch, so a missing model
    layer must not stop an import; it is only a cross-check that the converted
    directory names really are model-variant names.
    """
    try:
        from strikecast.models.registry import registered_names  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"registry unavailable ({type(exc).__name__}: {exc}); names not cross-checked"]

    problems = []
    for item in items:
        try:
            known = set(registered_names(item.experiment))
        except Exception as exc:  # pragma: no cover - defensive
            problems.append(f"{item.experiment}: {type(exc).__name__}: {exc}")
            continue
        if item.model not in known:
            problems.append(
                f"{item.label}: not a registered model variant of experiment "
                f"{item.experiment!r}"
            )
    return problems


def _validate_payload(path: Path) -> str:
    """The variant name recorded inside ``best_params.json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("variant", "best_params") if k not in payload]
    if missing:
        raise ValueError(f"{path} is not a tuning artefact: missing {missing}")
    if not isinstance(payload["best_params"], dict) or not payload["best_params"]:
        raise ValueError(f"{path}: best_params is empty")
    return str(payload["variant"])


def import_item(
    item: Item, store_root: Path, *, dry_run: bool, force: bool
) -> tuple[str, list[str]]:
    """Copy one variant's files. Returns ``(status, messages)``.

    ``status`` is ``"copied"``, ``"unchanged"``, ``"conflict"`` or ``"error"``.
    """
    messages: list[str] = []
    try:
        recorded = _validate_payload(item.source / "best_params.json")
    except Exception as exc:
        return "error", [f"{item.label}: {type(exc).__name__}: {exc}"]
    if recorded != item.model:
        messages.append(
            f"{item.label}: best_params.json records variant {recorded!r}, "
            f"directory says {item.model!r}"
        )

    destination = store_root / item.destination
    status = "unchanged"
    for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
        source_file = item.source / name
        if not source_file.is_file():
            if name in REQUIRED_FILES:
                return "error", [*messages, f"{item.label}: {name} is missing from {item.source}"]
            continue
        target = destination / name
        if target.is_file():
            if _digest(target) == _digest(source_file):
                continue
            if not force:
                messages.append(
                    f"{item.label}: {target} exists with different contents; "
                    f"re-run with --force to overwrite"
                )
                status = "conflict"
                continue
            messages.append(f"{item.label}: overwriting {target}")
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(source_file, tmp)
            tmp.replace(target)
        if status != "conflict":
            status = "copied"
    return status, messages


# --------------------------------------------------------------------------- #
# cell 56 of _regression_GBDT.ipynb (audit B1)
# --------------------------------------------------------------------------- #
def _cell_text(notebook: Path, cell_index: int) -> str:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    cells = payload["cells"]
    if cell_index >= len(cells):
        raise ValueError(f"{notebook} has {len(cells)} cells, no cell {cell_index}")
    cell = cells[cell_index]
    texts = []
    for output in cell.get("outputs", []):
        if "text" in output:
            texts.append("".join(output["text"]))
        elif "text/plain" in output.get("data", {}):
            texts.append("".join(output["data"]["text/plain"]))
    if len(texts) != 1:
        raise ValueError(f"{notebook} cell {cell_index}: expected one text output, got {len(texts)}")
    return texts[0].strip()


def parse_cell56(notebook: Path, cell_index: int = CELL56_INDEX) -> dict[str, dict[str, Any]]:
    """``variant -> best_params`` from the ``dict_items([...])`` repr in cell 56.

    Parsed with :func:`ast.literal_eval` on the list inside ``dict_items(...)``
    -- literals only, nothing is executed -- and type-checked: a list of
    ``(str, {str: int | float})`` pairs with finite numbers and no duplicates.
    Python's float repr round-trips exactly, so the values are the study's own
    float64s.
    """
    text = _cell_text(notebook, cell_index)
    prefix, suffix = "dict_items(", ")"
    if not (text.startswith(prefix) and text.endswith(suffix)):
        raise ValueError(
            f"{notebook} cell {cell_index} does not hold a dict_items(...) repr: {text[:60]!r}"
        )
    pairs = ast.literal_eval(text[len(prefix) : -len(suffix)])
    if not isinstance(pairs, list):
        raise ValueError(f"cell {cell_index}: expected a list of pairs, got {type(pairs).__name__}")
    out: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if not (isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str)):
            raise ValueError(f"cell {cell_index}: not a (variant, params) pair: {pair!r}")
        variant, params = pair
        if not isinstance(params, dict) or not params:
            raise ValueError(f"cell {cell_index}: {variant}: params is not a non-empty dict")
        for name, value in params.items():
            if not isinstance(name, str) or isinstance(value, bool):
                raise ValueError(f"cell {cell_index}: {variant}: bad entry {name!r}: {value!r}")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"cell {cell_index}: {variant}: {name}={value!r} is not finite")
        if variant in out:
            raise ValueError(f"cell {cell_index}: duplicate variant {variant!r}")
        out[variant] = dict(params)
    return out


def cell56_payload(variant: str, params: dict[str, Any], cell_index: int) -> dict[str, Any]:
    """``best_params.json`` in the :data:`BEST_PARAMS_KEYS` layout.

    The study statistics did not survive the lost pickles and are ``null``.
    """
    return {
        "variant": variant,
        "source_dir": None,
        "source_file": f"{DEFAULT_NOTEBOOK.name}#cell{cell_index} (best_params_by_variant.items())",
        "best_params": dict(params),
        "best_value": None,
        "n_trials": None,
        "n_pruned": None,
        "n_complete": None,
        "n_failed": None,
        "sampler": "TPESampler",
        "pruner": "MedianPruner",
        "direction": "MINIMIZE",
        "study_name": None,
        "provenance": {
            "audit": "2026-09-26 B1",
            "features": "legacy expdecay7 (series.window.expdecay=legacy_alpha)",
            "note": "tuned on Google Colab; pickle lost, params recovered from the saved "
            "notebook output at full float64 precision",
        },
    }


def check_cell56_anchor(cell: dict[str, dict[str, Any]], golden: Path) -> list[str]:
    """Problems if cell 56's ``lightgbm_poisson`` is not the converted study's."""
    path = golden / "converted" / "tuning" / "checkpoints_tune" / CELL56_ANCHOR / "best_params.json"
    if not path.is_file():
        return [f"cannot anchor the cell-56 parse: {path} is missing"]
    want = json.loads(path.read_text(encoding="utf-8"))["best_params"]
    got = cell.get(CELL56_ANCHOR)
    if got != want:
        return [f"cell 56 {CELL56_ANCHOR} != {path}: {got!r} vs {want!r}"]
    return []


def check_cluster_pickle(cell: dict[str, dict[str, Any]], golden: Path) -> tuple[str, list[str]]:
    """``("absent"|"match"|"mismatch"|"error", messages)`` for the cluster pkl."""
    path = golden / CLUSTER_PICKLE
    if not path.is_file():
        return "absent", []
    import pickle  # noqa: PLC0415  (a local, trusted artefact of the thesis run)

    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)  # noqa: S301
        best = obj[0] if isinstance(obj, tuple) else getattr(obj, "best_params", obj)
        study = obj[1] if isinstance(obj, tuple) and len(obj) > 1 else None
    except Exception as exc:  # pragma: no cover - depends on the artefact
        return "error", [f"{path}: {type(exc).__name__}: {exc}"]
    if study is not None and hasattr(study, "best_params") and study.best_params != best:
        return "mismatch", [f"{path}: the pickled study's best_params differ from its dict"]
    if dict(best) != cell.get("lightgbm_tweedie"):
        return "mismatch", [
            f"{path}: lightgbm_tweedie {dict(best)!r} != cell 56 {cell.get('lightgbm_tweedie')!r}"
        ]
    return "match", []


def import_cell56(
    notebook: Path,
    golden: Path,
    store_root: Path,
    *,
    variants: list[str] | None,
    dry_run: bool,
    force: bool,
) -> tuple[dict[str, int], list[str]]:
    """Write the five cell-56 variants; returns ``(counts, lines)``."""
    counts = {"copied": 0, "unchanged": 0, "conflict": 0, "error": 0}
    lines: list[str] = []
    try:
        cell = parse_cell56(notebook)
    except Exception as exc:
        counts["error"] += 1
        return counts, [f"FAIL    cell 56 of {notebook}: {type(exc).__name__}: {exc}"]
    missing = [v for v in (CELL56_ANCHOR, *CELL56_VARIANTS) if v not in cell]
    problems = check_cell56_anchor(cell, golden)
    if missing:
        problems.append(f"cell 56 lacks {missing}")
    status, pkl_messages = check_cluster_pickle(cell, golden)
    lines.append(f"cluster pickle {golden / CLUSTER_PICKLE}: {status}")
    if status in {"mismatch", "error"}:
        problems.extend(pkl_messages)
    if problems:
        counts["error"] += 1
        return counts, [*lines, *(f"FAIL    {p}" for p in problems)]
    lines.append(f"ok      cell 56 {CELL56_ANCHOR} == golden/converted (parse anchored)")

    for variant in CELL56_VARIANTS:
        if variants and variant not in variants:
            continue
        target = store_root / "count" / "tuning" / variant / "best_params.json"
        body = json.dumps(cell56_payload(variant, cell[variant], CELL56_INDEX), indent=2) + "\n"
        if target.is_file():
            if target.read_text(encoding="utf-8") == body:
                counts["unchanged"] += 1
                lines.append(f"ok      count/{variant:26s} <- cell 56")
                continue
            if not force:
                counts["conflict"] += 1
                lines.append(
                    f"SKIP    count/{variant:26s} {target} differs (a tuned study?); --force "
                    "overwrites"
                )
                continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(target)
        counts["copied"] += 1
        lines.append(f"{'would ' if dry_run else ''}import  count/{variant:26s} <- cell 56")
    return counts, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, type=Path, help="golden/ directory")
    parser.add_argument(
        "--store-root", default=DEFAULT_STORE_ROOT, type=Path, help="run store root (default: runs)"
    )
    parser.add_argument("--experiment", action="append", help="restrict to these experiments")
    parser.add_argument("--model", action="append", help="restrict to these model variants")
    parser.add_argument("-n", "--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument(
        "--force", action="store_true", help="overwrite a destination whose contents differ"
    )
    parser.add_argument(
        "--notebook",
        default=DEFAULT_NOTEBOOK,
        type=Path,
        help="the notebook whose cell 56 holds the count GBDT params (audit B1)",
    )
    parser.add_argument(
        "--no-cell56", action="store_true", help="skip the cell-56 count GBDT import"
    )
    args = parser.parse_args(argv)

    golden = args.golden.resolve()
    store_root = args.store_root.resolve()
    if not golden.is_dir():
        print(f"error: {golden} is not a directory", file=sys.stderr)
        return 2

    items = discover(golden)
    if args.experiment:
        wanted = set(args.experiment)
        items = [i for i in items if i.experiment in wanted]
    if args.model:
        wanted = set(args.model)
        items = [i for i in items if i.model in wanted]

    cell56_counts = {"copied": 0, "unchanged": 0, "conflict": 0, "error": 0}
    wants_cell56 = (
        not args.no_cell56
        and (not args.experiment or "count" in args.experiment)
        and (not args.model or any(m in CELL56_VARIANTS for m in args.model))
    )
    if wants_cell56:
        cell56_counts, lines = import_cell56(
            args.notebook.resolve(),
            golden,
            store_root,
            variants=args.model,
            dry_run=args.dry_run,
            force=args.force,
        )
        for line in lines:
            print(line)
    if not items and not wants_cell56:
        print(f"error: no converted tuning artefacts found under {golden / 'converted' / 'tuning'}")
        return 2

    for problem in check_against_registry(items):
        print(f"warning: {problem}", file=sys.stderr)

    counts = dict(cell56_counts)
    for item in items:
        status, messages = import_item(item, store_root, dry_run=args.dry_run, force=args.force)
        counts[status] += 1
        for message in messages:
            print(f"warning: {message}", file=sys.stderr)
        verb = {"copied": "import", "unchanged": "ok", "conflict": "SKIP", "error": "FAIL"}[status]
        prefix = "would " if args.dry_run and status == "copied" else ""
        print(f"{prefix}{verb:7s} {item.label:32s} <- {item.source.relative_to(golden)}")

    n_cell56 = sum(cell56_counts.values())
    print(
        f"\n{len(items) + n_cell56} variants: {counts['copied']} copied, {counts['unchanged']} unchanged, "
        f"{counts['conflict']} conflicting, {counts['error']} failed"
        + (" (dry run, nothing written)" if args.dry_run else f" -> {store_root}")
    )
    return 1 if counts["conflict"] or counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
