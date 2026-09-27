"""``scripts/import_golden_params.py``: the cell-56 count GBDT params (audit B1).

``_regression_GBDT.ipynb`` cell 56 lost its source but kept its output,
``dict_items([...])`` of the best params of all six count GBDTs. The importer
parses it with ``ast.literal_eval`` only, anchors the parse on the converted
lightgbm_poisson study, optionally cross-checks the cluster's
``lightgbm_tweedie_best.pkl`` and writes the five missing variants in the
``best_params.json`` format ``resolve_params`` reads.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path

import pytest

from strikecast.tuning import BEST_PARAMS_KEYS

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "_regression_GBDT.ipynb"
GOLDEN = REPO_ROOT / "golden"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location(
        "import_golden_params", REPO_ROOT / "scripts" / "import_golden_params.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]  (dataclasses need it)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _notebook(tmp_path: Path, text: str, index: int = 56) -> Path:
    cells = [{"cell_type": "code", "source": "", "outputs": []} for _ in range(index + 1)]
    cells[index]["outputs"] = [
        {"output_type": "execute_result", "data": {"text/plain": [text]}, "metadata": {}}
    ]
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return path


def test_the_real_cell_56_holds_all_six_count_gbdts(importer) -> None:
    if not NOTEBOOK.is_file():
        pytest.skip("the legacy notebook is not present")
    cell = importer.parse_cell56(NOTEBOOK)
    assert set(cell) == {"lightgbm_poisson", *importer.CELL56_VARIANTS}
    # audit B1, verbatim (full float64 precision)
    assert cell["catboost_tweedie"] == {
        "depth": 4,
        "learning_rate": 0.012071584113760261,
        "iterations": 900,
        "l2_leaf_reg": 5.0,
        "subsample": 0.9977756396798295,
        "tweedie_variance_power": 1.3947466852349566,
    }
    assert cell["xgboost_poisson"]["reg_lambda"] == 0.0036107454705973092
    assert cell["lightgbm_tweedie"]["tweedie_variance_power"] == 1.3329833121584336
    # the anchor: cell 56's lightgbm_poisson IS the converted golden study
    if not (GOLDEN / "converted").is_dir():
        pytest.skip("golden/ is git-ignored and not present (e.g. CI); anchor not checked")
    assert importer.check_cell56_anchor(cell, GOLDEN) == []


def test_parse_is_literal_only(importer, tmp_path) -> None:
    evil = _notebook(tmp_path, "dict_items([('x', {'a': __import__('os').getpid()})])")
    with pytest.raises(ValueError):
        importer.parse_cell56(evil)
    not_items = _notebook(tmp_path, "{'x': {'a': 1}}")
    with pytest.raises(ValueError, match="dict_items"):
        importer.parse_cell56(not_items)
    non_finite = _notebook(tmp_path, "dict_items([('x', {'a': float('nan')})])")
    with pytest.raises(ValueError):
        importer.parse_cell56(non_finite)
    dup = _notebook(tmp_path, "dict_items([('x', {'a': 1}), ('x', {'a': 2})])")
    with pytest.raises(ValueError, match="duplicate"):
        importer.parse_cell56(dup)


def _golden_copy(tmp_path: Path) -> Path:
    """A minimal golden/ holding only the anchor study."""
    source = GOLDEN / "converted" / "tuning" / "checkpoints_tune" / "lightgbm_poisson"
    if not (source / "best_params.json").is_file():
        pytest.skip("the converted lightgbm_poisson study is not present")
    target = tmp_path / "golden" / "converted" / "tuning" / "checkpoints_tune" / "lightgbm_poisson"
    target.mkdir(parents=True)
    shutil.copy(source / "best_params.json", target / "best_params.json")
    return tmp_path / "golden"


def test_import_writes_the_five_variants_and_is_idempotent(importer, tmp_path) -> None:
    if not NOTEBOOK.is_file():
        pytest.skip("the legacy notebook is not present")
    golden = _golden_copy(tmp_path)
    store = tmp_path / "runs"
    counts, lines = importer.import_cell56(
        NOTEBOOK, golden, store, variants=None, dry_run=False, force=False
    )
    assert counts == {"copied": 5, "unchanged": 0, "conflict": 0, "error": 0}, lines
    for variant in importer.CELL56_VARIANTS:
        payload = json.loads((store / "count" / "tuning" / variant / "best_params.json").read_text())
        assert payload["variant"] == variant
        assert set(BEST_PARAMS_KEYS) <= set(payload)
        assert payload["best_params"] == importer.parse_cell56(NOTEBOOK)[variant]
    # resolve_params' loader reads it
    from strikecast.tuning import load_best_params  # noqa: PLC0415

    assert load_best_params(store / "count" / "tuning" / "catboost_tweedie")["depth"] == 4

    again, _ = importer.import_cell56(
        NOTEBOOK, golden, store, variants=None, dry_run=False, force=False
    )
    assert again["unchanged"] == 5 and again["copied"] == 0
    # a differing (e.g. freshly tuned) study is never clobbered without --force
    target = store / "count" / "tuning" / "xgboost_poisson" / "best_params.json"
    target.write_text(json.dumps({"variant": "xgboost_poisson", "best_params": {"x": 1}}))
    clash, _ = importer.import_cell56(
        NOTEBOOK, golden, store, variants=["xgboost_poisson"], dry_run=False, force=False
    )
    assert clash["conflict"] == 1
    assert json.loads(target.read_text())["best_params"] == {"x": 1}


def test_cluster_pickle_is_cross_checked(importer, tmp_path) -> None:
    if not NOTEBOOK.is_file():
        pytest.skip("the legacy notebook is not present")
    cell = importer.parse_cell56(NOTEBOOK)
    golden = tmp_path / "golden"
    assert importer.check_cluster_pickle(cell, golden) == ("absent", [])

    path = golden / importer.CLUSTER_PICKLE
    path.parent.mkdir(parents=True)
    path.write_bytes(pickle.dumps((dict(cell["lightgbm_tweedie"]), None)))
    assert importer.check_cluster_pickle(cell, golden)[0] == "match"

    other = dict(cell["lightgbm_tweedie"], num_leaves=99)
    path.write_bytes(pickle.dumps((other, None)))
    status, messages = importer.check_cluster_pickle(cell, golden)
    assert status == "mismatch" and "lightgbm_tweedie" in messages[0]
