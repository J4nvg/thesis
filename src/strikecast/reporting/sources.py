"""Two data back-ends behind one interface (audit C design (a).1).

A *model row* is named the way ``results/analyse_results.ipynb`` names it:
``(family, paradigm, model)`` with ``family`` in the notebook's ``Modelname``
vocabulary (``diff``, ``log``, ``lstm``, ``chronos2``, ``gbdt``,
``finalhurdle``). :class:`LegacySource` reads ``results/<family>/`` with the
notebook's own path rules; :class:`StoreSource` maps the run store's
``(experiment, model, paradigm)`` onto the same vocabulary:

=============  ==========================================================
store          family
=============  ==========================================================
``count``      ``lstm`` for the RNNs (``lstm*``/``gru*``), ``gbdt`` otherwise
``diff``       ``diff``
``chronos2``   ``chronos2`` (store paradigm ``global`` shown as ``local``,
               the thesis' name for "regions as one multivariate predictor")
``hurdle``     ``finalhurdle`` (primary component ``hurdle_cal``, D6)
=============  ==========================================================

The archived ``log`` family exists only in ``results/`` (B18) and damage is not
a thesis output (B19/B20). Store model names carry no ``_tuned`` suffix.

Every accessor raises :class:`MissingInput` (never returns something partial)
so the builder can record the item as skipped with the reason.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "LEGACY_FAMILIES",
    "STORE_EXPERIMENTS",
    "LegacySource",
    "MissingInput",
    "ResultsSource",
    "StoreSource",
    "make_source",
]

logger = logging.getLogger(__name__)

#: The families ``analyse_results.ipynb`` cell 4 concatenates, in its order.
LEGACY_FAMILIES: tuple[str, ...] = ("diff", "log", "lstm", "chronos2", "gbdt", "finalhurdle")

#: Store experiment -> how its rows enter the master leaderboard.
STORE_EXPERIMENTS: tuple[str, ...] = ("diff", "count", "chronos2", "hurdle")

_RNN_PREFIXES = ("lstm", "gru")
_BEST_PARAM_KEYS = {
    # T8 row -> (legacy golden path relative to golden/, store (experiment, model))
    "catboost_tweedie": (None, ("count", "catboost_tweedie")),
    "lightgbm_poisson": (
        "converted/tuning/checkpoints_tune/lightgbm_poisson/best_params.json",
        ("count", "lightgbm_poisson"),
    ),
    "lstm_poisson_w28": (
        "converted/tuning/checkpoints_tune/lstm_poisson_w28/best_params.json",
        ("count", "lstm_poisson_w28"),
    ),
    "chronos2_fine_tuned": (
        "checkpoints/chronos2_best/best_params.json",
        ("chronos2", "chronos2_fine_tuned"),
    ),
}


#: Chronos-2 store name -> suffix of ``results/chronos2/feature_importance_chronos2_<s>.csv``.
_LEGACY_CHRONOS_FI = {"chronos2_fine_tuned": "ft", "chronos2_zero_shot": "zs"}


class MissingInput(RuntimeError):
    """An input an item needs is not in this source (the item is skipped)."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise MissingInput(f"missing {path}")
    return pd.read_csv(path)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise MissingInput(f"missing {path}")
    return pd.read_parquet(path)


class ResultsSource:
    """The interface; see the two implementations."""

    name = "abstract"

    # -- leaderboards ------------------------------------------------------
    def leaderboards(self) -> pd.DataFrame:
        """Test-split rows ``Modelname, paradigm, model, MAE, RMSE`` of every family."""
        raise NotImplementedError

    # -- per-model views -----------------------------------------------------
    def per_region(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        raise NotImplementedError

    def per_horizon(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        raise NotImplementedError

    def predictions(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        """Long test predictions with ``region, horizon, y_true, y_pred``."""
        raise NotImplementedError

    # -- hurdle ---------------------------------------------------------------
    def hurdle_classifier_probs(self) -> pd.DataFrame:
        """``region, fold, horizon, date, y_true, y_prob_uncal, y_prob_cal`` (test)."""
        raise NotImplementedError

    def hurdle_regressor_predictions(self) -> pd.DataFrame:
        """The count head's test predictions (all rows) ``region, y_true, y_pred``."""
        raise NotImplementedError

    # -- importance / tuning --------------------------------------------------
    def gbdt_importance(self) -> pd.DataFrame:
        """``results/gbdt/importance_all.csv`` schema (``model``, ``Feature``, ...)."""
        raise NotImplementedError

    def chronos_importance(self, model: str = "chronos2_fine_tuned") -> pd.DataFrame:
        """``Feature, importance, ...`` of one Chronos-2 variant (``chronos2_fine_tuned`` /
        ``chronos2_zero_shot``)."""
        raise NotImplementedError

    def best_params(self, key: str) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


# --------------------------------------------------------------------------- #
# legacy results/
# --------------------------------------------------------------------------- #
class LegacySource(ResultsSource):
    """``results/`` read with the path rules of ``analyse_results.ipynb``.

    Parameters
    ----------
    results_dir:
        the thesis' ``results/`` folder;
    golden_dir:
        ``golden/`` (the converted tuning studies and the Chronos params, T8);
    gbdt_notebook:
        ``_regression_GBDT.ipynb``, whose cell 56 output holds the count GBDT
        params that survive only there (audit B1; read as a literal, nothing
        is executed).
    """

    name = "legacy"

    def __init__(
        self,
        results_dir: str | Path = "results",
        golden_dir: str | Path = "golden",
        gbdt_notebook: str | Path = "_regression_GBDT.ipynb",
    ) -> None:
        self.root = Path(results_dir)
        self.golden = Path(golden_dir)
        self.gbdt_notebook = Path(gbdt_notebook)
        if not self.root.is_dir():
            raise MissingInput(f"results directory {self.root} does not exist")

    def describe(self) -> str:
        return f"legacy results/ at {self.root}"

    def leaderboards(self) -> pd.DataFrame:
        frames = []
        for family in LEGACY_FAMILIES:
            path = self.root / family / "leaderboard.csv"
            if not path.is_file():
                logger.warning("legacy leaderboard missing: %s", path)
                continue
            df = pd.read_csv(path)
            if "split" in df.columns:
                df = df[df["split"] == "test"].copy()
            df["Modelname"] = family
            if "model" not in df.columns:
                df["model"] = family
            if "paradigm" not in df.columns:
                df["paradigm"] = "local"
            frames.append(df[["Modelname", "paradigm", "model", "MAE", "RMSE"]])
        if not frames:
            raise MissingInput(f"no leaderboard.csv under {self.root}")
        return pd.concat(frames, ignore_index=True)

    # the notebook's path rules (cells 18/19, 26, 27, 29)
    def _region_path(self, family: str, paradigm: str, model: str) -> Path:
        if family == "finalhurdle":
            return self.root / family / "per_region_test_global_hurdle_cal.csv"
        if family == "chronos2":
            return self.root / family / f"per_region_{model}.csv"
        return self.root / family / f"per_region_test_{paradigm}_{model}.csv"

    def _horizon_path(self, family: str, paradigm: str, model: str) -> Path:
        if family == "finalhurdle":
            return self.root / family / "per_horizon_test_global_hurdle_cal.csv"
        if family == "chronos2":
            return self.root / family / f"per_horizon_{model}.csv"
        return self.root / family / f"per_horizon_test_{paradigm}_{model}.csv"

    def _pred_path(self, family: str, paradigm: str, model: str) -> Path:
        if family == "finalhurdle":
            return self.root / "finalhurdle" / "global_test_hurdle_cal_preds.parquet"
        if family == "chronos2":
            return self.root / "chronos2" / f"predictions_long_{model}.parquet"
        return self.root / family / f"predictions_long_test_{paradigm}_{model}.parquet"

    def per_region(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        return _read_csv(self._region_path(family, paradigm, model))

    def per_horizon(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        return _read_csv(self._horizon_path(family, paradigm, model))

    def predictions(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        return _read_parquet(self._pred_path(family, paradigm, model))

    def hurdle_classifier_probs(self) -> pd.DataFrame:
        # analyse_results.ipynb cell 34 `load_and_merge_data`
        uncal = _read_parquet(self.root / "finalhurdle" / "global_test_classifier_probs.parquet")
        cal = _read_parquet(self.root / "finalhurdle" / "global_test_classifier_cal_probs.parquet")
        keys = ["region", "fold", "horizon", "date"]
        uncal = uncal.rename(columns={"y_prob": "y_prob_uncal"})
        cal = cal.rename(columns={"y_prob": "y_prob_cal"})
        return pd.merge(uncal, cal[[*keys, "y_prob_cal"]], on=keys, how="inner")

    def hurdle_regressor_predictions(self) -> pd.DataFrame:
        return _read_parquet(self.root / "finalhurdle" / "global_test_regressor_preds.parquet")

    def gbdt_importance(self) -> pd.DataFrame:
        return _read_csv(self.root / "gbdt" / "importance_all.csv")

    def chronos_importance(self, model: str = "chronos2_fine_tuned") -> pd.DataFrame:
        if model not in _LEGACY_CHRONOS_FI:
            raise MissingInput(f"no legacy Chronos-2 importance file for {model!r}")
        df = _read_csv(self.root / "chronos2" / f"feature_importance_chronos2_{_LEGACY_CHRONOS_FI[model]}.csv")
        return df.rename(columns={"Unnamed: 0": "Feature"})

    def best_params(self, key: str) -> dict[str, Any]:
        if key not in _BEST_PARAM_KEYS:
            raise MissingInput(f"no best-params mapping for {key!r}")
        rel, _ = _BEST_PARAM_KEYS[key]
        if rel is None:
            return _cell56_params(self.gbdt_notebook, key)
        path = self.golden / rel
        if not path.is_file():
            raise MissingInput(f"missing {path}")
        return dict(json.loads(path.read_text(encoding="utf-8"))["best_params"])


def _cell56_params(notebook: Path, variant: str, cell_index: int = 56) -> dict[str, Any]:
    """``_regression_GBDT.ipynb`` cell 56 output (``dict_items([...])``), literal only.

    Same parse as ``scripts/import_golden_params.parse_cell56`` (audit B1).
    """
    if not notebook.is_file():
        raise MissingInput(f"missing {notebook} (cell 56 holds the {variant} params, B1)")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    texts = []
    for output in cells[cell_index].get("outputs", []):
        if "text" in output:
            texts.append("".join(output["text"]))
        elif "text/plain" in output.get("data", {}):
            texts.append("".join(output["data"]["text/plain"]))
    text = "".join(texts).strip()
    if not (text.startswith("dict_items(") and text.endswith(")")):
        raise MissingInput(f"{notebook} cell {cell_index} holds no dict_items(...) output")
    pairs = dict(ast.literal_eval(text[len("dict_items(") : -1]))
    if variant not in pairs:
        raise MissingInput(f"{notebook} cell {cell_index} has no {variant!r}")
    return dict(pairs[variant])


# --------------------------------------------------------------------------- #
# run store
# --------------------------------------------------------------------------- #
def _family_of(experiment: str, model: str) -> str:
    if experiment == "count":
        return "lstm" if model.startswith(_RNN_PREFIXES) else "gbdt"
    if experiment == "hurdle":
        return "finalhurdle"
    return experiment


class StoreSource(ResultsSource):
    """The run store at one evaluation seed (default 42, the thesis seed).

    Seed aggregation (mean/CI over ``seeds.eval_seeds``) is deliberately not
    done here: the thesis numbers are single-seed and the per-experiment
    ``strikecast report`` already writes the seed CIs.
    """

    name = "store"

    def __init__(self, root: str | Path, *, seed: int = 42, tuning_seed: int = 42) -> None:
        from strikecast.store.run_store import RunStore  # noqa: PLC0415

        self.root = Path(root)
        if not self.root.is_dir():
            raise MissingInput(f"store root {self.root} does not exist")
        self.store = RunStore(self.root)
        self.seed = int(seed)
        self.tuning_seed = int(tuning_seed)

    def describe(self) -> str:
        return f"run store at {self.root} (seed={self.seed})"

    # family/paradigm/model -> store run key
    def _key(self, family: str, paradigm: str, model: str) -> tuple[str, str, str]:
        if family in ("gbdt", "lstm"):
            experiment = "count"
        elif family == "finalhurdle":
            experiment = "hurdle"
            model = "hurdle" if model == "finalhurdle" else model
        else:
            experiment = family
        if experiment == "chronos2" and paradigm == "local":
            paradigm = "global"
        if experiment == "diff" and model == "naive_weekly" and paradigm == "local":
            # analyse_results.ipynb cell 11 relabels the seasonal naive `local` for
            # display (results/diff/ stores its test views as *_local_naive_weekly);
            # the store ran it under `global`.
            paradigm = "global"
        return experiment, model, paradigm

    def _run(self, family: str, paradigm: str, model: str):
        from strikecast.store.run_store import RunKey  # noqa: PLC0415

        experiment, model, paradigm = self._key(family, paradigm, model)
        return RunKey(experiment, model, paradigm, self.seed)

    def leaderboards(self) -> pd.DataFrame:
        from strikecast.evaluation.leaderboard import collect_metric_rows  # noqa: PLC0415

        records = []
        for experiment in STORE_EXPERIMENTS:
            if not (self.root / experiment).is_dir():
                continue
            rows = collect_metric_rows(
                self.root, experiment=experiment, stages=["test"], seeds=[self.seed]
            )
            for row in rows:
                if "MAE" not in row.metrics or "RMSE" not in row.metrics:
                    continue
                paradigm = row.paradigm
                if experiment == "chronos2" and paradigm == "global":
                    paradigm = "local"
                records.append(
                    {
                        "Modelname": _family_of(experiment, row.model),
                        "paradigm": paradigm,
                        "model": row.model,
                        "MAE": float(row.metrics["MAE"]),
                        "RMSE": float(row.metrics["RMSE"]),
                    }
                )
        if not records:
            raise MissingInput(f"no test-stage global.json at seed={self.seed} under {self.root}")
        return pd.DataFrame.from_records(records)

    def _metrics_csv(self, family: str, paradigm: str, model: str, view: str) -> pd.DataFrame:
        run = self._run(family, paradigm, model)
        return _read_csv(self.store.metrics_dir(run, "test") / f"{view}.csv")

    def per_region(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        return self._metrics_csv(family, paradigm, model, "per_region")

    def per_horizon(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        return self._metrics_csv(family, paradigm, model, "per_horizon")

    def _load(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        run = self._run(family, paradigm, model)
        preds = self.store.load_predictions(run, "test", legacy_order=True)
        frame = preds.frame
        if frame.empty:
            raise MissingInput(f"no test predictions under {self.store.resolve(run)}")
        return frame

    def predictions(self, family: str, paradigm: str, model: str) -> pd.DataFrame:
        frame = self._load(family, paradigm, model)
        if family == "finalhurdle":
            probs = self.hurdle_classifier_probs()
            count = frame[frame["channel"] == "count"].reset_index(drop=True)
            out = count[["region", "fold", "horizon", "date", "y_true"]].copy()
            out["y_pred"] = probs["y_prob_cal"].to_numpy() * count["y_pred"].to_numpy()
            return out
        channels = frame["channel"].unique()
        if len(channels) != 1:
            raise MissingInput(f"{family}/{model}: expected one channel, got {list(channels)}")
        return frame.drop(columns=["channel"])

    def hurdle_classifier_probs(self) -> pd.DataFrame:
        from strikecast.evaluation.calibration import (  # noqa: PLC0415
            apply_calibrators_per_horizon,
            calibrators_from_json,
        )
        from strikecast.store.run_store import RunKey  # noqa: PLC0415

        frame = self._load("finalhurdle", "global", "hurdle")
        prob = frame[frame["channel"] == "prob"].reset_index(drop=True)
        if prob.empty:
            raise MissingInput("hurdle test predictions hold no 'prob' channel")
        source = RunKey("hurdle", "hurdle", "global", self.tuning_seed)
        path = self.store.artifacts_dir(source) / "calibrators.json"
        if not path.is_file():
            raise MissingInput(f"missing calibrators {path} (hurdle cv stage, C18)")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "prob" not in payload:
            raise MissingInput(f"{path} holds no 'prob' channel")
        method = "sigmoid"
        try:
            cfg = self.store.read_config(self._run("finalhurdle", "global", "hurdle"))
            method = str((cfg.get("calibration") or {}).get("method", method))
        except Exception:  # noqa: BLE001 - the snapshot is optional here
            pass
        out = prob.rename(columns={"y_pred": "y_prob_uncal"})
        work = prob.rename(columns={"y_pred": "y_prob"})
        out["y_prob_cal"] = apply_calibrators_per_horizon(
            work, calibrators_from_json(payload["prob"]), method=method
        )
        return out[["region", "fold", "horizon", "date", "y_true", "y_prob_uncal", "y_prob_cal"]]

    def hurdle_regressor_predictions(self) -> pd.DataFrame:
        frame = self._load("finalhurdle", "global", "hurdle")
        count = frame[frame["channel"] == "count"].reset_index(drop=True)
        if count.empty:
            raise MissingInput("hurdle test predictions hold no 'count' channel")
        return count.drop(columns=["channel"])

    def gbdt_importance(self) -> pd.DataFrame:
        """Every count run's ``importance/importance.csv`` at this seed, concatenated.

        Read per run (sorted paths) rather than from the collected
        ``count/report/importance/seed=<s>/importance_all.csv``, which concurrent
        importance jobs rewrite last-writer-wins (Stream 3 hand-off); the
        collected file is the fallback. Each per-run file carries the legacy
        ``model`` label (``activity_<k>_catboost_tweedie``), so the figures only
        select by label and the row order does not matter.
        """
        runs = sorted((self.root / "count").glob(f"*/*/seed={self.seed}/importance/importance.csv"))
        if runs:
            return pd.concat([pd.read_csv(p) for p in runs], ignore_index=True)
        path = self.root / "count" / "report" / "importance" / f"seed={self.seed}"
        return _read_csv(path / "importance_all.csv")

    def chronos_importance(self, model: str = "chronos2_fine_tuned") -> pd.DataFrame:
        candidates = [
            self.root / "chronos2" / "report" / "importance" / f"seed={self.seed}"
            / f"feature_importance_{model}.csv",
            self.store.run_dir("chronos2", model, "global", self.seed)
            / "importance" / "importance.csv",
        ]
        for path in candidates:
            if path.is_file():
                return pd.read_csv(path).rename(columns={"Unnamed: 0": "Feature"})
        raise MissingInput(f"missing Chronos-2 importance (looked in {candidates[0]})")

    def best_params(self, key: str) -> dict[str, Any]:
        if key not in _BEST_PARAM_KEYS:
            raise MissingInput(f"no best-params mapping for {key!r}")
        _, (experiment, model) = _BEST_PARAM_KEYS[key]
        path = self.store.tuning_dir(experiment, model) / "best_params.json"
        if not path.is_file():
            raise MissingInput(f"missing {path}")
        return dict(json.loads(path.read_text(encoding="utf-8"))["best_params"])


def make_source(kind: str, **kwargs: Any) -> ResultsSource:
    if kind == "legacy":
        return LegacySource(
            kwargs.get("results_dir", "results"),
            kwargs.get("golden_dir", "golden"),
            kwargs.get("gbdt_notebook", "_regression_GBDT.ipynb"),
        )
    if kind == "store":
        return StoreSource(kwargs["store_root"], seed=kwargs.get("seed", 42))
    raise ValueError(f"unknown source {kind!r} (store | legacy)")
