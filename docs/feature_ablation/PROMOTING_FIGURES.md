# Promoting a feature-ablation figure into the thesis figure set

`strikecast figures --store-root <store>` builds every registered item of
`src/strikecast/reporting/build.py` into `<store>/_figures/` (plus `MANIFEST.md`). The
feature-ablation figures live outside that registry until one is wanted in the thesis. This is
the checklist to move one in (F1 to F5 of `figures.py`; `<name>` below is e.g.
`selector_value`, `single_groups`, `leave_one_out`, `per_horizon`, `cumulative_path`).

`figures.py` is already split for this: `*_frame()` turns the analysis tables into one tidy
frame per figure, `plot_*()` takes that frame and returns a matplotlib `Figure` (no I/O, no rc
changes). Both move into the package unchanged; only the loading changes.

## Checklist

1. **Move the code into the package** (`docs/` is not importable):
   - `src/strikecast/reporting/feature_ablation.py`: copy `GROUPS`, `GROUP_LABELS`, the colour
     constants and the `*_frame` / `plot_*` functions of `figures.py` verbatim, and the
     computation of `analysis.py` (`Stores`, `complete`, `load`, `feature_space`, `contrast`,
     `analyse_run`, `holm`) so the builder can produce `metrics` / `results` / `per_horizon`
     frames from the stores. Then make `docs/feature_ablation/{analysis,figures}.py` import from
     there so there is one copy.
   - Keep `analyse_run`'s printing optional (a `verbose=False` flag) so the builder stays quiet.

2. **Sources helper** in `src/strikecast/reporting/sources.py`: tell the build where the
   ablation stores are, and raise `MissingInput` when they are absent so the item is
   `skipped`, never `failed`:

   ```python
   # ResultsSource
   def feature_ablation_root(self) -> Path:
       raise MissingInput("feature ablation needs the run store source")

   # StoreSource
   def feature_ablation_root(self) -> Path:
       root = self.root.parent / "runs_feature_ablation"   # sibling of the publication store
       if not root.is_dir():
           raise MissingInput(f"missing {root} (rsync it from the cluster, docs/feature_ablation/README.md)")
       return root
   ```

   (If the stores live elsewhere, add a `--feature-ablation-root` option to `strikecast figures`
   in `cli/main.py::_cmd_figures` and pass it through `make_source`.)

3. **Shared, lazily computed input** on `BuildContext` (`build.py`, next to `master` / `tiers`),
   so several builders reuse one analysis pass:

   ```python
   @functools.cached_property
   def feature_ablation(self) -> dict[str, pd.DataFrame]:
       from strikecast.reporting import feature_ablation as FA  # noqa: PLC0415
       stores = FA.Stores(self.source.feature_ablation_root(), self.source.root, selftest=False)
       metrics, per_h, results = FA.analyse_run("catboost_tweedie", "global", stores,
                                                FA.read_variants(...), n_boot=1000,
                                                ref_rmse=FA.reference_rmse(self.source.root))
       return {"metrics": metrics, "results": results, "per_horizon": per_h}
   ```

   `read_variants(...)` needs `docs/feature_ablation/variants.tsv`; either pass its path (repo
   root is `Path(__file__).parents[3]`) or move the table next to the module.

4. **Builder** in `build.py` (pattern: `build_importance_share`), one per figure:

   ```python
   def build_feature_ablation_<name>(ctx: BuildContext) -> list[Path]:
       """F24: <one line>."""
       from strikecast.reporting import feature_ablation as FA  # noqa: PLC0415

       fa = ctx.feature_ablation
       frame = FA.selector_frame(fa["metrics"], fa["results"])      # or forest_frame(..., section="R3") ...
       if frame.empty:
           raise MissingInput("feature ablation: <which variants> not complete")
       ctx.out.mkdir(parents=True, exist_ok=True)
       csv = ctx.out / "feature_ablation_<name>.csv"
       frame.to_csv(csv, index=False, float_format="%.10g", lineterminator="\n")
       with thesis_style():
           fig = FA.plot_selector(frame)                               # the matching plot_*
           return [ctx.figure("feature_ablation_<name>.svg", fig), csv]
   ```

   `ctx.figure` writes a byte-stable SVG (`style.save_svg`). Drop the PNG; the thesis uses SVG.

5. **Register** it in `_items()` (the list around `build.py:700-810`; IDs F1-F23 are taken, so
   start at F24):

   ```python
   Item("F24", "fig:feature_ablation_<name>", None, "EXTRA", "figure",
        ("feature_ablation_<name>.svg", "feature_ablation_<name>.csv"),
        "runs_feature_ablation/<variants> test predictions + feature_space.json (catboost_tweedie@global)",
        build_feature_ablation_<name>, "docs/feature_ablation/figures.py F1; see its README"),
   ```

   `line` stays `None` while the figure is not in `main.tex`; once Jan includes it, put the
   `\includegraphics` line number there and change the class from `"EXTRA"` to `"RESULTS"`.

6. **Check**: `uv run strikecast figures --store-root runs_publication_20260929 --only F24`
   (`skipped: missing .../runs_feature_ablation` without the stores, `generated` with them), then
   open the SVG. Add a unit test that runs the builder on a tiny synthetic store, or at least
   `plot_*` on the CSV from `docs/feature_ablation/output/`.

Shortcut if the store-based path is too much: a builder that reads the analysis CSVs from
`docs/feature_ablation/output/` (`MissingInput` when absent) and calls the same `*_frame` /
`plot_*`. Quicker, but the figure then depends on someone having run `analysis.py` first, which
`strikecast figures` cannot see; prefer the store path for anything that goes into the paper.
