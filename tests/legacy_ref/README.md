# `tests/legacy_ref` — legacy backtest loops, verbatim

These modules are **test-only oracles**. They hold byte-for-byte copies of the
backtest loops that the thesis actually ran, so the Phase 2 engine
(`strikecast.backtest`) and the Phase 2 forecasters (`strikecast.models`) can be
compared against them fold by fold, on identical inputs, with the legacy
behaviour — bugs included — as the definition of "correct".

## Why the copies exist

The originals cannot be imported:

* `_diff_regression.py` executes the entire pipeline (data loading, feature
  selection, tuning, plotting) at module level, so importing it runs a thesis.
* `final_hurdle.ipynb` and `damage_classifier.ipynb` keep their loops inside
  notebook cells, reading notebook-level globals.

## Contents

| module | source (commit `b1bfd79`) |
|---|---|
| `diff_runners.py` | `_diff_regression.py` lines 660-1005 |
| `hurdle_runners.py` | `final_hurdle.ipynb` cells 15, 24, 29, 34 (+ `src/ts_specific_tools.py:105`) |
| `damage_runners.py` | `damage_classifier.ipynb` cells 17, 26 |

## Rules

1. **Nothing under `src/` may import this package.** It is not part of the
   shipped library and is not on the wheel's package list
   (`pyproject.toml` ships `src/strikecast` only). If production code ever needs
   one of these behaviours, it gets reimplemented in `src/strikecast` and
   *compared* against the copy here.
2. **Do not "clean up" the bodies.** They are evidence, not code. Alignment
   whitespace, `print` calls, semicolon one-liners, shadowed names and latent
   bugs are all intentional. Lint rules that would rewrite them are disabled
   per file with a `# ruff: noqa` header that names the rule.
3. **Every deviation from the source is listed in the module docstring.** The
   only deviations allowed are mechanical ones: imports moved to the top of the
   file, and notebook/module globals turned into keyword parameters *with the
   same names* so the loop bodies stay byte-identical. The docstrings carry the
   global → parameter mapping.
4. **If a source cell changes, the copy is re-extracted, not edited.**

`diff_runners` imports `_maybe_scale_covs` from the legacy top-level `src`
package at the repository root; `tests/legacy_ref/__init__.py` puts the
repository root on `sys.path` so the modules import standalone as well as under
pytest.
