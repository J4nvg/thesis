"""Verbatim copies of the legacy backtest loops, for equivalence testing only.

These modules are ORACLES. They are never imported by ``src/`` and never by
production code -- see ``README.md`` next to this file.

The legacy loops live in places that cannot be imported: ``_diff_regression.py``
executes the whole pipeline at module level, and the hurdle / damage loops live
inside notebook cells. The functions here were copied out of those sources with
the loop bodies untouched, so that the Phase 2 engine can be compared against
them fold by fold.

``diff_runners`` imports ``src.evaluation_tools``, the legacy top-level ``src``
package, which lives at the repository root. ``tests/conftest.py`` already puts
the repository root on ``sys.path`` when pytest collects it, but the line below
makes these modules importable on their own (``python -c "import
tests.legacy_ref.diff_runners"``) as well.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
