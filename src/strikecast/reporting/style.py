"""Deterministic figure styling (audit C5).

* DejaVu Sans, which ships with matplotlib and is what every thesis SVG used
  (they were drawn on Linux/WSL; macOS picked ArialMT and changed the sizes);
* ``svg.hashsalt`` fixed, so matplotlib's clip-path/glyph ids are stable;
* ``metadata={"Date": None}`` plus a regex strip of any ``<dc:date>`` block;
* NO global ``rcParams`` mutation: everything runs inside
  :func:`thesis_style`, a ``matplotlib.rc_context`` seeded with seaborn's
  ``whitegrid`` axes style (what the analysis notebook's ``sns.set_theme``
  applied globally), so importing or running this module leaves the caller's
  matplotlib state untouched.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

__all__ = ["BASE_RC", "save_svg", "thesis_style"]

#: The rc overrides every figure is drawn under.
BASE_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.hashsalt": "strikecast",
    "svg.fonttype": "path",
    "path.simplify": True,
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "axes.unicode_minus": True,
}

_DC_DATE = re.compile(r"\s*<dc:date>.*?</dc:date>", re.DOTALL)


def _seaborn_rc(style: str | None, context: str = "notebook") -> dict[str, Any]:
    if style is None:
        return {}
    import seaborn as sns  # noqa: PLC0415

    rc = dict(sns.axes_style(style))
    rc.update(sns.plotting_context(context))
    # seaborn's axes_style names Arial first; keep the pinned font.
    rc.pop("font.sans-serif", None)
    rc.pop("font.family", None)
    return rc


@contextlib.contextmanager
def thesis_style(
    seaborn_style: str | None = "whitegrid", extra: Mapping[str, Any] | None = None
) -> Iterator[None]:
    """Scope every rc change to the ``with`` block (``matplotlib.rc_context``).

    ``seaborn_style=None`` draws on matplotlib's defaults (the EDA notebooks
    never called ``sns.set_theme``).
    """
    import matplotlib as mpl  # noqa: PLC0415

    rc = {**_seaborn_rc(seaborn_style), **BASE_RC, **dict(extra or {})}
    with mpl.rc_context(rc):
        yield


def save_svg(fig: Any, path: str | Path, *, bbox_inches: str | None = "tight") -> Path:
    """Write ``fig`` as a byte-reproducible SVG and close it."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches=bbox_inches, metadata={"Date": None})
    plt.close(fig)
    text = path.read_text(encoding="utf-8")
    stripped = _DC_DATE.sub("", text)
    if stripped != text:
        path.write_text(stripped, encoding="utf-8")
    return path
