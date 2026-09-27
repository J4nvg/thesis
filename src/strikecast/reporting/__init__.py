"""Every thesis figure and table, generated (audit 2026-09-26 WP4, C1/C2/C5, D10).

``strikecast figures`` builds the DATA and RESULTS items of the thesis inventory
(audit C Part 1) into one folder: SVG figures, ``.tex`` table fragments that
``main.tex`` can ``\\input``, a CSV next to every table and a manifest.

* :mod:`.sources` -- two back-ends behind one interface: the run store
  (:class:`~.sources.StoreSource`) and the stored thesis outputs in ``results/``
  (:class:`~.sources.LegacySource`, read the way ``results/analyse_results.ipynb``
  reads them; the verification path);
* :mod:`.compute` -- the analysis-notebook computations as pure functions;
* :mod:`.plots` / :mod:`.eda` -- the figures (results / raw data);
* :mod:`.style` -- deterministic, scoped matplotlib styling;
* :mod:`.build` -- the item registry, the runner and the manifest.
"""

__all__: list[str] = []
