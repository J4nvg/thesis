"""Loading of the fixed reference files and the master timeseries parquet.

Behaviour-preserving port of ``src/prevalent_functions.py::load_data`` plus
``src/dataset_specific_tools.py::load_actors_dict``. The only difference is
that the legacy ``print`` calls became ``logging`` calls; nothing else changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ._legacy_helpers import (
    construct_path,
    get_list_from_file,
    load_actors_dict,
    load_region_activity_dictionary,
)

logger = logging.getLogger(__name__)

__all__ = ["Inputs", "load_inputs"]


@dataclass(frozen=True)
class Inputs:
    """Everything the panel builder needs, loaded once.

    Attributes
    ----------
    regions:
        Region slugs, lower-cased, read from ``<fixed_dir>/regions.txt``.
    master:
        The wide master timeseries (``<dataset_dir>/master_combined_timeseries.parquet``).
        Its index is a non-default integer ``Index`` with no name, so
        ``master.reset_index()`` produces a column literally called ``"index"``.
        The legacy pipeline relies on this (see :mod:`strikecast.data.panel`).
    activity_by_region:
        ``{region: tier}`` where tier is ``little=0, low=1, medium=2, high=3``.
        Legacy name: ``regions_activity``.
    actors:
        The raw ``actors.json`` payload used by the GDELT aggregation.
    """

    regions: list[str]
    master: pd.DataFrame
    activity_by_region: dict[str, int]
    actors: dict = field(repr=False)


def load_inputs(fixed_dir: str | Path, dataset_dir: str | Path) -> Inputs:
    """Load regions, the master parquet, the activity tiers and the actors dict.

    Parameters
    ----------
    fixed_dir:
        Directory holding ``regions.txt``, ``regions_activity_cat.json`` and
        ``actors.json`` (legacy ``FIXED_DATA_PATH`` / ``data_path``).
    dataset_dir:
        Directory holding ``master_combined_timeseries.parquet`` (legacy
        ``DATASET_PATH`` / ``dataset_path``).
    """
    fixed_dir = str(fixed_dir)
    dataset_dir = str(dataset_dir)

    regions = get_list_from_file(construct_path(fixed_dir, "regions.txt"))
    master = pd.read_parquet(construct_path(dataset_dir, "master_combined_timeseries.parquet"))
    activity_by_region = load_region_activity_dictionary(
        construct_path(fixed_dir, "regions_activity_cat.json")
    )
    actors = load_actors_dict(construct_path(fixed_dir, "actors.json"))

    logger.info("master_timeseries shape: %s", master.shape)
    logger.info("regions: %s  -  %s ...", len(regions), regions[:5])
    logger.info(
        "start: %s end: %s", master["event_date"].min(), master["event_date"].max()
    )
    return Inputs(
        regions=regions,
        master=master,
        activity_by_region=activity_by_region,
        actors=actors,
    )
