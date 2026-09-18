"""Hydra composition of the ``configs/`` tree into an :class:`ExperimentConfig`.

This is the **only** module in ``strikecast`` that imports Hydra or OmegaConf
(refactor plan sec. 5.6: "Hydra lives only in ``cli/``; the library takes plain
pydantic configs"). Everything downstream receives a validated pydantic object
and never sees a ``DictConfig``.

The pipeline is deliberately three flat steps::

    compose(config_name="experiment/<name>", overrides=[...])   # Hydra
    OmegaConf.to_container(cfg, resolve=True)                   # -> plain dict
    ExperimentConfig(**payload)                                 # -> pydantic

so that a config error is either a Hydra composition error (a group or a key
that does not exist) or a pydantic validation error (a value that is not
allowed), never a silent coercion. Every schema in
:mod:`strikecast.config.schema` sets ``extra="forbid"``, so a typo in a YAML key
fails loudly here rather than at model-build time.

Layout of the tree this composes (sec. 5.1)::

    configs/
      experiment/{count,diff,hurdle,damage}.yaml   primary configs
      backtest/weekly_retrain.yaml                 -> `stages`
      paradigm/{global,activity,local}.yaml        -> `paradigms`
      tracking/{wandb_online,wandb_offline,noop}.yaml
      seeds/default.yaml

Every group file carries ``# @package _global_``, so a group's contents land at
the top level of the composed config under the field name the schema uses
(``stages``, ``paradigms``, ``tracking``, ``seeds``) rather than under the group
name. The experiment file then reads exactly like the pydantic model.

The experiment files carry ``# @package _global_`` as well: a primary config
composed as ``experiment/<name>`` would otherwise be packaged under
``experiment`` because of the directory it lives in, and
:class:`~strikecast.config.schema.ExperimentConfig` would see one unexpected
``experiment`` key instead of its own fields.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from strikecast.config.schema import ExperimentConfig

__all__ = ["CONFIG_DIR_ENV", "default_config_dir", "load_experiment"]

#: Environment variable that overrides the discovered ``configs/`` directory.
#: Useful on the cluster, where the package may be installed outside the repo.
CONFIG_DIR_ENV = "STRIKECAST_CONFIG_DIR"

#: The Hydra group every experiment file lives in.
EXPERIMENT_GROUP = "experiment"


def default_config_dir() -> Path:
    """The repository's ``configs/`` directory.

    Resolution order:

    1. ``$STRIKECAST_CONFIG_DIR`` if set;
    2. ``<repo root>/configs``, derived from this file's location
       (``<repo>/src/strikecast/config/loader.py`` -> three parents up).
    """
    env = os.environ.get(CONFIG_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "configs"


def _resolve(path_or_name: str | Path, config_dir: Path) -> str:
    """Turn ``"count"`` / ``"experiment/count"`` / a YAML path into a config name.

    Hydra's ``config_name`` may contain a group path, so an experiment is always
    composed as ``experiment/<stem>``. A filesystem path is accepted for
    convenience (``strikecast run configs/experiment/diff.yaml``) and reduced to
    its stem; it must live in the composed tree, because Hydra searches
    ``config_dir`` and not the path's own directory.
    """
    text = str(path_or_name)
    stem = Path(text).stem if text.endswith((".yaml", ".yml")) else text
    stem = stem.removeprefix(f"{EXPERIMENT_GROUP}/")
    if not (config_dir / EXPERIMENT_GROUP / f"{stem}.yaml").is_file():
        available = sorted(
            p.stem for p in (config_dir / EXPERIMENT_GROUP).glob("*.yaml")
        )
        raise FileNotFoundError(
            f"no experiment config {stem!r} under {config_dir / EXPERIMENT_GROUP}; "
            f"available: {available}"
        )
    return f"{EXPERIMENT_GROUP}/{stem}"


def load_experiment(
    path_or_name: str | Path,
    overrides: Sequence[str] = (),
    *,
    config_dir: str | Path | None = None,
) -> ExperimentConfig:
    """Compose one experiment and validate it into an :class:`ExperimentConfig`.

    Args:
        path_or_name: ``"count"``, ``"experiment/count"`` or a path to
            ``configs/experiment/count.yaml``.
        overrides: Hydra override strings, exactly as on the command line, e.g.
            ``["models=[lightgbm_poisson]", "seeds.eval_seeds=[1,2]",
            "tracking=noop"]``.
        config_dir: the ``configs/`` tree to compose from; defaults to
            :func:`default_config_dir`.

    Returns:
        A fully validated, resolved experiment configuration.

    Notes:
        Hydra keeps its state in a process-global singleton, so this function
        clears it before and after composing. That makes repeated calls in one
        process (tests, a multirun driver, a notebook) safe, and it is the
        reason ``initialize_config_dir`` is used as a context manager rather
        than at import time.
    """
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.core.global_hydra import GlobalHydra  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    root = Path(config_dir).expanduser().resolve() if config_dir else default_config_dir()
    if not root.is_dir():
        raise FileNotFoundError(
            f"config directory {root} does not exist; set ${CONFIG_DIR_ENV} or pass config_dir="
        )
    config_name = _resolve(path_or_name, root)

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(root)):
            cfg = compose(config_name=config_name, overrides=list(overrides))
        payload = OmegaConf.to_container(cfg, resolve=True)
    finally:
        GlobalHydra.instance().clear()

    if not isinstance(payload, dict):
        raise TypeError(f"composed config for {config_name!r} is not a mapping: {type(payload)}")
    # `hydra` only appears when a job asks for it; strip it so that
    # `extra="forbid"` stays meaningful for everything else.
    payload.pop("hydra", None)
    return ExperimentConfig(**payload)  # type: ignore[arg-type]
