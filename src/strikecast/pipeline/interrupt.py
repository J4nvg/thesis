"""SIGTERM / SIGUSR1 handling for cluster jobs (audit 2026-09-26 C21, C22).

SLURM stops a job with SIGTERM (wall clock, ``scancel``, ``scontrol requeue``)
and, when a job is submitted with ``--signal=B:USR1@<s>``, warns the batch
shell with SIGUSR1 first; ``scripts/slurm/job.sbatch`` forwards that warning to
the Python process as SIGTERM. Without a handler CPython dies on SIGTERM with no
Python-level cleanup, so a stage stayed ``running`` in ``state.json`` forever.

:func:`install` gives the two signals a meaning per subcommand:

``run`` / ``featsel`` / ``importance`` / everything else
    raise :class:`StageInterrupted` (a :class:`KeyboardInterrupt`, so the
    ``except Exception`` blocks that turn a model error into ``failed`` do NOT
    swallow it). ``run_stage`` marks its stage ``interrupted``; the CLI's
    top-level handler marks whatever else this process had started
    (:func:`strikecast.store.run_store.interrupt_active_stages`, which covers the
    composite stages too) and exits with :data:`EXIT_INTERRUPTED`.

``tune``
    exit IMMEDIATELY with :data:`EXIT_INTERRUPTED` (``os._exit``). Raising into
    Optuna would mark the running trial ``FAIL``, and a ``FAIL`` counts against
    the study's total budget (``optuna_runner._n_finished``); a trial killed
    outright stays ``RUNNING`` in the SQLite study, is NOT counted, and is simply
    run again after the requeue. The study itself is committed per trial, so
    nothing else is lost.

Only the first signal acts; later ones (SLURM repeats SIGTERM before SIGKILL)
are ignored so the bookkeeping is not interrupted halfway.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Any

__all__ = ["EXIT_INTERRUPTED", "StageInterrupted", "install", "restore"]

logger = logging.getLogger(__name__)

#: Exit status of a job stopped by a signal (neither 0 nor a Python traceback's 1).
EXIT_INTERRUPTED = 99

_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGUSR1") if hasattr(signal, name)
)


class StageInterrupted(KeyboardInterrupt):
    """Raised in the main thread when SLURM (or anyone) asks the job to stop."""

    def __init__(self, signame: str = "SIGTERM") -> None:
        super().__init__(signame)
        self.signame = signame

    def __str__(self) -> str:  # what lands in state.json's `error`
        return f"interrupted by {self.signame}"


_fired = False


def _signame(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:  # pragma: no cover - not a known signal number
        return str(signum)


def _raise_handler(signum: int, _frame: Any) -> None:
    global _fired
    if _fired:
        return
    _fired = True
    name = _signame(signum)
    logger.warning("%s received: stopping and marking the running stage interrupted", name)
    raise StageInterrupted(name)


def _exit_handler(signum: int, _frame: Any) -> None:
    global _fired
    if _fired:
        return
    _fired = True
    name = _signame(signum)
    from strikecast.store.run_store import interrupt_active_stages  # noqa: PLC0415

    interrupt_active_stages(f"interrupted by {name}")
    print(
        f"[strikecast] {name} received during tuning: exiting now; the running trial "
        "stays RUNNING in the study (not counted) and is re-run after the requeue",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()
    os._exit(EXIT_INTERRUPTED)


def install(command: str) -> dict[int, Any]:
    """Install the handlers for one CLI subcommand; returns the previous ones.

    Pass the return value to :func:`restore` when the command is done, so an
    in-process caller (the tests call ``cli.main`` directly) gets its own
    handlers back.
    """
    global _fired
    _fired = False
    handler = _exit_handler if command == "tune" else _raise_handler
    previous: dict[int, Any] = {}
    for sig in _SIGNALS:
        try:
            previous[sig] = signal.signal(sig, handler)
        except ValueError:  # pragma: no cover - not the main thread
            logger.debug("cannot install a %s handler outside the main thread", sig)
    return previous


def restore(previous: dict[int, Any]) -> None:
    """Undo :func:`install`."""
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, TypeError):  # pragma: no cover - not the main thread
            pass
