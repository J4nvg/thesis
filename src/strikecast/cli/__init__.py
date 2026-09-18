"""The command-line interface. Hydra lives here and in ``config/loader.py`` only.

Deliberately empty of imports: ``strikecast.cli.main`` must keep naming the
MODULE, not the function inside it, so ``from strikecast.cli import main`` gives
the module and ``strikecast.cli.main:main`` (the console script) resolves.
"""
