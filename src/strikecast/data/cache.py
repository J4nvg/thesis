"""Tiny content-hash cache: parquet on disk, never pickle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object as _hash_pandas_object

# pandas re-exports this through a lazy module shim that type checkers cannot
# resolve; the call itself is the documented public API.
hash_pandas_object: Any = _hash_pandas_object

__all__ = ["cached_parquet", "content_hash"]


def _hash_values(obj: pd.DataFrame | pd.Series) -> bytes:
    # `index=True` is hash_pandas_object's default: the index is part of the hash.
    return np.asarray(hash_pandas_object(obj)).tobytes()


def _stable_bytes(obj: Any) -> bytes:
    """Deterministic byte representation of one object."""
    if isinstance(obj, pd.DataFrame):
        parts = [
            json.dumps([str(c) for c in obj.columns]).encode(),
            json.dumps([str(d) for d in obj.dtypes]).encode(),
            _hash_values(obj),
        ]
        return b"|df|".join(parts)
    if isinstance(obj, pd.Series):
        return b"|s|".join([str(obj.name).encode(), str(obj.dtype).encode(), _hash_values(obj)])
    if isinstance(obj, pd.Index):
        return b"|i|" + _hash_values(pd.Series(list(obj), dtype=object))
    if isinstance(obj, (set, frozenset)):
        return b"|set|" + json.dumps(sorted(map(str, obj))).encode()
    if isinstance(obj, bytes):
        return b"|b|" + obj
    if isinstance(obj, Path):
        return b"|p|" + str(obj).encode()
    try:
        return b"|j|" + json.dumps(obj, sort_keys=True, default=str).encode()
    except TypeError:
        return b"|r|" + repr(obj).encode()


def content_hash(*objs: Any) -> str:
    """sha256 over a stable representation of ``objs``.

    Dataframes and series hash through ``pd.util.hash_pandas_object`` (values
    plus index) together with their column names and dtypes; dicts, lists and
    scalars go through canonical JSON; sets are sorted first.
    """
    h = hashlib.sha256()
    for obj in objs:
        h.update(_stable_bytes(obj))
        h.update(b"\x00")
    return h.hexdigest()


def cached_parquet(path: str | Path, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return ``path`` if it exists, otherwise build it, write it atomically and return it."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    df = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(path)
    return df
