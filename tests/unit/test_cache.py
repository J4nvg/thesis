"""Unit tests for the content-hash cache helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from strikecast.data.cache import cached_parquet, content_hash


def test_hash_is_stable_and_order_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 0.5, 0.5]})
    assert content_hash(df) == content_hash(df.copy())
    assert content_hash(df, {"k": 1}) == content_hash(df, {"k": 1})
    assert content_hash(df, {"k": 1}) != content_hash({"k": 1}, df)


def test_hash_changes_with_values_columns_and_dtypes():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert content_hash(df) != content_hash(pd.DataFrame({"a": [1, 2, 4]}))
    assert content_hash(df) != content_hash(pd.DataFrame({"b": [1, 2, 3]}))
    assert content_hash(df) != content_hash(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))


def test_hash_ignores_set_and_dict_ordering():
    assert content_hash({"a", "b"}) == content_hash({"b", "a"})
    assert content_hash({"x": 1, "y": 2}) == content_hash({"y": 2, "x": 1})
    assert content_hash(["x", "y"]) != content_hash(["y", "x"])


def test_cached_parquet_builds_once(tmp_path):
    path = tmp_path / "nested" / "panel.parquet"
    calls = []

    def builder():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2, 3]})

    first = cached_parquet(path, builder)
    second = cached_parquet(path, builder)

    assert calls == [1]
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))
    pd.testing.assert_frame_equal(first, second)


def test_cached_parquet_propagates_builder_errors(tmp_path):
    path = tmp_path / "panel.parquet"

    def builder():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cached_parquet(path, builder)
    assert not path.exists()
