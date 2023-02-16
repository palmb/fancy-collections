#!/usr/bin/env python
import pandas as pd
import pytest
from fancy_collections import DictOfPandas, SliceDict, TypedSliceDict
from typing import KeysView  # do not import from collection, fails for py3.10

T, F = True, False


@pytest.fixture(params=[DictOfPandas, TypedSliceDict, SliceDict])
def klass(request):
    return request.param


@pytest.mark.parametrize(
    "indexer,expected_keys",
    [
        # slices
        (slice(None, 1), ["a"]),
        (slice(1, None), ["b"]),
        (slice(None), ["a", "b"]),
        # list like
        (["b"], ["b"]),
        (["a", "b"], ["a", "b"]),
        (pd.Series(["a"], index=[10]), ["a"]),
        (pd.Series(["a", "b"], index=[10, 20]), ["a", "b"]),
        (pd.Index(["a"]), ["a"]),
        (pd.Index(["a", "b"]), ["a", "b"]),
        # boolean list like
        ([F, F], []),
        ([T, F], ["a"]),
        (pd.Series([F, T], index=[10, 20]), ["b"]),
        (pd.Series([T, T], index=[10, 20]), ["a", "b"]),
    ],
)
def test__getitem__(klass, indexer, expected_keys):
    instance = klass(a=pd.Series([0.0]), b=pd.Series([1.0]))
    result = instance[indexer]
    assert result.keys() == KeysView(expected_keys)
