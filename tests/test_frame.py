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
        ([T,F], ['a']),
        (pd.Series([T,F], index=[10,20]), ['a'])
    ],
)
def test__getitem__(klass, indexer, expected_keys):
    instance = klass(a=pd.Series([0.]), b=pd.Series([1.]))
    result = instance[indexer]
    assert result.keys() == KeysView(expected_keys)
