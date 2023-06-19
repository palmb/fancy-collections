#!/usr/bin/env python
from __future__ import annotations
import pickle

import pandas as pd
import pytest
from fancy_collections import SliceDict, TypedSliceDict, DictOfPandas


@pytest.fixture(params=[DictOfPandas, TypedSliceDict, SliceDict])
def klass(request):
    return request.param


@pytest.mark.parametrize(
    "kwargs", [dict(), dict(a=pd.Series(range(10))), dict(a=pd.Series([]))]
)
def test_pickling(klass, kwargs):
    inst = klass(**kwargs)
    result = pickle.loads(pickle.dumps(inst))
    assert isinstance(result, SliceDict)
    assert isinstance(result, klass)
    assert inst.keys() == result.keys()
    for k in inst.keys():
        assert result[k].equals(inst[k])
        assert result[k].index.equals(inst[k].index)
