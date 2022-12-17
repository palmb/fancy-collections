#!/usr/bin/env python
import pandas as pd
import pytest
from fancy_collections import DictOfPandas

# pytestmark = pytest.mark.skip


@pytest.mark.parametrize(
    "data",
    [
        None,
        dict(),
        [],
        dict(a=pd.Series(dtype=float), b=pd.DataFrame()),
        [pd.Series(dtype=float), pd.DataFrame()],
    ],
)
@pytest.mark.parametrize("index", [None, [1, 2, 3], pd.DatetimeIndex([0, 5])])
@pytest.mark.parametrize("dtype", [None, int, "float", object, "O"])
@pytest.mark.parametrize("copy", [None, False, True])
def test_constructor(data, index, dtype, copy):
    frame = DictOfPandas(data=data, columns=None, index=index, dtype=dtype, copy=copy)
    for c in frame:
        assert isinstance(frame[c], (pd.Series, pd.DataFrame))


@pytest.mark.parametrize(
    "data,columns,expected",
    [
        # empty DictOfPandas
        (None, None, pd.Index([])),
        ([], None, pd.Index([])),
        (dict(), None, pd.Index([])),
        #
        # create empty columns
        (None, ["a", "b"], pd.Index(["a", "b"])),
        ([], ["a", "b"], pd.Index(["a", "b"])),
        (dict(), ["a", "b"], pd.Index(["a", "b"])),
        #
        # given columns ignores keys of data
        (
            dict(
                a=[],
            ),
            ["a", "b"],
            pd.Index(["a", "b"]),
        ),
        (dict(a=[], b=[]), ["a", "b"], pd.Index(["a", "b"])),
        (dict(a=[], b=[], c=[]), ["a", "b"], pd.Index(["a", "b"])),
        (dict(c=[]), ["a", "b"], pd.Index(["a", "b"])),
        #
        # columns=None takes keys from data
        (dict(a=[], b=[]), None, pd.Index(["a", "b"])),
    ],
)
def test_constructor_columns(data, columns, expected):
    frame = DictOfPandas(data=data, columns=columns)
    assert frame.columns.equals(expected)


#
# @pytest.mark.parametrize(
#     "data,index",
#     [
#         (None,None)
#         (dict(),None)
#         (dict(a=pd.Series(dtype=float)),None)
#         (dict(a=pd.DataFrame()),None)
#         (dict(a=pd.Series(dtype=float), b=pd.DataFrame()),None)
#     ],
# )
# def test_constructor_index(data, index):
#     frame = DictOfPandas(data=data, index=index)
