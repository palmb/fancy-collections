#!/usr/bin/env python
import numpy as np
import pytest
import pandas as pd
from fancy_collections.core import Axis
from sliceable_dict import SliceDict

T, F = True, False


class IndexContainer(SliceDict):
    index = Axis("index")


class ColumnContainer(SliceDict):
    columns = Axis("columns")


@pytest.mark.parametrize("klass", [ColumnContainer, IndexContainer])
@pytest.mark.parametrize("kwargs", [dict(), dict(a=99), dict(x=99)])
@pytest.mark.parametrize("args", [(), ([[1, 2]],), (dict(a="a", b="b"),)])
def test_creation(klass, args, kwargs):
    inst = klass(*args, **kwargs)
    assert isinstance(inst, klass)
    assert isinstance(inst, SliceDict)
    assert isinstance(inst, dict)


@pytest.mark.parametrize("klass", [ColumnContainer, IndexContainer])
@pytest.mark.parametrize("attr", dir(dict))
def test_attrs(klass, attr):
    assert hasattr(klass, attr)
    if issubclass(klass, ColumnContainer):
        assert hasattr(klass, "columns")
    if issubclass(klass, IndexContainer):
        assert hasattr(klass, "index")


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
@pytest.mark.parametrize("key", [None, 1, 1.0, "a", b"a", np.nan])
def test_axis(klass, axis_name, key):
    inst = klass(zzz=None)
    assert getattr(inst, axis_name).equals(pd.Index(["zzz"]))
    inst[key] = None
    assert getattr(inst, axis_name).equals(pd.Index(["zzz", key]))
    del inst[key]
    assert getattr(inst, axis_name).equals(pd.Index(["zzz"]))


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
@pytest.mark.parametrize(
    "setter_name,args,expected",
    [
        # `{"a": None}` is in the container per default
        ("__copy__", (), pd.Index(["a"])),
        ("__or__", ({"b": 1},), pd.Index(["a", "b"])),
        ("__ror__", ({"b": 1},), pd.Index(["b", "a"])),
        ("__ior__", ({"b": 1},), pd.Index(["a", "b"])),  # works inplace and return self
        ("fromkeys", (["b"],), pd.Index(["b"])),
        ("copy", (), pd.Index(["a"])),
    ],
)
def test_index_update__methods_with_result(
        klass, axis_name, setter_name, args, expected
):
    inst = klass(a=None)
    result = getattr(inst, setter_name)(*args)
    assert isinstance(result, SliceDict)
    assert isinstance(result, klass)
    assert getattr(result, axis_name).equals(expected)


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
@pytest.mark.parametrize(
    "setter_name,args,expected",
    [
        # `{"a": None}` is in the container per default
        ("__setitem__", ("b", 1), pd.Index(["a", "b"])),
        ("__delitem__", ("a",), pd.Index([])),
        ("__ior__", ({"b": 1},), pd.Index(["a", "b"])),  # works inplace and return self
        ("setdefault", ("b", 1), pd.Index(["a", "b"])),
        ("setdefault", ("a", 1), pd.Index(["a"])),
        ("pop", ("a",), pd.Index([])),
        ("pop", ("b", None), pd.Index(["a"])),
        ("popitem", (), pd.Index([])),
        ("update", ({"b": 1},), pd.Index(["a", "b"])),
        ("clear", (), pd.Index([])),
    ],
)
def test_index_update__inplace_methods(klass, axis_name, setter_name, args, expected):
    inst = klass(a=None)
    getattr(inst, setter_name)(*args)
    assert getattr(inst, axis_name).equals(expected)


@pytest.mark.parametrize(
    "klass,axis_name", [(ColumnContainer, "columns"), (IndexContainer, "index")]
)
def test_index_setter(klass, axis_name):
    inst = klass(a=10, b=20, c=30)
    setattr(inst, axis_name, [1, 2, 3])
    assert inst.keys() == dict.fromkeys([1, 2, 3]).keys()
    with pytest.raises(ValueError):
        setattr(inst, axis_name, [1, 2])


def test_set_index_fails_and_keep_old_keys():
    class RestrictedChild(ColumnContainer):
        def __setitem_single__(self, key, value):
            if not isinstance(key, str):
                raise TypeError()
            super().__setitem_single__(key, value)

    rc = RestrictedChild(a=10, b=20, c=30)
    with pytest.raises(TypeError):
        rc.columns = ['x', 'y', 9999]
    assert rc.keys() == dict.fromkeys(['a', 'b', 'c']).keys()
