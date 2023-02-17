#!/usr/bin/env python
from __future__ import annotations

import functools
import logging

import pandas as pd


def log_call(func: callable) -> callable:
    """log the function name on call with level debug."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"{func.__name__} was called")
        return func(*args, **kwargs)

    return wrapper


def index_equals_other(this: pd.Index, other: pd.Index):
    assert isinstance(this, pd.Index)
    return isinstance(other, pd.Index) and this.equals(other)


def series_equals_other(this: pd.Series, other: pd.Series):
    assert isinstance(this, pd.Series)
    return (
        isinstance(other, pd.Series)
        and this.index.equals(other.index)
        and this.equals(other)
    )


def dataframe_equals_other(this: pd.DataFrame, other: pd.DataFrame):
    assert isinstance(this, pd.DataFrame)
    return (
        isinstance(other, pd.DataFrame)
        and this.columns.equals(other.columns)
        and this.index.equals(other.index)
        and this.equals(other)
    )


def pd_obj_equals_other(
    this: pd.DataFrame | pd.Series | pd.Index,
    other: pd.DataFrame | pd.Series | pd.Index,
):
    if isinstance(this, pd.DataFrame):
        return dataframe_equals_other(this, other)
    if isinstance(this, pd.Series):
        return series_equals_other(this, other)
    if isinstance(this, pd.Index):
        return index_equals_other(this, other)
    raise TypeError(f"{type(this)} is not a pandas object")


def other_equals_this(
    left,
    right,
    check_type=True,  # check isinstance(left,right) and vice versa
    check_values=True,  # left.equals(right) or left == right
    check_index=None,  # obj.index
    check_columns=None,  # obj.columns
    check_dtypes=False,  # series.dtype, index.dtype, dataframes.dtype
    check_index_dtype=False,  # obj.index.dtype
    check_columns_dtype=False,  # obj.columns.dtype
    check_names=False,  # index.name or series.name
    check_freq=None,  # obj.index.freq or index.freq, also equal if both are missing
):
    """
    Check if right equals left.

    Try to use ``equal`` method first before fallback to __eq__

    Parameters
    ----------
    left : Any
        Object to compare against.
    right : Any
        Object to check.
    check_type : bool, default True
        Check if ``right`` is also an instance of ``left``'s type
    check_values : bool, default True
        Check if ``right`` equals ``left`` in value(s)
    check_columns : bool or None, default None
        Check if columns are identical. If None, only
        perform check if ``left`` has a column axis.
    check_index : bool or None, default None
        Check if the index is identical. If None, only
        perform check if ``left` has an index.
    check_freq : bool or None, default None
        Check the `freq` attribute on ``left`` (e.g. if of type pandas.Index)
        or on the index of ``left`` (e.g. if of type pandas.Series)
        If None, only perform check if ``left` has `freq` attribute,
        freq on the index is irrelevant here.
    check_dtypes : bool, default False
        Check if the `dtype`/`dtypes` are equal.
    check_index_dtype : bool, default False
        Check if the `dtype` of the index are equal.
        Only relevant the index are checked.
    check_columns_dtype : bool, default False
        Check if the `dtype` of columns are equal.
        Only relevant columns are checked.
    check_names : bool, default False
        Check if the `name` is identical.

    Returns
    -------
    bool
    """
    if check_freq is None and hasattr(left, "freq"):
        check_freq = True
        if check_index is None:
            check_index = False
    if check_index is None and hasattr(left, "index"):
        check_index = True
    if check_columns is None and hasattr(left, "columns"):
        check_columns = True

    def eq(a, b, name):
        if name is not None:
            a = getattr(a, name)
            if hasattr(b, name):
                b = getattr(b, name)
            else:
                return False
        if a is b:
            return True
        if hasattr(a, "equals"):
            return a.equals(b)
        try:
            return bool(a == b)
        except (ValueError, TypeError, NotImplementedError):
            return False

    if left is right:
        return True

    if check_type and not isinstance(right, type(left)):
        return False

    if check_values and not eq(left, right, None):
        return False

    if check_index:  # pd.Series, pd.DataDframe
        if not eq(left, right, "index"):
            return False
        if check_index_dtype:
            if not eq(left.index, right.index, "dtype"):
                return False
        if check_freq:
            freqs = hasattr(left.index, "freq") + hasattr(right.index, "freq")
            if freqs == 1 or freqs == 2 and not eq(left.index, right.index, "freq"):
                return False
    elif check_freq:  # pd.Index
        if not eq(left, right, "freq"):
            return False

    if check_columns:
        if not eq(left, right, "columns"):
            return False
        if check_columns_dtype:
            if not eq(left.columns, right.columns, "dtype"):
                return False

    if check_dtypes:
        for name in ["dtypes", "dtype"]:
            if hasattr(left, name):
                break
        else:
            raise AttributeError(
                "'left' obj has neither a attribute 'dtype' nor 'dtypes', but 'check_dtypes' was True."
            )
        if not eq(left, right, name):
            return False

    if check_names and not eq(left, right, "names"):
        return False
