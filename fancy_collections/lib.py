#!/usr/bin/env python
from __future__ import annotations

import functools
import logging

import pandas as pd


def log_call(level="DEBUG"):
    level = level if isinstance(level, int) else logging.getLevelName(level)
    assert isinstance(level, int)

    def logit(func: callable) -> callable:
        """log the function name on call."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logging.log(level, f"{func.__name__} was called")
            return func(*args, **kwargs)

        return wrapper

    return logit


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
    this,
    other,
    check_type=True,  # check isinstance(this,other) and vice versa
    check_values=True,  # this.equals(other) or this == other
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
    this : Any
        Object to compare against.
    other : Any
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
    if check_freq is None and hasattr(this, "freq"):
        check_freq = True
        if check_index is None:
            check_index = False
    if check_index is None and hasattr(this, "index"):
        check_index = True
    if check_columns is None and hasattr(this, "columns"):
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

    if this is other:
        return True

    if check_type and not isinstance(other, type(this)):
        return False

    if check_values and not eq(this, other, None):
        return False

    if check_index:  # pd.Series, pd.DataDframe
        if not eq(this, other, "index"):
            return False
        if check_index_dtype:
            if not eq(this.index, other.index, "dtype"):
                return False
        if check_freq:
            freqs = hasattr(this.index, "freq") + hasattr(other.index, "freq")
            if freqs == 1 or freqs == 2 and not eq(this.index, other.index, "freq"):
                return False
    elif check_freq:  # pd.Index
        if not eq(this, other, "freq"):
            return False

    if check_columns:
        if not eq(this, other, "columns"):
            return False
        if check_columns_dtype:
            if not eq(this.columns, other.columns, "dtype"):
                return False

    if check_dtypes:
        for name in ["dtypes", "dtype"]:
            if hasattr(this, name):
                break
        else:
            raise AttributeError(
                "'this' obj has neither a attribute 'dtype' nor 'dtypes', but 'check_dtypes' was True."
            )
        if not eq(this, other, name):
            return False

    if check_names and not eq(this, other, "names"):
        return False
