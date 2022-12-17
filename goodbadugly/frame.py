#!/usr/bin/env python

from __future__ import annotations

import warnings

from typing import List, ValuesView, Iterable, Tuple, Any, Hashable

import abc
import numpy as np
import functools

import pandas as pd
from sliceable_dict import SliceDict
from .formatting import Formatter


class _Axis:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance: Frame | None, owner) -> pd.Index:
        if instance is None:  # class attribute access
            return self  # noqa
        return pd.Index(instance.keys())

    def __set__(self, instance: Frame, value: Any) -> None:
        value = pd.Index(value)
        if not value.is_unique:
            raise ValueError(f"{self._name} must not have duplicates.")
        if len(instance.keys()) != len(value):
            raise ValueError(
                f"Length mismatch: Expected {self._name} have "
                f"{len(instance.keys())} elements, new values "
                f"have {len(value)} elements."
            )
        # We must expand the zip now, because values()
        # are a view and would be empty after clear().
        data = dict(instance.data)  # shallow copy
        new = dict(zip(value, instance.data.values()))
        try:
            instance.data = {}
            # We don't set data directly, because inherit
            # classes may restrict key-types or some key-values,
            # so we use the regular __setitem__() via update().
            instance.update(new)
        except Exception as e:
            instance.data = data
            raise type(e)(f"setting new {self._name} failed, because {e}") from None


class IndexMixin:
    @abc.abstractmethod
    def values(self) -> ValuesView[pd.Series | pd.DataFrame]:
        ...

    def _get_indexes(self):
        indexes = []
        for obj in self.values():
            indexes.append(obj.index)
        return indexes

    def union_index(self):
        return functools.reduce(pd.Index.union, self._get_indexes(), pd.Index([]))

    def shared_index(self):
        indexes = self._get_indexes()
        if indexes:
            return functools.reduce(pd.Index.intersection, indexes)
        return pd.Index([])


class Frame(SliceDict, IndexMixin):
    @property
    def _constructor(self) -> type[Frame]:
        return Frame

    columns = _Axis("columns")

    @property
    def empty(self):
        return len(self.keys()) == 0

    def _set_single_item_callback(self, key, value):
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"Value must be of type pd.Series or "
                f"pd.DataFrame, not {type(value)}"
            )
        return super()._set_single_item_callback(key, value)

    def _uniquify_key(self, name):
        if name not in self.keys():
            return name
        i = 1
        while f"{name}({i})" in self.keys():
            i += 1
        return f"{name}({i})"

    def flatten(self):
        """
        Promote dataframe columns to first level columns.

        Prepend column names of an inner dataframe with the
        key/column name of the outer frame.

        Examples
        --------
        >>> frame = Frame(key0=pd.DataFrame(np.arange(4).reshape(2,2), columns=['c0', 'c1']))
        >>> frame
             key0 |
        ========= |
           c0  c1 |
        0   0   1 |
        1   2   3 |

        >>> frame.flatten()
           key0_c0 |    key0_c1 |
        ========== | ========== |
        0        0 | 0        1 |
        1        2 | 1        3 |
        """
        data = dict()
        for key, value in self.items():
            if isinstance(value, pd.DataFrame):
                for col, ser in dict(value).items():
                    data[self._uniquify_key(f"{key}_{col}")] = ser
            else:
                data[key] = value
        return self.__class__(data)

    def to_dataframe(self):
        return pd.DataFrame(dict(self.flatten()))

    def __repr__(self):
        return repr(dict(self))

    def __str__(self):
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        return self.to_string(max_rows=max_rows, min_rows=min_rows)

    def to_string(
            self,
            max_rows: int | None = None,
            min_rows: int | None = None,
            show_df_column_names: bool = True,
    ):
        """
        Render a Frame to a console-friendly tabular output.

        Parameters
        ----------
        max_rows : int, optional
            Maximum number of rows to display in the console.

        min_rows : int, optional
            The number of rows to display in the console in a
            truncated repr (when number of rows is above max_rows).

        show_df_column_names : bool, default True
            Prints column names of dataframes if True,
            otherwise colum names are hidden.

        Returns
        -------
        str
            Returns the result as a string.
        """
        return Formatter(self, max_rows, min_rows, show_df_column_names).to_string()


class SaqcFrame(Frame):  # dios replacement
    def _set_single_item_callback(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Keys must be of type string, not {type(key).__name__}")
        if isinstance(value, list):
            value = pd.Series(value)
        # if isinstance(value, pd.DataFrame):
        #     raise TypeError('saqc cant handle dataframes')

        return super()._set_single_item_callback(key, value)

    def to_df(self):
        warnings.warn(
            "to_df() is deprecated, use " "to_dataframe() instead.", DeprecationWarning
        )
        return self.to_dataframe()
