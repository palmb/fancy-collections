#!/usr/bin/env python
from __future__ import annotations
import warnings
from typing import List, ValuesView
import abc
import numpy as np
import functools
import pandas as pd
from sliceable_dict import SliceDict

from .formatting import Formatter
from .lib import VT, KT, PD


class Axis:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance: DictOfPandas | None, owner) -> pd.Index:
        if instance is None:  # class attribute access
            return self  # noqa
        return pd.Index(instance.keys())

    def __set__(self, instance: DictOfPandas, value: VT) -> None:
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
            # We cannot set data directly, because inherit classes
            # may restrict keys by type or value. So we need to use
            # the regular update() method.
            instance.update(new)
            data = instance.data
        except Exception as e:
            raise type(e)(f"Cannot set new {self._name}, because {e}") from None
        finally:
            instance.data = data


class IndexMixin:
    @abc.abstractmethod
    def values(self) -> ValuesView[PD]:
        ...

    def _get_indexes(self) -> List[pd.Index]:
        indexes = []
        for obj in self.values():
            if isinstance(obj, pd.Index):
                index = obj
            else:
                index = obj.index
            indexes.append(index)
        return indexes

    def union_index(self) -> pd.Index:
        return functools.reduce(pd.Index.union, self._get_indexes(), pd.Index([]))

    def shared_index(self) -> pd.Index:
        indexes = self._get_indexes()
        if indexes:
            return functools.reduce(pd.Index.intersection, indexes)
        return pd.Index([])


class DictOfPandas(SliceDict, IndexMixin):
    @property
    def _constructor(self) -> type[DictOfPandas]:
        return DictOfPandas

    columns = Axis("columns")

    @property
    def empty(self) -> bool:
        return len(self.keys()) == 0

    def __setitem_single__(self, key: KT, value: PD) -> None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            return super().__setitem_single__(key, value)
        raise TypeError(
            f"Value must be of type pd.Index, pd.Series or "
            f"pd.DataFrame, not {type(value).__name__}"
        )

    def _uniquify_name(self, name: str) -> str:
        if name not in self.keys():
            return name
        i = 1
        while f"{name}({i})" in self.keys():
            i += 1
        return f"{name}({i})"

    def flatten(self) -> DictOfPandas:
        """
        Promote dataframe columns to first level columns.

        Prepend column names of an inner dataframe with the
        key/column name of the outer frame.

        Examples
        --------
        >>> frame = DictOfPandas(key0=pd.DataFrame(np.arange(4).reshape(2,2), columns=['c0', 'c1']))
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
                    data[self._uniquify_name(f"{key}_{col}")] = ser
            else:
                data[key] = value
        return self.__class__(data)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(dict(self.flatten()))

    def __repr__(self) -> str:
        return repr(dict(self))

    def __str__(self) -> str:
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        return self.to_string(max_rows=max_rows, min_rows=min_rows)

    def to_string(
        self,
        max_rows: int | None = None,
        min_rows: int | None = None,
        show_df_column_names: bool = True,
    ) -> str:
        """
        Render a DictOfPandas to a console-friendly tabular output.

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


class DictOfSeries(DictOfPandas):
    def __setitem_single__(self, key: KT, value: pd.Series):
        if isinstance(value, pd.Series):
            return super().__setitem_single__(key, value)
        raise TypeError(
            f"Values must be of type pandas.Series, not {type(value).__name__}"
        )


class SaqcFrame(DictOfPandas):  # dios replacement
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
