#!/usr/bin/env python
from __future__ import annotations

import numpy as np

import functools
import warnings
from typing import List, Any

import pandas as pd
from sliceable_dict import TypedSliceDict, SliceDict

import fancy_collections.lib as lib
from fancy_collections.formatting import Formatter


class Axis:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance: DictOfPandas | None, owner) -> pd.Index:
        if instance is None:  # class attribute access
            return self  # noqa
        return pd.Index(instance.keys())

    def __set__(self, instance: DictOfPandas, value: Any) -> None:
        value = pd.Index(value)
        if not value.is_unique:
            raise ValueError(f"{self._name} must not have duplicates.")
        if len(instance.keys()) != len(value):
            raise ValueError(
                f"{self._name} has {len(instance.keys())} elements, "
                f"but {len(value)} values was passed."
            )
        data: dict = instance.data
        try:
            instance.data = {}
            # We cannot set data directly, because inherit classes
            # might restrict keys to some specific types or values,
            # so we use the regular update() method.
            instance.update(zip(value, data.values()))
            data = instance.data
        except Exception as e:
            raise type(e)(f"Cannot set new {self._name}, because {e}") from None
        finally:
            instance.data = data


class IndexMixin:
    def _get_indexes(self: SliceDict) -> List[pd.Index]:
        indexes = []
        for obj in self.values():
            if isinstance(obj, pd.Index):
                index = obj
            else:
                index = obj.index
            indexes.append(index)
        return indexes

    def union_index(self) -> pd.Index:
        indexes = self._get_indexes()
        if indexes:
            return functools.reduce(pd.Index.union, self._get_indexes())
        return pd.Index([])

    def shared_index(self) -> pd.Index:
        indexes = self._get_indexes()
        if indexes:
            return functools.reduce(pd.Index.intersection, indexes)
        return pd.Index([])


class DictOfPandas(TypedSliceDict, IndexMixin):
    # allow any keys, but restrict
    # values to pandas objects
    _key_types = ()
    _value_types = (pd.Series, pd.DataFrame, pd.Index)

    # .columns property
    columns = Axis("columns")

    @property
    def _constructor(self) -> type[DictOfPandas]:
        return type(self)

    @property
    def empty(self) -> bool:
        """
        Indicator whether DictOfPandas is empty.

        True if DictOfPandas is entirely empty (no items) or all
        items are empty themselves.

        Notes
        -----
        To only check if DictOfPandas has no items use ``len`` or ``bool``
        buildins.

        Examples
        --------
        >>> di1 = DictOfPandas()
        >>> di1.empty
        True

        A DictOfPandas is also considered empty if all items within it are empty

        >>> di2 = DictOfPandas(a=pd.Series(dtype=float), b=pd.DataFrame())
        >>> assert di2['a'].empty and di2['b'].empty
        >>> di2.empty
        True

        To differentiate between a DictOfPandas with no items a DictOfSeries
        with empty items use the buildin functions `len` or `bool`

        >>> len(di1)
        0
        >>> bool(di1)
        False
        >>> len(di2)
        2
        >>> bool(di2)
        True

        Returns
        -------
        bool
        """
        return len(self) == 0 or all(o.empty for o in self.values())

    def _uniquify_name(self, name: str) -> str:
        if name not in self.keys():
            return name
        i = 1
        self.union_index()
        while f"{name}({i})" in self.keys():
            i += 1
        return f"{name}({i})"

    def flatten(self, promote_index: bool = False, multiindex: bool = False) -> DictOfPandas:
        """
        Promote dataframe columns to first level columns.

        Prepend column names of an inner dataframe with the
        key/column name of the outer frame.

        Parameters
        ----------
        promote_index : bool, default False
            Makes `pandas.Series` from items of type `pandas.Index` if True.
            Every item of the resulting DictOfPandas be a series then.
        multiindex: bool, default False
            If True, the result column names will be pd.MultiIndex like, if False, unique
            column names will be generated from the different levels of column indices.

        Returns
        -------
        DictOfPandas

        Examples
        --------
        >>> frame = DictOfPandas(key0=pd.DataFrame({'c0': [1, 1], "c1": [2, 2]}))
        >>> frame   # doctest: +NORMALIZE_WHITESPACE
             key0 |
        ========= |
           c0  c1 |
        0   1   2 |
        1   1   2 |

        >>> frame.flatten()   # doctest: +NORMALIZE_WHITESPACE
        key0_c0 | key0_c1 |
        ======= | ======= |
        0     1 | 0     2 |
        1     1 | 1     2 |

        >>> frame.flatten(multiindex=True)   # doctest: +NORMALIZE_WHITESPACE
        ('key0', 'c0') | ('key0', 'c1') |
        ============== | ============== |
             0  1      |      0  2      |
             1  1      |      1  2      |
        """
        data = dict()
        for key, value in self.items():
            if isinstance(value, pd.DataFrame):
                for col, ser in dict(value).items():
                    if multiindex:
                        data[(key, col)] = ser
                    else:
                        data[self._uniquify_name(f"{key}_{col}")] = ser
            elif promote_index and isinstance(value, pd.Index):
                data[key] = value.to_series()
            else:
                data[key] = value
        return self.__class__(data)

    def to_dataframe(self, how="outer") -> pd.DataFrame:
        """
        Transform DictOfPandas to a pandas.DataFrame.

        .. deprecated:: 0.2.0
           use `DictOfPandas.to_pandas()` instead.
        """
        warnings.warn(
            f"`to_dataframe()` is deprecated use `to_pandas()` instead.",
            category=DeprecationWarning,
        )
        return self.to_pandas(how)

    def to_pandas(self, how="outer", fill_value=np.nan, multiindex=False) -> pd.DataFrame:
        """
        Transform DictOfPandas to a pandas.DataFrame.

        Because a pandas.DataFrame can not handle data of different
        length, but DictOfPandas can, the missing data is filled with
        NaNs or is dropped, depending on the keyword `how`.

        Items of type `pandas.Index` will become `pandas.Series` with
        identical index and values as the original Index.

        For items of type `pandas.Dataframe` each column will become an own
        column in the resulting frame (for more detail see ``flatten``)

        Parameters
        ----------
        how : {'outer', 'inner'}, default 'outer'
            Defines how the resulting DataFrame index is generated.

            - ``outer`` : The resulting DataFrame index is the combination
                of all indices merged together. If a column misses values at
                new index locations, `NaN`'s are filled.
            - ``inner`` : Only indices that are present in all columns are used
                for the resulting index. Filling logic is not needed, but values
                are dropped, if a column has indices that are not known to all
                other columns.
        fill_value:
            Value to use for missing values. Defaults to NaN, but can be any “compatible” value.
        multiindex: bool, default True
            If True, the result column names will be pd.MultiIndex like, if False, unique
            column names will be generated from the different levels of column indices

        Returns
        -------
        frame: pandas.DataFrame

        See Also
        --------
        DictOfPandas.flatten:  Make series from DataFrame items in a DictOfPandas

        Examples
        --------
        Missing data locations are filled with NaN's

        >>> from fancy_collections import DictOfPandas
        >>> a = pd.Series(11, index=range(2))
        >>> b = pd.Series(22, index=range(3))
        >>> c = pd.Series(33, index=range(1,9,3))
        >>> di = DictOfPandas(a=a, b=b, c=c)
        >>> di   # doctest: +NORMALIZE_WHITESPACE
            a |     b |     c |
        ===== | ===== | ===== |
        0  11 | 0  22 | 1  33 |
        1  11 | 1  22 | 4  33 |
              | 2  22 | 7  33 |

        >>> di.to_pandas()   # doctest: +NORMALIZE_WHITESPACE
              a     b     c
        0  11.0  22.0   NaN
        1  11.0  22.0  33.0
        2   NaN  22.0   NaN
        4   NaN   NaN  33.0
        7   NaN   NaN  33.0

        or is dropped if `how='inner'`

        >>> di.to_pandas(how='inner')   # doctest: +NORMALIZE_WHITESPACE
                a   b   c
            1  11  22  33

        todo: examples with dataframe
        """
        if how not in ["inner", "outer"]:
            raise ValueError("`how` must be one of 'inner' or 'outer'")
        flat = dict(self.flatten(promote_index=True, multiindex=multiindex))
        if how == "outer":
            target_index = self.union_index()
        else: # how == "inner"
            target_index = self.shared_index()
        df = pd.DataFrame({k: v.reindex(target_index, fill_value=fill_value) for k, v in flat.items()})
        return df

    def __repr__(self) -> str:
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

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, (dict, SliceDict)):
            return False
        if self.keys() != other.keys():
            return False
        # equivalent code:
        # for key in self.keys():
        #     if not _pandasObjEqual(self[key], other[key]):
        #         return False
        # return True
        values = zip(self.values(), other.values())
        eq = lib.pd_obj_equals_other
        return next(filter(lambda e: not eq(*e), values), True) is True
