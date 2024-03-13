#!/usr/bin/env python
from __future__ import annotations

import shutil
from typing import Tuple, List, Iterable, Any

from matplotlib import shutil
import pandas as pd


class Formatter:
    column_seperator = " | "
    header_seperator = "="

    def __init__(
        self,
        obj,
        max_rows: int | None = None,
        min_rows: int | None = None,
        show_df_column_names: bool = True,
    ):
        self.show_df_column_names = show_df_column_names
        self._trunc_options = dict(max_rows=max_rows, min_rows=min_rows)
        self._obj = obj
        self.__to_render = []

    def key_to_string(self, key: Any, obj: Any) -> str:  # noqa
        """
        stringify key.

        Parameters
        ----------
        key : Any
            The key to stringify.

        obj : Any
            The object associated with key. It is passed to allow to modify
            key by the value of object. E.g. to set a marker if the object
            is of a special type. The object must not be modified.

        Returns
        -------
        key: str
        """
        return str(key)

    def to_string(self) -> str:
        if len(self._obj) == 0:
            return self._stringify_empty_class(self._obj)

        objects = {}
        widths = {}
        for key, val in self._obj.items():
            key = self.key_to_string(key, val)
            lines = self.stringify(val).splitlines()
            objects[key] = lines
            widths[key] = self._get_maxlen(lines + [key])

        display_width = int(shutil.get_terminal_size().columns * 0.8)

        keys = tuple(widths.keys())
        front_keys, back_keys = set(), set()
        line_length = 0
        for fkey, bkey in zip(keys, keys[::-1]):
            line_length += widths[fkey]
            if line_length < display_width:
                front_keys.add(fkey)
            line_length += widths[bkey]
            if line_length < display_width:
                back_keys.add(bkey)

        for k, v in objects.items():
            if k in front_keys:
                self._add(k, v)

        if not front_keys.intersection(back_keys) and front_keys.union(back_keys) != set(keys):
            self._add("...", ["..."])

        for k, v in objects.items():
            if k in back_keys and k not in front_keys:
                self._add(k, objects[k])

        return self._render()

    def stringify(self, obj: Any) -> str:
        if isinstance(obj, pd.Index):
            return self._stringify_Index(obj)
        if isinstance(obj, pd.Series):
            return self._stringify_Series(obj)
        if isinstance(obj, pd.DataFrame):
            return self._stringify_DataFrame(obj)
        return str(obj)

    def _stringify_DataFrame(self, df: pd.DataFrame) -> str:
        # if df.empty and not df.index.empty:
        #     return f"Empty DataFrame\n" + self._stringify_Index(df.index)

        if df.empty:
            r, c = df.shape
            return self._justify(
                f"{self._stringify_empty_class(df)}\n"  # prevent black formatting
                f" rows:    {r}\n"
                f" columns: {c}\n"
            )
        return df.to_string(
            **self._trunc_options, header=self.show_df_column_names, index_names=False
        )

    def _stringify_Series(self, s: pd.Series) -> str:
        if s.empty:
            return self._justify(
                f"{self._stringify_empty_class(s)}\n"  # prevent black formatting
                f" rows: {len(s)}\n"
            )
        # We use to_frame because it uses less space between index and values
        # return s.to_string(**self._trunc_options, header=False)
        return s.to_frame(name=" ").to_string(
            **self._trunc_options, header=False, index_names=False
        )

    def _stringify_Index(self, idx: pd.Index) -> str:
        if idx.empty:
            return f"{self._stringify_empty_class(idx)}"
        idx = pd.Series("", index=idx, dtype=str)
        return idx.to_string(**self._trunc_options, header=False)

    def _render(self) -> str:
        string = self.__make_header()
        string += self.__make_seperator_row()
        while True:
            row = self.__make_body_row()
            if row is None:
                break
            string += row
        return string

    def __make_header(self) -> str:
        line = ""
        for key, _, n in self.__to_render:
            line += key.rjust(n) + self.column_seperator
        return line + "\n"

    def __make_seperator_row(self) -> str:
        line = ""
        for _, _, n in self.__to_render:
            line += self.header_seperator * n + self.column_seperator
        return line + "\n"

    def __make_body_row(self) -> str | None:
        line = ""
        count = 0
        for _, gen, _ in self.__to_render:
            empty, s = next(gen)  # see Formatter.__iter4ever()
            line += s + self.column_seperator
            count += empty
        if count == len(self.__to_render):
            # all generators are exhausted
            return None
        return line + "\n"

    def _add(self, key: str, lines: List[str]) -> None:
        n = self._get_maxlen(lines + [key])
        gen = self.__iter4ever(lines, n)
        self.__to_render.append((key, gen, n))

    @staticmethod
    def __iter4ever(lines: List[str], n: int) -> Tuple[bool, str]:
        for line in lines:
            yield False, line.center(n)
        while True:
            yield True, " " * n

    @staticmethod
    def _stringify_empty_class(obj) -> str:
        return f"Empty {obj.__class__.__name__}"

    @staticmethod
    def _get_maxlen(obj: Iterable) -> int:
        return max(map(len, obj))

    @staticmethod
    def _justify(
        str_or_lines: str | List[str], width: int | None = None, site="left"
    ) -> str | List[str]:
        """
        Justify a string containing newlines or a list of strings.

        Parameters
        ----------
        str_or_lines : str or list of str
            If a list of strings is given, all string are justified
            by `width` and a list is returned.
            If a string is given, all substring (split by the newline
            character, `\n`) are justified by `width` and a string is
            returned.

        width : int or None, default None
            The width to justify each (sub)string with. If `None`
            width is the length of the longest string.

        site : 'left' or 'right', default 'left'
            Site to justify on.

        Returns
        -------
        justified: str or list of str
        """
        if isinstance(str_or_lines, str):
            lines = str_or_lines.splitlines()
        else:
            lines = str_or_lines
        if width is None:
            width = Formatter._get_maxlen(lines)
        just = str.ljust if site == "left" else str.rjust
        lines = [just(s, width) for s in lines]  # noqa
        if isinstance(str_or_lines, str):
            return "\n".join(lines)
        return lines
