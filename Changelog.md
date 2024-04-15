
0.x.x
=====

0.3.0
-----
- Printing/rendering now respects terminal size 
- added multiindex option to `to_pandas` and `flatten`
- added keyword argument `max_colwidth` to `Formatter`

0.2.1
-----

- renamed `DictOfPandas.to_dataframe` to `DictOfPandas.to_pandas`
- `SliceDict`, `TypedSliceDict` and `DictOfPandas` do **not** inherit from `dict` anymore

0.1.3
-----

- rendering (`__str__` and `__repr__`) fixes