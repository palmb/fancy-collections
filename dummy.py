#!/usr/bin/env python

import pandas as pd
from fancy_collections import DictOfPandas

if __name__ == "__main__":
    class Fail:
        pass
        # def __repr__(self):
        #     raise NotImplementedError
        # def __str__(self):
        #     raise NotImplementedError
    fail = pd.Series(Fail(), dtype=object)
    x = fail.to_string()
    df=pd.DataFrame(columns=list('abc'), index=range(7))
    df.iloc[2,2] = 'fooo'
    print(x)
    f = DictOfPandas(
        df_empty=pd.DataFrame(),
        df_empty_w_index=pd.DataFrame(index=range(400)),
        df_empty_w_columns=pd.DataFrame(columns=['a', 'b']),
        s_empty=pd.Series(dtype=float),
        s2=pd.Series(range(2)),
        aVeryLongNameThatIsNowFine=pd.Series(range(4)),
        df=df,
        fail=fail,
    )
    print(DictOfPandas())
    # print(f.to_string())
    # print(f.to_string(show_df_column_names=False))
    print(f)
    print(repr(f))
