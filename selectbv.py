import pickle
# import joblib
import pandas as pd
import numpy as np
from typing import Any

with open("all_metrics.p", "rb") as fp:
    metr_lstm = pickle.load(fp)
lstm_nse = metr_lstm["NSE"]["lstm_NSE"]
ser_mod_ = pd.DataFrame().from_dict(lstm_nse, orient="columns")["ensemble"]


def select_bv_by_class(ser_mod: pd.Series = None, size: int = None, how: Any = None):
    """
    Select a sub-sample of basin using the pandas.cut() method to make classes. One basin is selected on each class
    by specifying whether it must be the last, the first or a randomly chosen.

    :param ser_mod: Series holding the features to selected, preferred as key indexed object
    :param size: how many classes to have (equivalent to number of desired basins)
    :param how: either of first (leftmost), last (rightmost), or randomly
    :return: list of selected basins
    """
    if ser_mod is None:
        ser_mod = pd.DataFrame().from_dict(lstm_nse, orient="columns")["ensemble"]
    if size is None:
        size = 56
    if how is None:
        how = "last"
    df_mod = ser_mod.to_frame()
    df_mod.index.name = "index"
    df_mod["rank"] = df_mod.rank(ascending=False, axis=0, method="first").astype(int)
    df_mod["class_rank"] = pd.cut(df_mod["rank"].values, bins=size, precision=0)
    grp = df_mod.reset_index().sort_values(by=ser_mod.name)[["index", "class_rank"]].groupby("class_rank")
    grp.index = range(1, size + 1)
    selected = list(grp.agg(how).values[:, 0])
    return selected

