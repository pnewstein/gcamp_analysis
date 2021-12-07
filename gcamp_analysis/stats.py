"""
Does the stats exactly as done in Average_Stim_Bar_Graph.m
"""
from typing import Optional
import warnings

from scipy.stats import kstest, ttest_ind, ranksums
import pandas as pd
import numpy as np

def get_p_value(data: pd.DataFrame) -> float:
    """
    Gets the p value of difference between two samples
    """
    # get out the data as two pd.Series
    # using lambda indexing based on ATR status
    group0 = data.loc[lambda df: df["ATR status"], "ΔF/F"]
    "ATR+ ΔF/F"
    group1 = data.loc[lambda df: np.logical_not(df["ATR status"]), "ΔF/F"]
    "ATR- ΔF/F"
    # unless we are more than 95% confident that
    # data is not normal, do t test
    p0 = kstest(group0, 'norm').pvalue
    p1 = kstest(group1, 'norm').pvalue

    p_value: Optional[float] = None
    if p0 >= 0.5 and p1 >= 0.5:
        # t test
        p_value = ttest_ind(group0, group1).pvalue
    else:
        # Mann-Whitney test
        warnings.warn(
            "Not normal. using mann-whitney test \n"
            "due do differences between python and matlab algorithms \n"
            "you can use the following matlab script\n\t"
            f"[p, ~] = ranksum({repr(group0.to_list())}, {repr(group1.to_list())})"
        )
        p_value = ranksums(group0, group1).pvalue
    if p_value is None:
        raise SyntaxError()
    return p_value

def format_p_value(p_value: float) -> str:
    "Formats the p value based on significance"
    if p_value > 0.05:
        return "ns"
    elif p_value <= 0.0001:
        return "P ≤ 0.0001"
    elif p_value <= 0.001:
        return "P ≤ 0.001"
    elif p_value <= 0.01:
        return "P ≤ 0.01"
    else:
        raise SyntaxError()
