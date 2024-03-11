"""contains pandas-specific functions for operating on dataframes and series"""
from typing import Union

import geopandas as gpd
import pandas as pd


def lowercase_columns(df: Union[pd.DataFrame, gpd.GeoDataFrame]):
    """lowercases the column names of a dataframe or geodataframe"""
    # modify in place
    df.columns = [x.lower() for x in df.columns.values]
    return df
