from __future__ import annotations
import pandas as pd
from pandas import DataFrame
import numpy as np

def apply_noise_to_column(df:DataFrame,cols_to_noise:list = None,noise_multiplier:float = 0.1) -> DataFrame:
    to_apply_on = df.copy()
    for col in cols_to_noise:
            col_std = to_apply_on[col].std()
            noise = np.random.normal(loc=0, scale=col_std * noise_multiplier, size=len(to_apply_on))
            to_apply_on[col] = to_apply_on[col] + noise
    return to_apply_on

def get_dummy_nday_forecast(ndays:int,cols_to_forecast:list,df:DataFrame) -> DataFrame:
    """
    ndays: number of days to forecast
    cols_to_forecast: list of columns to apply dummy forecast on
    df: dataframe to copy and apply dummy forecast on
    """
    to_apply_on = df.copy()
    all_shifted = {}
    for c in cols_to_forecast:
        try:
            for i in range(1, ndays):
                all_shifted[f'frcst_{i}_{c}'] = to_apply_on[c].shift(-i)
        except Exception as e:
            raise Exception("Couldn't create dummy forecast") from e
    to_apply_on = pd.concat([to_apply_on, pd.DataFrame(all_shifted)], axis=1)
    return to_apply_on

def apply_noise_to_forecasted_days(forecasted_days:int, noisemap:dict, df:DataFrame, variables_to_noise:list = None) -> DataFrame:
    to_apply_on = df.copy()

    for i in range(1,forecasted_days):
        if variables_to_noise:
            frcst_i = [c for c in to_apply_on.columns if f'frcst_{i}' in c and c.replace(f'frcst_{i}','') in variables_to_noise]
        else:
            frcst_i = [c for c in to_apply_on.columns if f'frcst_{i}' in c]

        multiplier = noisemap.get(i, 0.1)
        
        for col in frcst_i:
            col_std = to_apply_on[col].std()
            noise = np.random.normal(loc=0, scale=col_std * multiplier, size=len(to_apply_on))
            to_apply_on[col] = to_apply_on[col] + noise

    return to_apply_on