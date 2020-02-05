import numpy as np
import pandas as pd

def recenter(df, center):
    # Recenter data by subtracting center from all values for each column, if negative roll over to 360
    new_df = df - center[np.newaxis,:]
    mask = new_df < 0
    new_df[mask] = 360 + new_df[mask]
    return new_df

def calculate_center(df, start_idx=0, end_idx=300):
    return np.mean(df[start_idx:end_idx,:], axis=0)

def check_data_quality(df):
    df_diff = np.diff(df, n=1, axis=0, prepend=0)
    df_idx1 = np.argwhere(df_diff > 355) # 0 => 360
    df_idx2 = np.argwhere(df_diff < -355) # 360 => 0

    if df_idx1.shape[0]==0:
        # Case 1: No crossings
        if df_idx2.shape[0]==0:
            pass
        # Case 2: 1 Crossing from 360 => 0
        else:
            pass
    elif df_idx2.shape[0]==0:
        # Case 3: 1 Crossing from 0 => 360
        pass
    else:
        # Case 4,5,6: Multiple crossings, must determine which is last crossing
        
        pass


    pass

def transform_data(df):
    
    pass

def full_pipeline(df):
    pass
