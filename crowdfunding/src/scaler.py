from sklearn import preprocessing
from ..statics.raw_objects import scalers, cat_cols


def apply_scaling(df,scaler='standard_scaler'):
    
    """
    Scale the data as instructed. Apply various scaling techniques available on the numerical data as
    encoded categorical values don't need scaling
    
    Args:
    -----
    df: Dataframe, which is scaled
    scaler: The scaling technique which is going to be applied
    
    Returns:
    --------
    df_ : The scaled data frame
    """
    num_df = df.drop(cat_cols, axis=1)
    cat_df = df[cat_cols]
    scaler = scalers[scaler]
    df_ = scaler.fit_transform(num_df)
    df_ = pd.concat([num_df, cat_df], axis=1)
    return df_, scaler