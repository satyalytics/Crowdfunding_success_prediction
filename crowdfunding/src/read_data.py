import pandas as pd

def read_data(path):
    
    """
    Read a dataset from github and return a dataframe
    
    Args:
    ----
    path: A url contains data
    
    Returns:
    --------
    df: A dataframe
    """
    
    df = pd.read_csv(path)
    return df