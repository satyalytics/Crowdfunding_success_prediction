import numpy as np
import pandas as pd
from sklearn import preprocessing

"""
This module holds many encoding methods which will be used with the training and validation on the dataset
The best encoding method will be choosed as per time of execution and accuracy in terms of roc auc score.

The encoding methods are:

1. Label Encoder
2. OneHot Encoder
3. Count Encoder
4. 
"""


def label_encode(col):
    
    """
    Feature Encoding of ordinal data. 
    
    Args:
    -----
    col: The column need to be encoded on training data.
    data: Which type of data is it, if train data then fit and transform else only transorm.
    
    Return:
    -------
    col_en : Encoded column
    """
    
    le = preprocessing.LabelEncoder()
    try:
        col_en = le.fit_transform(col)
        
    except Error e:
        print(e)
        
        
    return col_en, le



def onehot_encode(df, list_of_cols):
    
    """
    Apply one hot encoding on the given columns
    
    Args:
    ----
    df: A dataframe on which encoding is done.
    list_of_cols: The list of categorical columns, which needs to be encoded
    data [train, test]: Type of data, if train: it will be fit and transform else only transform
    
    Returns:
    --------
    df_ : A dataframe with encoded columns
    """
    
    df2 = df.copy(deep=True)
    df2_cat = df2[list_of_cols]
    df2_num = df2.drop(list_of_cols, axis=1)
    
    ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    try:
        enc_arr = ohe.fit_tranform(df2_cat)
        cols = ohe.categories_
        enc_df = pd.DataFrame(enc_arr, columns=cols)
    except:
        print("error is happened")
        
    return enc_df, ohe


