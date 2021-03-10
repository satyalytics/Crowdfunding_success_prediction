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
4. Backward difference encoder
5. BaseN encoder
6. Binary encoder
7. Catboost encoder
8. Generalized linear mixed model encoder
9. Hashing
10. Helmert 
11. James Stein
12. Leave one out
13. M estimate
14. Polynomial encoding
15. Sum encoding
16. Target encoding
17. Weight of Evidence 
18. Wrapper

We also try regrouping - top 10 most occuring groups in each category and else will be taken in other group
"""
from models import baseline
import static.raw_objects.encoders as enc
import static.raw_objects.cat_cols as cat_cols

def apply_encoder(df, encoder='label'):
    
    """
    Apply encoding method on categorical columns and return the transformed dataframe
    
    Args:
    -----
    df: The data frame which is going to be transformed
    encoder: The  type of encoder is going to be applied
    
    returns:
    -------
    df_ : The transformed dataframe
    """
    
    num_df = df.drop(cat_cols, axis=1)
    cat_df = df[cat_cols]
    
    enc = enc[encoder]
    cat_transformed = enc.fit_transform(cat_df)
    cat_transformed_df = pd.DataFrame(cat_transformed)
    df_ = pd.concat([num_df, cat_transformed_df], axis=1)
    
    return df_