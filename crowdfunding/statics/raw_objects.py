"""
This module holds various objects in terms of dictionary, which are also stored in bin folder as 
pickle file. Those objects contains various lists and dictionaries related to encoding, scaling and
modelling.
"""

cat_cols = []

to_drop = []

encoders = {
    'onehot': one_hot.OneHotEncoder,
    'label': ordinal.OrdinalEncoder,
    'backward difference': backward_difference.BackwardDifferenceEncoder,
    'base n': basen.BaseNEncoder,
    'binary': binary.BinaryEncoder,
    'catboost': cat_boost.CatBoostEncoder,
    'count encoder': count.CountEncoder,
    'glmm': glmm.GLMMEncoder,
    'hashing': hashing.HashingEncoder,
    'helmert': helmert.HelmertEncoder,
    'james stein': james_stein.JamesSteinEncoder,
    'leave one out': leave_one_out.LeaveOneOutEncoder,
    'm estimate': m_estimate.MEstimateEncoder,
    'polynomial': polynomial.PolynomialEncoder,
    'sum': sum_coding.SumEncoder,
    'polynomial': wrapper.PolynomialWrapper,
    'woe': woe.WOEEncoder,
    'target': target_encoder.TargetEncoder
}


scalers = {
    'normalizer': preprocessing.Normalizer,
    'standard scaler': preprocessing.StandardScaler,
    'minmax': preprocessing.MinMaxScaler,
    'max_abs': preprocessing.MaxAbsScaler,
    'robust': preprocessing.RobustScaler,
    'power': preprocessing.PowerTransformer,
    'quantile': preprocessing.QuantileTransformer,
    'manual': preprocessing.FunctionTransformer
}


classifiers = {
    'logistic': linear_model.LogisticRegression,
    'ridge': linear_model.RidgeClassifier,
    'svm': svm.SVC,
    'knn': neighbors.KNeighborsClassifier,
    'dt': tree.DecisionTreeClassifier,
    'rf': ensemble.RandomForestClassifier,
    'ada': ensemble.AdaBoostClassifier,
    'gradboost': ensemble.GradientBoostingClassifier,
    'xgb': XGBClassifier
}