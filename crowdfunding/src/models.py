from sklearn import ensemble
from sklearn import model_selection

def baseline(X,y):
    rf = ensemble.RandomForestClassifier()
    score = model_selection.cross_val_score(rf, X,y, cv=10)
    return score


def apply_model(df, model):
    pass