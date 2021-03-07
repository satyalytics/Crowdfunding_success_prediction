from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

def baseline(X,y):
    rf = RandomForestClassifier()
    score = model_selection.cross_val_score(rf, X,y, cv=10)
    return score