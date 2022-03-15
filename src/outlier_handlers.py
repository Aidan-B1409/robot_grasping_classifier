# Robot Grip Classifier
# AI-539 Machine Learning Challenges @ OSU
# Aidan Beery
# 03-06-2022


from scipy.stats.mstats import winsorize
from sklearn.ensemble import IsolationForest

RAND_STATE = 77


def DummyOutlierHandler(X, y):
    return X, y


def Winsorize(X, y):
    return X.apply(lambda x: winsorize(x, limits=[0.05, 0.05])), y


def DropOutliers(X, y):
    outlier_detector = IsolationForest(random_state=RAND_STATE)
    X['anomaly_score'] = outlier_detector.fit_predict(X)
    print(len(X[X['anomaly_score'] == -1].index))
    return (X.drop(index=X[X['anomaly_score'] == -1].index).drop('anomaly_score', axis=1),
            y.drop(index=X[X['anomaly_score'] == -1].index))
