# Robot Grip Classifier
# AI-539 Machine Learning Challenges @ OSU
# Aidan Beery
# 03-06-2022

from typing import List

import numpy as np

# Data processing tools
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

RAND_STATE = 77

parameters = {
    'logisticregression': {
        # Only test on l2 regularization since solvers only work with l2
        'logisticregression__penalty': ['l2'],
        'logisticregression__random_state': [RAND_STATE],
        'logisticregression__C': np.logspace(-4, 4, 10),
        'logisticregression__solver': ('lbfgs', 'newton-cg', 'saga'),
        'logisticregression__max_iter': (200, 400),
    },
    'randomforestclassifier': {
        'randomforestclassifier__n_estimators': (50, 100),
        'randomforestclassifier__criterion': ['gini'],
        'randomforestclassifier__max_features': ('sqrt', 'log2'),
        'randomforestclassifier__random_state': [RAND_STATE],
    },
    'kneighborsclassifier': {
        'kneighborsclassifier__n_neighbors': (3, 5),
        'kneighborsclassifier__algorithm': ['auto'],
        'kneighborsclassifier__metric': ['minkowski'],
        # always use minkowski distance metric, use power value of 1 to simulate manhattan, 2 to simulate euclidian
        'kneighborsclassifier__p': (1, 2, 3),
        'kneighborsclassifier__n_jobs': [-1]
    },
}


def get_tuned_models(ds, classifiers) -> List[Pipeline]:
    models = [make_pipeline(StandardScaler(), model()) for model in classifiers]
    return [tune_model(model, ds['validate']['X'], ds['validate']['y']) for model in models]


def tune_model(model: Pipeline, X, y) -> Pipeline:
    gs = GridSearchCV(
            estimator=model,
            param_grid=parameters[model.steps[1][0]],
            scoring='accuracy',
            refit=True,
            cv=3,
            n_jobs=-1,
            verbose=2,
            pre_dispatch=5
        )

    gs = gs.fit(X, y)
    return gs.best_estimator_  # type: ignore


