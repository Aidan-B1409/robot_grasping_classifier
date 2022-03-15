# Robot Grip Classifier
# AI-539 Machine Learning Challenges @ OSU
# Aidan Beery
# 03-06-2022

import argparse
import pickle
import os

from outlier_handlers import DummyOutlierHandler, Winsorize, DropOutliers
from train_hyperparams import get_tuned_models
from zero_handlers import DropEffort, IncrementFeatureSpace, DummyZeroHandler

import pandas as pd
import numpy as np

# well that's quite a rude name for a classifier
# he's doing his best ok :(
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


RAND_STATE = 77
LABEL = 'robustness'
scores = [precision_score, recall_score, accuracy_score]
classifiers = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier]


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A robot grip robustness classifier."
    )
    parser.add_argument(
        '-i', '--input', dest='input', required=True,
        help='File of robot grasping data. Expects 1 csv.'
    )
    parser.add_argument(
        '-m', '--models', dest='models', required=True,
        help='Location of model pickle file, if such a thing exists'
    )
    return parser.parse_args()


# Take a dataframe, create a test-train-validation split
# with equal length test and validation subsets.
# Split all groups into X and y.
# source for splitting logic:
# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213 # noqa: E501
# this function is the definition of insanity.
def make_dataset(df: pd.DataFrame, p: int, split) -> dict:
    return {k: v for k, v in zip(
                ('train', 'test', 'validate'),
                [{k1: v1 for k1, v1 in zip(
                    ('X', 'y'),
                    (s.drop(LABEL, axis=1), s[LABEL]))}
                 for s in split(df, p)])
            }


def grouped_ttv_split(df: pd.DataFrame, p: int):
    X, y = df.drop(LABEL, axis=1), df[LABEL]
    gs = GroupShuffleSplit(n_splits=2, train_size=(1-p), random_state=RAND_STATE)
    train_idx, holdout_idx = next(gs.split(X, y, df['experiment_number']))
    gs_test = GroupShuffleSplit(n_splits=2, train_size=0.5, random_state=RAND_STATE)
    val_idx, test_idx = next(gs_test.split(X.loc[holdout_idx],
                                           y.loc[holdout_idx],
                                           X.loc[holdout_idx]['experiment_number']))
    return df.loc[train_idx], df.loc[test_idx], df.loc[val_idx]


def ttv_split(df: pd.DataFrame, p: int):
    return np.split(df.sample(frac=1, random_state=RAND_STATE),
                    [int((1-(2*p))*len(df)), int((1-p)*len(df))])


def experiment(ds, outlier_handler, zero_handler, models, scores) -> pd.DataFrame:
    X_train, y_train = outlier_handler(ds['train']['X'], ds['train']['y'])
    X_test, y_test = outlier_handler(ds['test']['X'], ds['test']['y'])

    X_train, X_test = zero_handler(X_train), zero_handler(X_test)

    return pd.DataFrame(
        [pd.Series([s(y_test, m.predict(X_test)) for s in scores])
         .reset_index(drop=True)
         .set_axis([s.__name__ for s in scores])
         .rename(m.steps[1][0])
         for m in [m.fit(X_train, y_train) for m in models]]
    )


def main():
    args = parseargs()
    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()
    df[LABEL] = np.where(df[LABEL] > 100, int(1), int(0))

    ds = make_dataset(df, 0.15, ttv_split)
    ds_grouped = make_dataset(df, 0.15, grouped_ttv_split)
    ds_first_measure = make_dataset(df.groupby(['experiment_number']).apply(lambda x: x.iloc[0]), 0.15, ttv_split)

    datasets = [ds, ds_grouped, ds_first_measure]

    any(map(lambda x:
        any(map(lambda y:
            y['X'].drop(['experiment_number', 'measurement_number'], axis=1, inplace=True), x.values())), datasets))

    if os.path.isfile(args.models):
        models = pickle.load(open(args.models, 'rb'))
    else:
        models = get_tuned_models(ds, classifiers)
        pickle.dump(models, open(args.models, 'wb'))

    models[2].steps[1][1].set_params(n_jobs=-1)

    # Step 0 - Get hyperparam tuned models
    for model in models:
        pd.Series(model.get_params()).to_latex(f"{model.steps[1][0]}.tex", float_format="{:0.2f}".format)

    # Step 0.5 - Append a baseline classifier
    models.append(make_pipeline(StandardScaler(),
                                DummyClassifier(strategy='constant', constant=1, random_state=RAND_STATE)))

    # Step 1 - Evaulate ds grouping strategies
    any(map(
        lambda x:
            (experiment(x[1], DummyOutlierHandler, DummyZeroHandler, models, scores)
                .to_latex(f"{x[0]}_eval.tex", float_format="{:0.3f}".format)),
        [('dsAll', ds), ('dsGrouped', ds_grouped), ('dsFirst', ds_first_measure)]
    ))

    # Step 2 - Eval zero handlers
    any(map(
        lambda y:
            (experiment(ds, DummyOutlierHandler, y, models, scores)
                .to_latex(f"{y.__name__}_eval.tex", float_format="{:0.3f}".format)),
        [DropEffort, IncrementFeatureSpace]
    ))

    # Step 3 - Eval outlier handlers
    any(map(
        lambda z:
            (experiment(ds, z, DummyZeroHandler, models, scores)
                .to_latex(f"{z.__name__}_eval.tex", float_format="{:0.3f}".format)),
        [DropOutliers, Winsorize]
    ))


if __name__ == '__main__':
    main()
