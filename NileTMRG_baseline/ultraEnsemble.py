"""
Will contain an ensemble model, needs to have a fit and predict function to fit signature of sklearn models.
"""

# -------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV

# -------------------------------------------------------------------------------------

# class kippsSVC(LinearSVC):
#     def __init__(self, random_state=364):
#         super(self, LinearSVC).__init__(random_state=random_state)
#
#     def prefict_proba(self):


class TaskBEnsemble(object):
    def __init__(self, random_state=364):
        self._models = {
            # "MultinomialNB": MultinomialNB(),
            # "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
            # "SVC": SVC(kernel="sigmoid", random_state=random_state, probability=True),
            # LinearSVC(random_state=random_state),
            "CalibratedClassifierCV": CalibratedClassifierCV(
                base_estimator=LinearSVC(random_state=random_state), method="isotonic",
            ),
            # SklearnClassifier(SVC(kernel='linear', probability=True, random_state=random_state)),
            "MLPClassifier": MLPClassifier(
                hidden_layer_sizes=tuple([100] * 20),
                max_iter=1000,
                early_stopping=True,
                tol=0.0001,
                activation="relu",
                n_iter_no_change=100,
                random_state=random_state,
            )
            # RidgeClassifier(tol=1e-2, random_state=random_state, solver="sag"),
            # PassiveAggressiveClassifier(
            #     max_iter=50, random_state=random_state, n_jobs=-1
            # ),
            # KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
            # Perceptron(n_jobs=-1, random_state=random_state),
            # RandomForestClassifier(
            #     n_estimators=10000, n_jobs=-1, random_state=random_state
            # ),
        }
        self.__model_paramaters = {
            "MultinomialNB": {
                "alpha": [0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
                "fit_prior": (True, False),
            },
            "DecisionTreeClassifier": {"criterion": ("gini", "entropy")},
            "SVC": {
                "kernel": ("linear", "poly", "rbf", "sigmoid"),
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "degree": [2, 3, 4, 5],
                "gamma": ("scale", "auto"),
                "probability": [True],
            },
            "CalibratedClassifierCV": {
                "base_estimator": [LinearSVC(random_state=random_state)],
                "method": ["isotonic"],
            },
            "MLPClassifier": {
                "hidden_layer_sizes": [(100,), (200,), (300,), (400,), (500,), (1000,)],
                "activation": ("identity", "logistic", "tanh", "relu"),
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "learning_rate": ("constant", "invscaling", "adaptive"),
                "learning_rate_init": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            },
        }
        self.__ensemble_model = []

    def fit(self, X, Y):
        for i, model_name in enumerate(list(self._models.keys())):
            model = self._models[model_name]
            print(f"Running grid search on {model_name}...")
            clf = GridSearchCV(
                model, self.__model_paramaters[model_name], n_jobs=-1, verbose=2
            )
            clf.fit(X, Y)
            print(f"{model} best params: {clf.best_params_}")
            with open("gridsearch.txt", "a") as myfile:
                myfile.write(f"{model} best params: {clf.best_params_}\n")
            print(f"Training model {type(model).__name__}")
            model.fit(X, Y)

    def fit_ensemble(self, X, Y):

        estimators = []
        # create a dictionary of our models
        for model in self._models:
            estimators.append((f"{type(model).__name__}", model))
        # create our voting classifier, inputting our models
        ensemble = VotingClassifier(estimators, voting="soft", n_jobs=-1)
        ensemble.fit(X, Y)
        self.__ensemble_model.append(ensemble)

    def predict(self, X):
        preds = []
        for model in self._models:
            preds.append(model.predict(X))
        ret = []
        for i in range(len(preds[0])):
            count = {}
            for list_of_preds in preds:
                if list_of_preds[i] not in count.keys():
                    count[list_of_preds[i]] = 1
                else:
                    count[list_of_preds[i]] += 1
            highest_votes = [
                c_k for c_k in count.keys() if count[c_k] == max(count.values())
            ]
            # will always pick first element if there are ties
            ret.append(highest_votes[0])

        return self.__ensemble_model[0].predict(X)
        # return ret
