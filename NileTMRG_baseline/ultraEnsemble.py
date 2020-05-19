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
from sklearn.feature_selection import SelectFromModel

# -------------------------------------------------------------------------------------

# class kippsSVC(LinearSVC):
#     def __init__(self, random_state=364):
#         super(self, LinearSVC).__init__(random_state=random_state)
#
#     def prefict_proba(self):


class TaskBEnsemble(object):
    def __init__(self, random_state=364):
        self._transformer = None
        self._models = [
            # MultinomialNB(),
            DecisionTreeClassifier(random_state=random_state),
            SVC(kernel="sigmoid", random_state=random_state, probability=True),
            # LinearSVC(random_state=random_state),
            CalibratedClassifierCV(
                base_estimator=LinearSVC(random_state=random_state),
                method="isotonic",
                cv=None,  # your CV instance
            ),
            # SklearnClassifier(SVC(kernel='linear', probability=True, random_state=random_state)),
            MLPClassifier(
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
        ]
        self.__ensemble_model = []

    def fit(self, X, Y):
        for model in self._models:
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

    def predict(self, X, proba=False):
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

        ensemble =  self.__ensemble_model[0]
        if proba==False:
            return ensemble.predict(X)
        elif proba==True:
            return ensemble.predict_proba(X)
        # return ret
