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
from sklearn.ensemble import AdaBoostClassifier

# -------------------------------------------------------------------------------------

# class kippsSVC(LinearSVC):
#     def __init__(self, random_state=364):
#         super(self, LinearSVC).__init__(random_state=random_state)
#
#     def prefict_proba(self):


class TaskBEnsemble(object):
    def __init__(self, random_state=364):
        self._models = [
            CalibratedClassifierCV(
                base_estimator=Perceptron(n_jobs=-1, random_state=random_state), method='isotonic', cv=None
            ),
            MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
            DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                                   max_depth=None, max_features=None, max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, presort='deprecated', splitter='best',
                                   random_state=random_state),
            SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='scale', kernel='sigmoid',
                max_iter=-1, probability=True, shrinking=True, tol=0.001,
                verbose=False, random_state=random_state),
            CalibratedClassifierCV(base_estimator=LinearSVC(C=1.0, class_weight=None,
                                                            dual=True, fit_intercept=True,
                                                            intercept_scaling=1,
                                                            loss='squared_hinge',
                                                            max_iter=1000,
                                                            multi_class='ovr', penalty='l2', tol=0.0001,
                                                            verbose=0, random_state=random_state),
                                   cv=None, method='isotonic'),
            CalibratedClassifierCV(
                base_estimator=MLPClassifier(
                    hidden_layer_sizes=tuple([100] * 20),
                    max_iter=1000,
                    early_stopping=True,
                    tol=0.0001,
                    activation="relu",
                    n_iter_no_change=100,
                    random_state=random_state,
                ),cv=None, method='isotonic'
            ),
            CalibratedClassifierCV(
                base_estimator=PassiveAggressiveClassifier(
                    max_iter=50, random_state=random_state, n_jobs=-1
                ),  method='isotonic', cv=None
            ),
            # RandomForestClassifier(
            #     n_estimators=10000, n_jobs=-1, random_state=random_state
            # ),
        ]

        for i, model in enumerate(self._models):
            print("Making an Adaboost")
            self._models[i] = AdaBoostClassifier(base_estimator=model, n_estimators=500)

        self.__ensemble_model = []

    def fit(self, X, Y):
        for model in self._models:
            print(f"Training model {type(model).__name__}")
            model.fit(X, Y)

    def fit_ensemble(self, X, Y):

        estimators = []
        # create a dictionary of our models
        for i, model in enumerate(self._models):
            estimators.append((f"{type(model).__name__}{i}", model))
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
