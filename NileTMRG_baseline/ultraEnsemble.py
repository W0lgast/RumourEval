"""
Will contain an ensemble model, needs to have a fit and predict function to fit signature of sklearn models.
"""

# -------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB


# -------------------------------------------------------------------------------------

class TaskBEnsemble(object):
    def __init__(self):
        self._models = [
            MultinomialNB(),
            DecisionTreeClassifier(),
            MLPClassifier(hidden_layer_sizes=tuple([100] * 20), max_iter=1000, early_stopping=True,
                          random_state=364, tol=0.0001, activation="relu", n_iter_no_change=100)
        ]

    def fit(self, X, Y):
        for model in self._models:
            print(f"Training model {type(model).__name__}")
            model.fit(X,Y)

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
            highest_votes = [c_k for c_k in count.keys() if count[c_k] == max(count.values())]
            # will always pick first element if there are ties
            ret.append(highest_votes[0])
        return ret
