"""
Holds a majority lass classifier for the veracity and stance models
"""

# -------------------------------------------------------------------------------------

from branch2treelabels import branch2treelabelsStance
import numpy as np

# -------------------------------------------------------------------------------------

MAJORITY_CLASS_STANCE = 1           # COMMENT
MAJORITY_CLASS_VERACITY = 0         # true

class MCC_Stance(object):
    """
    Always predicts "comment"
    """
    def __init__(self):
        self._majority_class = MAJORITY_CLASS_STANCE

    def train(self, x, y):
        print("Training not yet implemented, majority class is hardcoded!")
        exit(0)

    def predict_classes(self, x):
        return np.array([MAJORITY_CLASS_STANCE]*x.shape[0])

class MCC_Veracity(object):
    """
    Always predicts "true"
    """
    def __init__(self):
        self._majority_class = MAJORITY_CLASS_STANCE

    def train(self, x, y):
        print("Training not yet implemented, majority class is hardcoded!")
        exit(0)

    def predict_classes(self, x):
        return np.array([MAJORITY_CLASS_STANCE]*x.shape[0])