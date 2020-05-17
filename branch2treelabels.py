"""
This is a postprocessing function that takes per-branch predicitons and takes
majority vote to generate per-tree prediction
"""
import numpy as np


def convertTaskAtoNumber(label):
    if label == "support":
        return 0
    elif label == "comment":
        return 1
    elif label == "deny":
        return 2
    elif label == "query":
        return 3


def convertNumbertoTaskA(label):
    if label == 0:
        return "support"
    elif label == 1:
        return "comment"
    elif label == 2:
        return "deny"
    elif label == 3:
        return "query"


def branch2treelabelsStance(ids_test, y_pred):
    trees = np.unique(ids_test)

    predictions = {}

    for tree in trees:

        treeindx = [i for i, x in enumerate(ids_test) if x == tree]

        treeVals = [x for x in ids_test if x == tree]

        for i, thread in enumerate(treeVals):
            for j, idtree in enumerate(thread):
                predictions[str(idtree)] = convertNumbertoTaskA(y_pred[treeindx[i]][j])

    return trees, predictions


def branch2treelabelsVeracity(ids_test, y_pred):
    trees = np.unique(ids_test)

    predictions = {}

    for tree in trees:

        treeindx = [i for i, x in enumerate(ids_test) if x == tree]

        treeVals = [x for x in ids_test if x == tree]

        for i, thread in enumerate(treeVals):
            pred = np.argmax(y_pred[treeindx[i]])
            # print(f"thred:{thread}")
            # print(f"pred:{pred}")
            # print(f"y_pred[treeindx[i]][j]:{y_pred[treeindx[i]]}")
            # print(f"y_pred[treeindx[i]][j][pred]:{y_pred[treeindx[i]][pred]}")
            predictions[str(thread)] = [pred, y_pred[treeindx[i]][pred]]

    return trees, predictions


def branch2treelabels(ids_test, y_test, y_pred, confidence):
    trees = np.unique(ids_test)
    tree_prediction = []
    tree_label = []
    tree_confidence = []
    for tree in trees:
        treeindx = np.where(ids_test == tree)[0]
        tree_label.append(y_test[treeindx[0]])
        tree_confidence.append(confidence[treeindx[0]])
        temp_prediction = [y_pred[i] for i in treeindx]
        # all different predictions from branches from one tree
        unique, counts = np.unique(temp_prediction, return_counts=True)
        tree_prediction.append(unique[np.argmax(counts)])
    return trees, tree_prediction, tree_label, tree_confidence


def branch2treelabels_test(ids_test, y_pred, confidence):
    trees = np.unique(ids_test)
    tree_prediction = []
    tree_confidence = []
    for tree in trees:
        treeindx = np.where(ids_test == tree)[0]
        tree_confidence.append(np.float(confidence[treeindx[0]]))
        temp_prediction = [y_pred[i] for i in treeindx]
        # all different predictions from branches from one tree
        unique, counts = np.unique(temp_prediction, return_counts=True)
        tree_prediction.append(unique[np.argmax(counts)])
    return trees, tree_prediction, tree_confidence
