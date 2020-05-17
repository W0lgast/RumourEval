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


def branch2treelabelsStance(ids_test, y_test, y_pred, answer_json):
    trees = np.unique(ids_test)
    tree_prediction = []
    tree_label = []

    remaining_labels = list(answer_json["subtaskaenglish"].keys())
    print(len(remaining_labels))
    seen = 0

    for tree in trees:

        if "ctd5nu9" in tree:
            print(tree)

        # treeindx = np.where(ids_test == tree)[0]
        treeindx = [i for i, x in enumerate(ids_test) if x == tree]
        treeVals = [x for x in ids_test if x == tree]
        temp_ids = []
        for t, treeid in enumerate(tree):
            if treeid in remaining_labels:
                id = treeVals[0][t]
                temp_ids.append(treeid)
                remaining_labels.remove(treeid)
                seen += 1
                # print(len(remaining_labels))
        # try:
        #     id = treeVals[0][1]
        # except:
        #     id = treeVals[0][-1]

        for treeid in temp_ids:
            label = answer_json["subtaskaenglish"][treeid]
            tree_label.append(convertTaskAtoNumber(label))

        # combine_labels = []
        # tree_label.append(y_test[treeindx[0]])
        temp_prediction = [y_pred[i] for i in treeindx]

        # for inc, x in enumerate(treeVals):
        #     num_b = len(x)
        #     temp_label = y_test[treeindx[inc]][:num_b]
        #     unique, counts = np.unique(temp_label, return_counts=True)
        #     combine_labels.append(unique[np.argmax(counts)])
        # unique, counts = np.unique(combine_labels, return_counts=True)
        # tree_label.append(unique[np.argmax(counts)])

        # all different predictions from branches from one tree
        for treeid in temp_ids:
            unique, counts = np.unique(temp_prediction, return_counts=True)
            tree_prediction.append(unique[np.argmax(counts)])
    print(f"seen: {seen}")
    print(f"remaining_labels: {len(remaining_labels)}")
    print(f"tree_pred: {len(tree_prediction)}")
    print(f"tree_label: {len(tree_label)}")
    return trees, tree_prediction, tree_label


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
