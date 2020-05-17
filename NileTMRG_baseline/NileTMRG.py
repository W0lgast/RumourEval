import logging
import sys
from optparse import OptionParser
from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.extmath import density
from preprocess import *
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from ultraEnsemble import TaskBEnsemble


from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier

def convertTaskBtoNumber(label):
    if label == "true":
        return 0
    elif label == "false":
        return 1
    elif label == "unverified":
        return 2

####
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

op = OptionParser()
op.add_option(
    "--sklearn",
    action="store_true",
    dest="sklearn",
    help="Run all functions from sklearn.",
)
op.add_option(
    "--chi2_select",
    action="store",
    type="int",
    dest="select_chi2",
    help="Select some number of features using a chi-squared test",
)
op.add_option(
    "--confusion_matrix",
    action="store_true",
    dest="print_cm",
    help="Print the confusion matrix.",
)
op.add_option(
    "--all_categories",
    action="store_true",
    dest="all_categories",
    help="Whether to use all categories or not.",
)
op.add_option("--use_hashing", action="store_true", help="Use a hashing vectorizer.")
op.add_option(
    "--n_features",
    action="store",
    type=int,
    default=2 ** 16,
    help="n_features when using the hashing vectorizer.",
)
op.add_option(
    "--filtered",
    action="store_true",
    help="Remove newsgroup information that is easily overfit: "
    "headers, signatures, and quoting.",
)


def is_interactive():
    return not hasattr(sys.modules["__main__"], "__file__")


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()
####
#%%
# Read tweets from each of the sets
all_data = read_data()

### TRAIN ###
# Get BOW
BOW_sents, all_extra_feats, Y, ids = get_features(all_data, whichset="training")

cv = CountVectorizer()

cv_fit = cv.fit_transform(BOW_sents)

BOW_features = cv_fit.toarray()

# Append features
X = []
for i in range(len(BOW_features)):
    line = list(BOW_features[i]) + all_extra_feats[i]
    X.append(line)

### DEV ###
# Get BOW
BOW_sents_dev, all_extra_feats_dev, Y_dev, ids_dev = get_features(
    all_data, whichset="development"
)

dev_cv = cv.transform(BOW_sents_dev)

BOW_features_dev = dev_cv.toarray()

# Append features
X_dev = []
for i in range(len(BOW_features_dev)):
    line = list(BOW_features_dev[i]) + all_extra_feats_dev[i]
    X_dev.append(line)

# scale
# scl = StandardScaler()
# X = scl.fit_transform(X)
# fit classifier


### TEST ###
test_BOW_sents, test_all_extra_feats, Y_test, ids_test = get_features(
    all_data, whichset="testing"
)
if not isinstance(Y_test[0], str):
    Y_test = [y[0] for y in Y_test]
test_cv_fit = cv.transform(test_BOW_sents)
test_BOW_features = test_cv_fit.toarray()

# Append features
X_test = []
for i in range(len(test_BOW_features)):
    line = list(test_BOW_features[i]) + test_all_extra_feats[i]
    X_test.append(line)

# X_test = scl.transform(X_test)

##############################   MODELS GO HERE   ##############################
if not opts.sklearn:
    # BASELINE
    #clf = LinearSVC(random_state=364)

    #clf = SVC(kernel="sigmoid", random_state=364)

    #kernel = 1.0 * RBF(1.0)
    #clf = GaussianProcessClassifier(kernel=kernel, random_state=364).fit(X, Y)

    #clf = SGDClassifier(alpha=0.0001, max_iter=50, penalty="elasticnet")

    #clf = MLPClassifier(hidden_layer_sizes=tuple([100]*20), max_iter=1000, early_stopping=True,
    #                    random_state=364, tol=0.0001, activation="relu", n_iter_no_change=100)

    #clf = DecisionTreeClassifier()
    #clf = MultinomialNB()

    clf = TaskBEnsemble(random_state=364)

    clf.fit(X, Y)

# Only used if running sklearn
def benchmark(clf):
    print("_" * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X, Y)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_dev)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(Y_dev, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, "coef_"):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(Y_dev, pred))

    print()
    clf_descr = str(clf).split("(")[0]
    return clf_descr, score, train_time, test_time


###########################   PREDICTIONS GO HERE   ###########################

if not opts.sklearn:
    Y_pred = clf.predict(X)

    # get scores
    print("Train Accuracy:")
    print(metrics.accuracy_score(Y, Y_pred))

    print("Train Macro F:")
    print(metrics.f1_score(Y, Y_pred, average="macro"))

    #print("Train RMSE:")
    #print(metrics.mean_squared_error([convertTaskBtoNumber(y) for y in Y],
    #                                 [convertTaskBtoNumber(y) for y in Y_pred], squared=False))

    Y_dev_pred = clf.predict(X_dev)

    # get scores
    print("Validation Accuracy:")
    print(metrics.accuracy_score(Y_dev, Y_dev_pred))

    print("Validation Macro F:")
    print(metrics.f1_score(Y_dev, Y_dev_pred, average="macro"))

    #print("Validation RMSE:")
    #print(metrics.mean_squared_error([convertTaskBtoNumber(y) for y in Y_dev],
    #                                 [convertTaskBtoNumber(y) for y in Y_dev_pred], squared=False))

    Y_pred = clf.predict(X_test)

    # get scores
    print("Testing Accuracy:")
    print(metrics.accuracy_score(Y_test, Y_pred))

    print("Testing Macro F:")
    print(metrics.f1_score(Y_test, Y_pred, average='macro'))

    #print("Testing RMSE:")
    #print(metrics.mean_squared_error([convertTaskBtoNumber(y) for y in Y_test],
    #                                 [convertTaskBtoNumber(y) for y in Y_pred], squared=False))

    #print(confusion_matrix(Y_test, Y_pred))

##

if opts.sklearn:
    results = []
    for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest"),
    ):
        print("=" * 80)
        print(name)
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        print("=" * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False, tol=1e-3)))

        # Train SGD model
        results.append(
            benchmark(SGDClassifier(alpha=0.0001, max_iter=50, penalty=penalty))
        )

    # Train SGD with Elastic Net penalty
    print("=" * 80)
    print("Elastic-Net penalty")
    results.append(
        benchmark(SGDClassifier(alpha=0.0001, max_iter=50, penalty="elasticnet"))
    )

    # Train NearestCentroid without threshold
    print("=" * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    print("=" * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=0.01)))
    results.append(benchmark(BernoulliNB(alpha=0.01)))
    results.append(benchmark(ComplementNB(alpha=0.1)))

    print("=" * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(
        benchmark(
            Pipeline(
                [
                    (
                        "feature_selection",
                        SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3)),
                    ),
                    ("classification", LinearSVC(penalty="l2")),
                ]
            )
        )
    )


#%%
if not opts.sklearn:
    # # Uncomment when need to test
    # submission_B = {}
    # for i, id in enumerate(ids_test):
    #     submission_B[id] = [Y_pred[i], 1]
    #
    # # def labell2strB(label):
    # #
    # #    if label == 0:
    # #        return("true")
    # #    elif label == 1:
    # #        return("false")
    # #    elif label == 2:
    # #        return("unverified")
    # #    else:
    # #        print(label)
    #
    # submission_B = {}
    # for i, id in enumerate(ids_test):
    #     submission_B[id] = [Y_pred[i], 1]
    #
    # subtaskaenglish = {}
    # subtaskbenglish = {}
    #
    # # for i,id in enumerate(idsA):
    # #    subtaskaenglish[id] = labell2strA(predictionsA[i])
    #
    # # for i,id in enumerate(idsB):
    # #    subtaskbenglish[id] = [labell2strB(predictionsB[i]),confidenceB[i]]
    #
    # answer = {}
    # answer["subtaskaenglish"] = {}
    # answer["subtaskbenglish"] = submission_B
    #
    # answer["subtaskadanish"] = {}
    # answer["subtaskbdanish"] = {}
    #
    # answer["subtaskarussian"] = {}
    # answer["subtaskbrussian"] = {}
    #
    # with open("answerB.json", "w") as f:
    #     json.dump(answer, f)
    print("Not saving test output...")
else:
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, 0.2, label="score", color="navy")
    # plt.barh(indices + 0.3, training_time, 0.2, label="training time", color="c")
    # plt.barh(indices + 0.6, test_time, 0.2, label="test time", color="darkorange")
    plt.yticks(())
    plt.legend(loc="best")
    # plt.subplots_adjust(left=0.25)
    # plt.subplots_adjust(top=0.95)
    # plt.subplots_adjust(bottom=0.05)

    for i, c in zip(indices, clf_names):
        plt.text(-0.1, i, c)

    plt.show()
