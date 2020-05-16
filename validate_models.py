"""
This will load models and test them on the test set, it will print nice things.
"""

# -------------------------------------------------------------------------------------

from keras.models import model_from_json
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
import numpy as np
import os

# -------------------------------------------------------------------------------------


def prediction_to_label(pred):
    if len(pred.shape) == 2:
        return np.argmax(pred, axis=1)
    if len(pred.shape) == 3:
        return np.argmax(np.mean(pred, axis=1), axis=1)
        # return np.argmax(np.mean(pred, axis=1), axis=1)
    print("ERROR! ERROR!1!!!!")
    exit(0)


# -------------------------------------------------------------------------------------

# load stance model
json_file = open("output/model_architecture_stance.json", "r")
loaded_stance_model_json = json_file.read()
json_file.close()
stance_model = model_from_json(loaded_stance_model_json)
# load weights into new model
stance_model.load_weights("output/my_model_stance_weights.h5")
print("Loaded stance model")

# load veracity model
json_file = open("output/model_architecture_veracity.json", "r")
loaded_veracity_model_json = json_file.read()
json_file.close()
veracity_model = model_from_json(loaded_veracity_model_json)
# load weights into new model
veracity_model.load_weights("output/my_model_veracity_weights.h5")
print("Loaded veracity model")

path = "preprocessing/saved_dataRumEval2019"
x_train = np.load(os.path.join(path, "train/train_array.npy"), allow_pickle=True)
y_train_stance = np.load(
    os.path.join(path, "train/fold_stance_labels.npy"), allow_pickle=True
)
y_train_veracity = np.load(os.path.join(path, "train/labels.npy"), allow_pickle=True)
x_val = np.load(os.path.join(path, "dev/train_array.npy"), allow_pickle=True)
y_val_stance = np.load(
    os.path.join(path, "dev/fold_stance_labels.npy"), allow_pickle=True
)
y_val_veracity = np.load(os.path.join(path, "dev/labels.npy"), allow_pickle=True)
x_test = np.load(os.path.join(path, "test/train_array.npy"), allow_pickle=True)
y_test_stance = np.load(
    os.path.join(path, "test/fold_stance_labels.npy"), allow_pickle=True
)
y_test_veracity = np.load(os.path.join(path, "test/labels.npy"), allow_pickle=True)
# ids_test = np.load(os.path.join(path, 'test/ids.npy'), allow_pickle=True)

### TESTING STANCE MODEL (PART A)
train_preds = prediction_to_label(veracity_model.predict(x_train))
val_preds = prediction_to_label(veracity_model.predict(x_val))
test_preds = prediction_to_label(veracity_model.predict(x_test))

for name, true, pred in zip(
    ["Training set", "val set", "Test set"],
    [y_train_stance, y_val_stance, y_test_stance],
    [train_preds, val_preds, test_preds],
):
    # if name == "Test set":
    #     for i in [8, 33, 85, 133, 200]:
    #         print(pred[i])
    mse = mean_squared_error(true, pred, squared=False)
    f1 = f1_score(true, pred, labels=np.unique(pred), average="macro")
    acc = accuracy_score(true, pred)
    print(
        "#-----------------------------------------------------------------------------"
    )
    print("A - STANCE: CALCULATING FOR " + name)
    print("A - STANCE: F1 score is " + str(f1) + " : EXPECTED BASELINE ON TEST: 0.4929")
    # print("A - STANCE: RMSE is " + str(mse))
    print("A - STANCE: Accuracy is " + str(acc))

print("#-----------------------------------------------------------------------------")
print("#-----------------------------------------------------------------------------")

### TESTING VERACITY MODEL (PART B)
train_preds = prediction_to_label(stance_model.predict(x_train))
val_preds = prediction_to_label(stance_model.predict(x_val))
test_preds = prediction_to_label(stance_model.predict(x_test))

for name, true, pred in zip(
    ["Training set", "val set", "Test set"],
    [y_train_veracity, y_val_veracity, y_test_veracity],
    [train_preds, val_preds, test_preds],
):

    # if name == "Test set":
    #     for i in [8, 33, 85, 133, 200]:
    #         print(pred[i])
    mse = mean_squared_error(true, pred, squared=False)
    f1 = f1_score(true, pred, labels=np.unique(pred), average="macro")
    acc = accuracy_score(true, pred)
    print(
        "#-----------------------------------------------------------------------------"
    )
    print("B - VERACITY: CALCULATING FOR " + name)
    print(
        "B - VERACITY: F1 score is " + str(f1) + " : EXPECTED BASELINE ON TEST: 0.3364"
    )
    print("B - VERACITY: RMSE is " + str(mse) + " : EXPECTED BASELINE ON TEST: 0.7806")
    print("B - VERACITY: Accuracy is " + str(acc))


# score = stance_model.evaluate(x_test, y_test, verbose=0)
# print("%s: %.2f%%" % (stance_model.metrics_names[1], score[1]*100))

# score = stance_model.evaluate(x_test, y_test, verbose=0)
# print("%s: %.2f%%" % (stance_model.metrics_names[1], score[1]*100))


exit(0)
