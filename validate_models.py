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
    return np.argmax(pred, axis=1)

# -------------------------------------------------------------------------------------

# load stance model
json_file = open('output/model_architecture_stance.json', 'r')
loaded_stance_model_json = json_file.read()
json_file.close()
stance_model = model_from_json(loaded_stance_model_json)
# load weights into new model
stance_model.load_weights("output/my_model_stance_weights.h5")
print("Loaded stance model")

# load veracity model
json_file = open('output/model_architecture_veracity.json', 'r')
loaded_veracity_model_json = json_file.read()
json_file.close()
veracity_model = model_from_json(loaded_veracity_model_json)
# load weights into new model
veracity_model.load_weights("output/my_model_veracity_weights.h5")
print("Loaded veracity model")

path = 'preprocessing/saved_dataRumEval2019'
x_train = np.load(os.path.join(path, 'train/train_array.npy'), allow_pickle=True)
y_train = np.load(os.path.join(path, 'train/labels.npy'), allow_pickle=True)
x_val = np.load(os.path.join(path, 'dev/train_array.npy'), allow_pickle=True)
y_val = np.load(os.path.join(path, 'dev/labels.npy'), allow_pickle=True)
x_test = np.load(os.path.join(path, 'test/train_array.npy'), allow_pickle=True)
y_test = np.load(os.path.join(path, 'test/labels.npy'), allow_pickle=True)
#ids_test = np.load(os.path.join(path, 'test/ids.npy'), allow_pickle=True)

# evaluate loaded model on test data
stance_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                     metrics=['accuracy'])

train_preds = prediction_to_label(veracity_model.predict(x_train))
val_preds = prediction_to_label(veracity_model.predict(x_val))
test_preds = prediction_to_label(veracity_model.predict(x_test))

for name, true, pred in zip(["Training set", "val set", "Test set"],
                            [y_train, y_val, y_test],
                            [train_preds, val_preds, test_preds]):
    mse = mean_squared_error(true, pred, squared=False)
    f1 = f1_score(true, pred, labels=[0, 1, 2], average="macro")
    acc = accuracy_score(true, pred)
    print("#-----------------------------------------------------------------------------")
    print("CALCULATING FOR " + name)
    print("F1 score is " + str(f1))
    print("RMSE is " + str(mse))
    print("Accuracy is " + str(acc))
#score = stance_model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (stance_model.metrics_names[1], score[1]*100))



exit(0)