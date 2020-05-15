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
x_dev = np.load(os.path.join(path, 'dev/train_array.npy'), allow_pickle=True)
y_dev = np.load(os.path.join(path, 'dev/labels.npy'), allow_pickle=True)
x_test = np.load(os.path.join(path, 'test/train_array.npy'), allow_pickle=True)
y_test = np.load(os.path.join(path, 'test/labels.npy'), allow_pickle=True)
#ids_test = np.load(os.path.join(path, 'test/ids.npy'), allow_pickle=True)

# evaluate loaded model on test data
stance_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                     metrics=['accuracy'])

train_preds = prediction_to_label(veracity_model.predict(x_train))
dev_preds = prediction_to_label(veracity_model.predict(x_dev))
test_preds = prediction_to_label(veracity_model.predict(x_test))

for name, true, pred in zip(["Training set", "Dev set", "Test set"],
                            [y_train, y_dev, y_test],
                            [train_preds, dev_preds, test_preds]):
    mse = mean_squared_error(true, pred, squared=False)
    f1 = f1_score(true, pred, labels=[0, 1, 2], average="macro")
    msg = "For " + name + ", f1 score is " + str(f1) + ". root mse is " + str(mse) + "."
    print(msg)
#score = stance_model.evaluate(x_test, y_test, verbose=0)
#print("%s: %.2f%%" % (stance_model.metrics_names[1], score[1]*100))



exit(0)