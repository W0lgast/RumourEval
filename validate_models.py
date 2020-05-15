"""
This will load models and test them on the test set, it will print nice things.
"""

# -------------------------------------------------------------------------------------

import pickle as pkl
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from preprocessing.preprocessing_reddit import load_test_data_reddit
from preprocessing.preprocessing_tweets import load_test_data_twitter

import numpy
import os

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

data_reddit = load_test_data_reddit()
data_tweets = load_test_data_twitter()

# evaluate loaded model on test data
stance_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'RMSE', 'F1'])
score = stance_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (stance_model.metrics_names[1], score[1]*100))


exit(0)