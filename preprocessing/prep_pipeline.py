"""
This is outer preprocessing file

To run:

python prep_pipeline.py

Main function has parameter that can be changed:

feats can be either 'text' for 'avgw2v' representation of the tweets or SemEvalfeatures for additional extra features concatenated with avgw2v.

"""
from preprocessing_tweets import load_dataset, load_test_data_twitter
from preprocessing_reddit import load_data, load_test_data_reddit
from transform_feature_dict import transform_feature_dict
from extract_thread_features import extract_thread_features_incl_response
import help_prep_functions
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
import json
import pickle as pkl

with open('../task_b_extra_features.json', 'r') as fp:
    TASK_B_EXTRA_FEATURES = json.load(fp)

#%%
with open("../data/test/final-eval-key.json", 'r') as f:
    TEST_DATA_LABELS = json.load(f)

def convert_label(label):
    if label == "true":
        return 0
    elif label == "false":
        return 1
    elif label == "unverified":
        return 2
    else:
        print(label)


def prep_pipeline(dataset="RumEval2019", feature_set=["avgw2v"]):
    use_reddit_data = True
    path = "saved_data" + dataset
    folds = {}
    folds = load_dataset()
    reddit = load_data()

    #folds["train"].extend(reddit["train"])
    #folds["dev"].extend(reddit["dev"])
    #folds["test"].extend(reddit["test"])

    folds["test"] = load_test_data_twitter()["test"]
    if use_reddit_data:
        reddit = load_data()
        folds['train'].extend(reddit['train'])
        folds['dev'].extend(reddit['dev'])
        reddit_test_data = load_test_data_reddit()['test']
        folds["test"].extend(reddit_test_data)

    help_prep_functions.loadW2vModel()

    ###

    #%%
    for fold in list(reversed(list(folds.keys()))):

        print(fold)
        feature_fold = []
        tweet_ids = []
        fold_stance_labels = []
        labels = []
        ids = []
        for conversation in folds[fold]:

            thread_feature_dict = extract_thread_features_incl_response(conversation)

            for an_id in thread_feature_dict.keys():
                if an_id in TASK_B_EXTRA_FEATURES.keys():
                    extra_feats = TASK_B_EXTRA_FEATURES[an_id]
                else:
                    print("not implemented yet, whatevers happening might be really bad")
                feat_dict = {str(i): e_f for i, e_f in enumerate(extra_feats)}
                thread_feature_dict[an_id].update(feat_dict)
            #dd here
            #TASK_B_EXTRA_FEATURES
            if fold == "test":
                # if it's in the test set it wont have veracity, assign it.
                conversation['veracity'] = TEST_DATA_LABELS["subtaskbenglish"][conversation['id']]
                conversation['source']['label'] = TEST_DATA_LABELS["subtaskaenglish"][conversation['id']]
                for reply in conversation['replies']:
                    reply['label'] = TEST_DATA_LABELS["subtaskaenglish"][reply['id_str']]


            (
                thread_features_array,
                thread_stance_labels,
                branches,
            ) = transform_feature_dict(
                thread_feature_dict, conversation, feature_set=feature_set+[str(i) for i in range(len(extra_feats))]
            )

            fold_stance_labels.extend(thread_stance_labels)
            tweet_ids.extend(branches)
            feature_fold.extend(thread_features_array)
            for i in range(len(thread_features_array)):
                labels.append(convert_label(conversation["veracity"]))
                ids.append(conversation["id"])

        #%
        if feature_fold != []:

            feature_fold = pad_sequences(
                feature_fold,
                maxlen=None,
                dtype="float32",
                padding="post",
                truncating="post",
                value=0.0,
            )

            fold_stance_labels = pad_sequences(
                fold_stance_labels,
                maxlen=None,
                dtype="float32",
                padding="post",
                truncating="post",
                value=0.0,
            )

            labels = np.asarray(labels)
            path_fold = os.path.join(path, fold)
            if not os.path.exists(path_fold):
                os.makedirs(path_fold)

            np.save(os.path.join(path_fold, "train_array"), feature_fold)
            np.save(os.path.join(path_fold, "labels"), labels)
            np.save(os.path.join(path_fold, "fold_stance_labels"), fold_stance_labels)
            np.save(os.path.join(path_fold, "ids"), ids)
            np.save(os.path.join(path_fold, "tweet_ids"), tweet_ids)


#%%
def main(data="RumEval2019", feats="SemEvalfeatures"):

    if feats == "text":
        prep_pipeline(dataset="RumEval2019", feature_set=["avgw2v"])
    elif feats == "SemEvalfeatures":
        SemEvalfeatures = [
            "avgw2v",
            "hasnegation",
            "hasswearwords",
            "capitalratio",
            "hasperiod",
            "hasqmark",
            "hasemark",
            "hasurl",
            "haspic",
            "charcount",
            "wordcount",
            "issource",
            "Word2VecSimilarityWrtOther",
            "Word2VecSimilarityWrtSource",
            "Word2VecSimilarityWrtPrev",
        ]
        prep_pipeline(dataset="RumEval2019", feature_set=SemEvalfeatures)


if __name__ == "__main__":
    main()
