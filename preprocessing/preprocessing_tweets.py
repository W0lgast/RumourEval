# -*- coding: utf-8 -*-
"""
This file contains function to load tweets

"""
import os
import json
from tree2branches import tree2branches
import pickle as pkl
HACK = pkl.load( open( "../hack.pkl", "rb" ) )
HACK_TWEETS = pkl.load(open("../hack_tweets.pkl", "rb"))
#%%
PATH_TO_TEST_TWITTER = "../data/test/twitter-en-test-data"


def load_true_labels():

    tweet_label_dict = {}
    veracity_label_dict = {}
    path_dev = "../data/train/dev-key.json"
    with open(path_dev, "r") as f:
        dev_key = json.load(f)

    path_train = "../data/train/train-key.json"
    with open(path_train, "r") as f:
        train_key = json.load(f)

    path_test = "../data/test/final-eval-key.json"
    with open(path_test, "r") as f:
        test_key = json.load(f)

    tweet_label_dict["dev"] = dev_key["subtaskaenglish"]
    tweet_label_dict["train"] = train_key["subtaskaenglish"]
    tweet_label_dict["test"] = test_key["subtaskaenglish"]

    veracity_label_dict["dev"] = dev_key["subtaskbenglish"]
    veracity_label_dict["train"] = train_key["subtaskbenglish"]
    veracity_label_dict["test"] = test_key["subtaskbenglish"]

    return tweet_label_dict, veracity_label_dict


def load_test_data_twitter(set_path=PATH_TO_TEST_TWITTER):
    allconv = []
    train_dev_split = {}
    train_dev_split["dev"] = []
    train_dev_split["train"] = []
    train_dev_split["test"] = []
    tweet_data = sorted(os.listdir(set_path))
    newfolds = [i for i in tweet_data if i[0] != "."]
    tweet_data = newfolds  # conversation ids, source post id == conversation id
    conversation = {}
    # build conversations for tweet group
    for tweet_topic in tweet_data:
        path = os.path.join(set_path, tweet_topic)
        tweet_topic_data = sorted(os.listdir(path))
        tweet_topic_data = [i for i in tweet_topic_data if i[0] != "."]
        for foldr in tweet_topic_data:
            flag = 0
            conversation["id"] = foldr
            tweets = []
            path_repl = path + "/" + foldr + "/replies"
            files_t = sorted(os.listdir(path_repl))
            newfolds = [i for i in files_t if i[0] != "."]
            files_t = newfolds
            flag = "test"
            if files_t != []:
                # iterate over json reply files
                for repl_file in files_t:
                    with open(os.path.join(path_repl, repl_file)) as f:
                        for line in f:
                            tw = json.loads(line)
                            tw["used"] = 0
                            tw["set"] = flag
                            tweets.append(tw)
                            if tw["text"] is None:
                                print("Tweet has no text", tw["id"])
                conversation["replies"] = tweets
                path_src = path + "/" + foldr + "/source-tweet"
                files_t = sorted(os.listdir(path_src))
                with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src["used"] = 0
                        scrcid = src["id_str"]
                        src["set"] = flag
                conversation["source"] = src
                if src["text"] is None:
                    print("Tweet has no text", src["id"])
                path_struct = path + "/" + foldr + "/structure.json"
                with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
                if len(struct) > 1:
                    new_struct = {}
                    new_struct[foldr] = struct[foldr]
                    struct = new_struct
                    # Take item from structure if key is same as source tweet id
                conversation["structure"] = struct
                branches = tree2branches(conversation["structure"])
                conversation["branches"] = branches
                train_dev_split[flag].append(conversation.copy())
                allconv.append(conversation.copy())
            # if no replies are present, still add just source
            else:
                flag = "test"
                path_src = path + "/" + foldr + "/source-tweet"
                files_t = sorted(os.listdir(path_src))
                with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src["used"] = 0
                        scrcid = src["id_str"]
                        src["set"] = flag
                conversation["source"] = src
                if src["text"] is None:
                    print("Tweet has no text", src["id"])
                path_struct = path + "/" + foldr + "/structure.json"
                with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
                if len(struct) > 1:
                    # print "Structure has more than one root"
                    new_struct = {}
                    new_struct[foldr] = struct[foldr]
                    struct = new_struct
                    # Take item from structure if key is same as source tweet id
                conversation["structure"] = struct
                branches = tree2branches(conversation["structure"])
                conversation["branches"] = branches
                train_dev_split[flag].append(conversation.copy())
                allconv.append(conversation.copy())
                print(foldr)
    return train_dev_split


#%%
def load_dataset():

    # Load labels and split for task A and task B
    tweet_label_dict, veracity_label_dict = load_true_labels()
    dev = tweet_label_dict["dev"]
    train = tweet_label_dict["train"]
    test = tweet_label_dict["test"]
    dev_tweets = dev.keys()
    train_tweets = train.keys()
    test_tweets = test.keys()
    # Load folds and conversations
    path_to_folds = "../data/train/twitter-english"
    folds = sorted(os.listdir(path_to_folds))
    newfolds = [i for i in folds if i[0] != "."]
    folds = newfolds
    cvfolds = {}
    allconv = []
    train_dev_split = {}
    train_dev_split["dev"] = []
    train_dev_split["train"] = []
    train_dev_split["test"] = []
    for nfold, fold in enumerate(folds):
        path_to_tweets = os.path.join(path_to_folds, fold)
        tweet_data = sorted(os.listdir(path_to_tweets))
        newfolds = [i for i in tweet_data if i[0] != "."]
        tweet_data = newfolds
        conversation = {}
        for foldr in tweet_data:
            flag = 0
            conversation["id"] = foldr
            tweets = []
            path_repl = path_to_tweets + "/" + foldr + "/replies"
            files_t = sorted(os.listdir(path_repl))
            newfolds = [i for i in files_t if i[0] != "."]
            files_t = newfolds
            if files_t != []:
                for repl_file in files_t:
                    with open(os.path.join(path_repl, repl_file)) as f:
                        for line in f:
                            tw = json.loads(line)
                            tw["used"] = 0
                            replyid = tw["id_str"]
                            if replyid in dev_tweets:
                                tw["set"] = "dev"
                                tw["label"] = dev[replyid]
                                #                        train_dev_tweets['dev'].append(tw)
                                if flag == "train":
                                    print("The tree is split between sets", foldr)
                                flag = "dev"
                            elif replyid in train_tweets:
                                tw["set"] = "train"
                                tw["label"] = train[replyid]
                                #                        train_dev_tweets['train'].append(tw)
                                if flag == "dev":
                                    print("The tree is split between sets", foldr)
                                flag = "train"
                            elif replyid in test_tweets:
                                tw["set"] = "test"
                                tw["label"] = train[replyid]
                                #                        train_dev_tweets['train'].append(tw)
                                if flag == "dev":
                                    print("The tree is split between sets", foldr)
                                flag = "test"
                            else:
                                print("Tweet was not found! ID: ", foldr)
                            tweets.append(tw)
                            if tw["text"] is None:
                                print("Tweet has no text", tw["id"])
                conversation["replies"] = tweets

                path_src = path_to_tweets + "/" + foldr + "/source-tweet"
                files_t = sorted(os.listdir(path_src))
                with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src["used"] = 0
                        scrcid = src["id_str"]
                        src["set"] = flag
                        src["label"] = tweet_label_dict[flag][scrcid]

                conversation["source"] = src
                conversation["veracity"] = veracity_label_dict[flag][scrcid]
                if src["text"] is None:
                    print("Tweet has no text", src["id"])
                path_struct = path_to_tweets + "/" + foldr + "/structure.json"
                with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
                if len(struct) > 1:
                    # I had to alter the structure of this conversation
                    if foldr == "553480082996879360":
                        new_struct = {}
                        new_struct[foldr] = struct[foldr]
                        new_struct[foldr]["553495625527209985"] = struct[
                            "553485679129534464"
                        ]["553495625527209985"]
                        new_struct[foldr]["553495937432432640"] = struct[
                            "553490097623269376"
                        ]["553495937432432640"]
                        struct = new_struct
                    else:
                        new_struct = {}
                        new_struct[foldr] = struct[foldr]
                        struct = new_struct
                    # Take item from structure if key is same as source tweet id
                conversation["structure"] = struct

                branches = tree2branches(conversation["structure"])
                conversation["branches"] = branches
                train_dev_split[flag].append(conversation.copy())
                allconv.append(conversation.copy())
            else:
                flag = "train"
                path_src = path_to_tweets + "/" + foldr + "/source-tweet"
                files_t = sorted(os.listdir(path_src))
                with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src["used"] = 0
                        scrcid = src["id_str"]
                        src["set"] = flag
                        src["label"] = tweet_label_dict[flag][scrcid]

                conversation["source"] = src
                conversation["veracity"] = veracity_label_dict[flag][scrcid]
                if src["text"] is None:
                    print("Tweet has no text", src["id"])

                path_struct = path_to_tweets + "/" + foldr + "/structure.json"
                with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
                if len(struct) > 1:
                    # print "Structure has more than one root"
                    new_struct = {}
                    new_struct[foldr] = struct[foldr]
                    struct = new_struct
                    # Take item from structure if key is same as source tweet id
                conversation["structure"] = struct
                branches = tree2branches(conversation["structure"])

                conversation["branches"] = branches
                train_dev_split[flag].append(conversation.copy())
                allconv.append(conversation.copy())

                print(foldr)

        cvfolds[fold] = allconv
        allconv = []

    return train_dev_split
