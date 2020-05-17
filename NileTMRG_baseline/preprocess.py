import json
import os
import numpy as np
import json
import gensim
import nltk
import re
from nltk.corpus import stopwords
from copy import deepcopy
import pickle
from nltk.tokenize import TweetTokenizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing_tweets import load_dataset, load_true_labels
from preprocessing_reddit import load_data

#%%
def get_features(all_data, whichset="training"):

    tknzr = TweetTokenizer(reduce_len=True)

    # Can join train and dev

    if whichset == "training":
        training_set = all_data["train"]

    elif whichset == "development":
        training_set = all_data["dev"]

    elif whichset == "testing":
        training_set = all_data["test"]

    elif whichset == "training+development":
        training_set = all_data["train"] + all_data["dev"]

    BOW_sents = []
    all_extra_feats = []
    Y = []
    ids = []
    for conversation in training_set:

        # work with source tweet
        tw = conversation["source"]
        tw_text = conversation["source"]["text"]
        # Preprocess
        # remove stop-words
        # remove 'rt' and 'via'
        # remove punctuation
        words = tknzr.tokenize(tw_text)
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        twitter_stops = ["rt", "via", "RT", "Via", "VIA"]
        words = [w for w in words if not w in twitter_stops]
        punct = [
            "!",
            "@",
            "£",
            "$",
            "%",
            "^",
            "&",
            "*",
            "(",
            ")",
            "<",
            ">",
            "?",
            ",",
            ".",
            "/",
            "{",
            "}",
            "#",
        ]
        words = [w for w in words if not w in punct]
        sent = ""
        for w in words:
            sent = sent + w + " "
        # Extract features
        # add to list to later put in BOW
        BOW_sents.append(sent)
        ids.append(conversation["id"])
        # extract other feats and store them for later

        hashash = 0
        hasurl = 0

        if "entities" in list(tw.keys()):

            if tw["entities"]["hashtags"] != []:
                hashash = 1

            if tw["entities"]["urls"] != []:
                hasurl = 1

        s = 0
        d = 0
        q = 0

        if whichset == "testing":

            submission_file = (
                #"../data/test/final-eval-key.json" # these are the true answers
                "../tempAnswers.json"
            )
            submission_full = json.load(open(submission_file, "r"))
            submission = submission_full["subtaskaenglish"]

            if submission[tw["id_str"]] == "support":
                s = s + 1
            elif submission[tw["id_str"]] == "deny":
                d = d + 1
            elif submission[tw["id_str"]] == "query":
                q = q + 1

            for repl in conversation["replies"]:
                if submission[repl["id_str"]] == "support":
                    s = s + 1
                elif submission[repl["id_str"]] == "deny":
                    d = d + 1
                elif submission[repl["id_str"]] == "query":
                    q = q + 1

            #kmf: addition here, might be an error
            conversation['veracity'] = submission_full["subtaskbenglish"][conversation['id']]

        elif whichset == "development":

            submission_file = "stance_answer_dev.json"  # insert file with predictions of stance labels for dev set here
            submission_full = json.load(open(submission_file, "r"))
            submission = submission_full["subtaskaenglish"]

            if submission[tw["id_str"]] == "support":
                s = s + 1
            elif submission[tw["id_str"]] == "deny":
                d = d + 1
            elif submission[tw["id_str"]] == "query":
                q = q + 1

            for repl in conversation["replies"]:
                if submission[repl["id_str"]] == "support":
                    s = s + 1
                elif submission[repl["id_str"]] == "deny":
                    d = d + 1
                elif submission[repl["id_str"]] == "query":
                    q = q + 1

        Y.append(conversation["veracity"])

        tweet_label_dict, _ = load_true_labels()

        stance_labels = tweet_label_dict["train"]

        stance_labels.update(tweet_label_dict["dev"])

        stance_labels.update(tweet_label_dict["test"])

        if stance_labels[tw["id_str"]] == "support":
            s = s + 1
        elif stance_labels[tw["id_str"]] == "deny":
            d = d + 1
        elif stance_labels[tw["id_str"]] == "query":
            q = q + 1

        for repl in conversation["replies"]:
            if stance_labels[repl["id_str"]] == "support":
                s = s + 1
            elif stance_labels[repl["id_str"]] == "deny":
                d = d + 1
            elif stance_labels[repl["id_str"]] == "query":
                q = q + 1

        ntweets = len(conversation["replies"]) + 1
        support_stanceratio = float(s) / ntweets
        deny_stanceratio = float(d) / ntweets
        question_stanceratio = float(q) / ntweets

        #### loop through replies, find number of upvotes of support, deny, query
        if 'favorite_count' in conversation['source'].keys():
            fav_count_source = conversation['source']['favorite_count']
        else:
            fav_count_source = conversation['source']['data']['children'][0]['data']['score']
        comment_up_count = 0
        query_up_count = 0
        support_up_count = 0
        deny_up_count = 0
        for reply in conversation['replies']:
            if 'id' in reply.keys():
                key = 'id'
            else: key = 'id_str'
            reply_stance = stance_labels[str(reply[key])]

            if reply_stance == "comment":
                if 'favorite_count' in reply.keys():
                    comment_up_count += reply['favorite_count']
                elif 'score' in reply['data'].keys():
                    comment_up_count += reply['data']['score']
            elif reply_stance == "query":
                if 'favorite_count' in reply.keys():
                    query_up_count += reply['favorite_count']
                elif 'score' in reply['data'].keys():
                    query_up_count += reply['data']['score']
            elif reply_stance == "support":
                if 'favorite_count' in reply.keys():
                    support_up_count += reply['favorite_count']
                elif 'score' in reply['data'].keys():
                    support_up_count += reply['data']['score']
            elif reply_stance == "deny":
                if 'favorite_count' in reply.keys():
                    deny_up_count += reply['favorite_count']
                elif 'score' in reply['data'].keys():
                    deny_up_count += reply['data']['score']
            else:
                print("UNKNOWN STANCE!")
                exit(0)

            comment_up_count = max(comment_up_count/max(fav_count_source,1),0)
            query_up_count = max(query_up_count/max(fav_count_source,1),0)
            deny_up_count = deny_up_count/max(fav_count_source,1)
            support_up_count = support_up_count/max(fav_count_source,1)
            if deny_up_count<0 and support_up_count<0:
                deny_up_count = 0
                support_up_count = 0
            elif deny_up_count < 0:
                support_up_count = support_up_count - deny_up_count
                deny_up_count = 0
            elif support_up_count < 0:
                deny_up_count = deny_up_count - support_up_count
                support_up_count = 0

            num_replies = len(conversation['replies'])/max(fav_count_source,1)

        extra_feats = [
            hashash,
            hasurl,
            support_stanceratio,
            deny_stanceratio,
            question_stanceratio,
            #comment_up_count,
            #query_up_count,
            #deny_up_count,
            #support_up_count,
            #num_replies
        ]

        all_extra_feats.append(extra_feats)

    return BOW_sents, all_extra_feats, Y, ids


#%%


def read_data():

    data = load_dataset()
    reddit = load_data()

    data["train"].extend(reddit["train"])
    data["dev"].extend(reddit["dev"])
    data["test"].extend(reddit["test"])

    return data
