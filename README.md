# RumourEval

Repo for SemEval 2019 Task 7: <https://competitions.codalab.org/competitions/19938>

## Task A (SDQC)

Related to the objective of predicting a rumour's veracity, the first subtask will deal with the complementary objective of tracking how other sources orient to the accuracy of the rumourous story. A key step in the analysis of the surrounding discourse is to determine how other users in social media regard the rumour. We propose to tackle this analysis by looking at the replies to the post that presented the rumourous statement, i.e. the originating rumourous (source) post. We will provide participants with a tree-structured conversation formed of posts replying to the originating rumourous post, where each post presents its own type of support with regard to the rumour. We frame this in terms of supporting, denying, querying or commenting on (SDQC) the claim. Therefore, we introduce a subtask where the goal is to label the type of interaction between a given statement (rumourous post) and a reply post (the latter can be either direct or nested replies). Each tweet in the tree-structured thread will have to be categorised into one of the following four categories:

-   Support: the author of the response supports the veracity of the rumour they are responding to.

-   Deny: the author of the response denies the veracity of the rumour they are responding to.

-   Query: the author of the response asks for additional evidence in relation to the veracity of the rumour they are responding to.

-   Comment: the author of the response makes their own comment without a clear contribution to assessing the veracity of the rumour they are responding to.

## Task B (verification)

The goal of the second subtask is to predict the veracity of a given rumour. The rumour is presented as a post reporting or querying a claim but deemed unsubstantiated at the time of release. Given such a claim, and a set of other resources provided, systems should return a label describing the anticipated veracity of the rumour as true or false. The ground truth of this task is manually established by journalist and expert members of the team who identify official statements or other trustworthy sources of evidence that resolve the veracity of the given rumour. Additional context will be provided as input to veracity prediction systems; this context will consist of snapshots of relevant sources retrieved immediately before the rumour was reported, including a snapshot of an associated Wikipedia article, a Wikipedia dump, news articles from digital news outlets retrieved from NewsDiffs, as well as preceding tweets from the same event. Critically, no external resources may be used that contain information from after the rumour's resolution. To control this, we will specify precise versions of external information that participants may use. This is important to make sure we introduce time sensitivity into the task of veracity prediction. We take a simple approach to this task, using only true/false labels for rumours. In practice, however, many claims are hard to verify; for example, there were many rumours concerning Vladimir Putin's activities in early 2015, many wholly unsubstantiable. Therefore, we also expect systems to return a confidence value in the range of 0-1 for each rumour; if the rumour is unverifiable, a confidence of 0 should be returned.

# Instructions from Competiton Page

## RumourEval 2019 Data

This is the data for the competition. It is to be used responsibly.

-   Training/development data with key: rumoureval-2019-training-data.zip
-   Scoring script for home use: home_scorer_macro.py
-   Test data: rumoureval-2019-test-data.zip
-   Gold test key: final-eval-key.json

## Distribution and license

This data is distributed as CC-BY (see LICENSE file) and under the Twitter license.

If you use the data, you must cite the following work:

Genevieve Gorrell, Ahmet Aker, Kalina Bontcheva, Elena Kochkina, Maria Liakata, Arkaitz Zubiaga, Leon Derczynski (2019). SemEval-2019 Task 7: RumourEval, Determining Rumour Veracity and Support for Rumours. Proceedings of the 13th International Workshop on Semantic Evaluation, ACL.

BibTeX follows at the end of this readme.

## Description

The trial data contains the full English Twitter and Reddit training/dev set.

In both Twitter and Reddit cases, the original post and each comment post are provided separately, with a key indicating the tree structure of the comment. Reddit JSON has a nested format, but comments have been separated out to mimic the structure of the Twitter data as closely as possible. The original raw JSON is also provided.

You can use the scorer over the development data to validate your system. Final submissions are to be made on the CodaLab site.

## Source data

The bulk of the dataset used for training and development testing is from this PLoS article:

Zubiaga A, Liakata M, Procter R, Wong Sak Hoi G, Tolmie P (2016) Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads. PLoS ONE 11(3): e0150989. doi:10.1371/journal.pone.0150989
The original data can be found here:

PHEME rumour scheme dataset: journalism use case, version 2
Additional Twitter and Reddit data has been added using the same methodology.

For task B, the following wikipedia dump may be used:

20160901: <https://archive.org/details/enwiki-20160901/>

## Reference

BibTeX for reference for CC-BY Attribution:

@inproceedings{gorrell-etal-2019-semeval,
    title = "{S}em{E}val-2019 Task 7: {R}umour{E}val, {D}etermining Rumour Veracity and Support for Rumours",
    author = "Gorrell, Genevieve  and
      Aker, Ahmet  and
      Bontcheva, Kalina and
      Kochkina, Elena and
      Liakata, Maria and
      Zubiaga, Arkaitz and
      Derczynski, Leon",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "<https://www.aclweb.org/anthology/S19-2147">,
    pages = "845--854"
}

# Instructions from original repo

## RumourEval2019 Baselines for Task A and  Task B

## Prerequisites

Keras '2.0.8'

Hyperopt '0.1'

## Preprocessing (for both tasks)

1.  Download the data from competition Codalab.

<https://competitions.codalab.org/competitions/19938>

2.  Download 300d word vectors pre-trained on Google News corpus.

<https://code.google.com/archive/p/word2vec/>

3.  Change filepaths for data and for word embeddings if needed:

in `help_prep_functions.py` in `loadW2vModel()` function insert filepath for word embeddings

in `preprocessing_tweets.py` and `preprocessing_reddit.py` change filepaths for data if needed.

4.  Choose features option:

In `prep_pipeline.py` on line 98:

`def main(data ='RumEval2019', feats = 'SemEvalfeatures')`

feats can be either `text` for avgw2v representation of the tweets or `SemEvalfeatures` for additional extra features concatenated with avgw2v.

5.  Run preprocessing script


    python prep_pipeline.py

## Running the model

The description of the model architecture can be found in <https://www.aclweb.org/anthology/S/S17/S17-2083.pdf>
The features used in this code are different to the ones used in the paper.

1.  In `outer_semeval2019.py` you can choose the number of trials that the search algorithm performs while searching for the parameter combination.

2.  In `parameter_search.py` you can define search_space.

3.  Run the baseline


    python outer_semeval2019.py

If you have any questions feel free to contact me E.Kochkina@warwick.ac.uk or other task organisers rumoureval-2019-organizers@googlegroups.com
