# RumourEval
Repo for SemEval 2019 Task 7: https://competitions.codalab.org/competitions/19938 

<h3>Task A (SDQC)<h3>


Related to the objective of predicting a rumour's veracity, the first subtask will deal with the complementary objective of tracking how other sources orient to the accuracy of the rumourous story. A key step in the analysis of the surrounding discourse is to determine how other users in social media regard the rumour. We propose to tackle this analysis by looking at the replies to the post that presented the rumourous statement, i.e. the originating rumourous (source) post. We will provide participants with a tree-structured conversation formed of posts replying to the originating rumourous post, where each post presents its own type of support with regard to the rumour. We frame this in terms of supporting, denying, querying or commenting on (SDQC) the claim. Therefore, we introduce a subtask where the goal is to label the type of interaction between a given statement (rumourous post) and a reply post (the latter can be either direct or nested replies). Each tweet in the tree-structured thread will have to be categorised into one of the following four categories:

- Support: the author of the response supports the veracity of the rumour they are responding to.

- Deny: the author of the response denies the veracity of the rumour they are responding to.

- Query: the author of the response asks for additional evidence in relation to the veracity of the rumour they are responding to.

- Comment: the author of the response makes their own comment without a clear contribution to assessing the veracity of the rumour they are responding to.

<h3>Task B (verification)<h3>


The goal of the second subtask is to predict the veracity of a given rumour. The rumour is presented as a post reporting or querying a claim but deemed unsubstantiated at the time of release. Given such a claim, and a set of other resources provided, systems should return a label describing the anticipated veracity of the rumour as true or false. The ground truth of this task is manually established by journalist and expert members of the team who identify official statements or other trustworthy sources of evidence that resolve the veracity of the given rumour. Additional context will be provided as input to veracity prediction systems; this context will consist of snapshots of relevant sources retrieved immediately before the rumour was reported, including a snapshot of an associated Wikipedia article, a Wikipedia dump, news articles from digital news outlets retrieved from NewsDiffs, as well as preceding tweets from the same event. Critically, no external resources may be used that contain information from after the rumour's resolution. To control this, we will specify precise versions of external information that participants may use. This is important to make sure we introduce time sensitivity into the task of veracity prediction. We take a simple approach to this task, using only true/false labels for rumours. In practice, however, many claims are hard to verify; for example, there were many rumours concerning Vladimir Putin's activities in early 2015, many wholly unsubstantiable. Therefore, we also expect systems to return a confidence value in the range of 0-1 for each rumour; if the rumour is unverifiable, a confidence of 0 should be returned.
