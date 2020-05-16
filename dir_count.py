import os
import numpy as np

num_posts = 0
num_replies = 0

max_replies = 0

source_ids = []
reply_ids = []

dir = "data/test/twitter-en-test-data"

for x in os.listdir(dir):
    for y in os.listdir(f"{dir}/{x}"):
        source_ids.append(y)
        num_posts += 1
        for z in os.listdir(f"{dir}/{x}/{y}"):
            if z == "replies":
                r = 0
                for a in os.listdir(f"{dir}/{x}/{y}/replies"):
                    r += 1
                    reply_ids.append(a[:-5])
                    num_replies += 1
            if r > max_replies:
                print(f"{dir}/{x}")
                max_replies = r

dir = "data/test/reddit-test-data"

for x in os.listdir(dir):
    num_posts += 1
    source_ids.append(x)
    for y in os.listdir(f"{dir}/{x}"):
        for z in os.listdir(f"{dir}/{x}/replies"):
            reply_ids.append(z[:-5])
            num_replies += 1
print(num_posts)
print(num_replies)
print(len(reply_ids))

u, c = np.unique(reply_ids, return_counts=True)

print(f"{np.max(c)}, {np.min(c)}")

print(len(set(source_ids)))
print(len(set(reply_ids)))
print(set(source_ids).intersection(reply_ids))

print(len(set(reply_ids)) + len(set(source_ids)))

print(max_replies)
