import os

num = 0

dir = "data/test/twitter-en-test-data"

for x in os.listdir(dir):
    print(x)
    for y in os.listdir(f"{dir}/{x}"):
        for z in os.listdir(f"{dir}/{x}/{y}"):
            if z == "replies":
                for a in os.listdir(f"{dir}/{x}/{y}/replies"):
                    num += 1

dir = "data/test/reddit-test-data"

for x in os.listdir(dir):
    print(x)
    for y in os.listdir(f"{dir}/{x}"):
        if y == "replies":
            for a in os.listdir(f"{dir}/{x}/replies"):
                num += 1

print(num)
