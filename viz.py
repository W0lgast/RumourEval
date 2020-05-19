"""
This will make some very cool visualisations for the write up.
"""
# -------------------------------------------------------------------------------------

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------------------

with open("data/train/train-key.json", "r") as f:
    TRAIN_DATA_LABELS = json.load(f)

with open("data/train/dev-key.json", "r") as f:
    VAL_DATA_LABELS = json.load(f)

with open("data/test/final-eval-key.json", "r") as f:
    TEST_DATA_LABELS = json.load(f)

# -------------------------------------------------------------------------------------

train_a_comment = len([v for v in TRAIN_DATA_LABELS['subtaskaenglish'].values() if v=="comment"])
train_a_support = len([v for v in TRAIN_DATA_LABELS['subtaskaenglish'].values() if v=="support"])
train_a_query = len([v for v in TRAIN_DATA_LABELS['subtaskaenglish'].values() if v=="query"])
train_a_deny = len([v for v in TRAIN_DATA_LABELS['subtaskaenglish'].values() if v=="deny"])

val_a_comment = len([v for v in VAL_DATA_LABELS['subtaskaenglish'].values() if v=="comment"])
val_a_support = len([v for v in VAL_DATA_LABELS['subtaskaenglish'].values() if v=="support"])
val_a_query = len([v for v in VAL_DATA_LABELS['subtaskaenglish'].values() if v=="query"])
val_a_deny = len([v for v in VAL_DATA_LABELS['subtaskaenglish'].values() if v=="deny"])

test_a_comment = len([v for v in TEST_DATA_LABELS['subtaskaenglish'].values() if v=="comment"])
test_a_support = len([v for v in TEST_DATA_LABELS['subtaskaenglish'].values() if v=="support"])
test_a_query = len([v for v in TEST_DATA_LABELS['subtaskaenglish'].values() if v=="query"])
test_a_deny = len([v for v in TEST_DATA_LABELS['subtaskaenglish'].values() if v=="deny"])

train_a_true = len([v for v in TRAIN_DATA_LABELS['subtaskbenglish'].values() if v=="true"])
train_a_false = len([v for v in TRAIN_DATA_LABELS['subtaskbenglish'].values() if v=="false"])
train_a_unvalidated = len([v for v in TRAIN_DATA_LABELS['subtaskbenglish'].values() if v=="unverified"])

val_a_true = len([v for v in VAL_DATA_LABELS['subtaskbenglish'].values() if v=="true"])
val_a_false = len([v for v in VAL_DATA_LABELS['subtaskbenglish'].values() if v=="false"])
val_a_unvalidated = len([v for v in VAL_DATA_LABELS['subtaskbenglish'].values() if v=="unverified"])

test_a_true = len([v for v in TEST_DATA_LABELS['subtaskbenglish'].values() if v=="true"])
test_a_false = len([v for v in TEST_DATA_LABELS['subtaskbenglish'].values() if v=="false"])
test_a_unvalidated = len([v for v in TEST_DATA_LABELS['subtaskbenglish'].values() if v=="unverified"])

def plot_a_class_dist():
    # set width of bar
    barWidth = 0.2
    # set height of bar
    bars1 = [train_a_comment, val_a_comment, test_a_comment]
    bars2 = [train_a_support, val_a_support, test_a_support]
    bars3 = [train_a_query, val_a_query, test_a_query]
    bars4 = [train_a_deny, val_a_deny, test_a_deny]
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    # Make the plot
    plt.bar(r1, bars1, color='#AD33FF', width=barWidth, edgecolor='white', label='comment')
    plt.bar(r2, bars2, color='#3362FF', width=barWidth, edgecolor='white', label='support')
    plt.bar(r3, bars3, color='#33FF99', width=barWidth, edgecolor='white', label='query')
    plt.bar(r4, bars4, color='#FFE733', width=barWidth, edgecolor='white', label='deny')
    # Add xticks on the middle of the group bars
    plt.xlabel('Dataset', fontweight='bold')
    plt.xticks([r + barWidth+0.1 for r in range(len(bars1))], ['Training', 'Validation', 'Testing'])
    # Create legend & Show graphic
    plt.legend()
    plt.savefig("figures/task_a_class_distibution.png")
    plt.show()


def plot_b_class_dist():
    # set width of bar
    barWidth = 0.25
    # set height of bar
    bars1 = [train_a_true, val_a_true, test_a_true]
    bars2 = [train_a_false, val_a_false, test_a_false]
    bars3 = [train_a_unvalidated, val_a_unvalidated, test_a_unvalidated]
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # Make the plot
    plt.bar(r1, bars1, color='#33FF7C', width=barWidth, edgecolor='white', label='true')
    plt.bar(r2, bars2, color='#FF9D33', width=barWidth, edgecolor='white', label='false')
    plt.bar(r3, bars3, color='#339DFF', width=barWidth, edgecolor='white', label='unverified')
    # Add xticks on the middle of the group bars
    plt.xlabel('Dataset', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Training', 'Validation', 'Testing'])
    # Create legend & Show graphic
    plt.legend()
    plt.savefig("figures/task_b_class_distibution.png")
    plt.show()

plot_a_class_dist()
plot_b_class_dist()