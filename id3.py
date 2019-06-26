#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
# Starter code provided by Daniel Lowd
#
#
import sys
import re
from math import log
# Node class for the decision tree
import node

train=None
varnames=None
test=None
testvarnames=None
root=None


# Helper function computes entropy of Bernoulli distribution with parameter p
def entropy(p):
    q = 1 - p
    if p > 0 and q > 0:
        entropy = (-p * log(p, 2.0)) - (q * log(q, 2.0))
    else:
        entropy = 0
    return entropy


# Partition data based on a given attribute index and a given value
def split(data, attribute):
    total = len(data[0]) - 1
    split = []
    branch_0 = []
    branch_1 = []
    for i in range(len(data)):
        if data[i][attribute] == 0:
            branch_0.append(data[i])
        else:
            branch_1.append(data[i])
    split.append(branch_0)
    split.append(branch_1)
    return split


def collect_counts(data, attribute):
# list[int] - collect counts for each variable value with each class label (where classification is positive)
# list[int] - collect counts for each variable value with each class label
# int - totals number of positive classifications
# int - totals the number of data entries

    py_pxi = [0, 0]
    pxi = [0, 0]
    py = 0
    total = len(data)
    class_index = len(data[0]) - 1

    for i in range(total):
        if data[i][class_index] == 1:
            py += 1
        if data[i][attribute] == 0:
            pxi[0] += 1
        if data[i][attribute] == 1:
            pxi[1] += 1
        if data[i][class_index] == 1 and data[i][attribute] == 0:
            py_pxi[0] += 1
        if data[i][class_index] == 1 and data[i][attribute] == 1:
            py_pxi[1] += 1

    return (py_pxi, pxi, py, total)



# Compute information gain for a particular split, given the counts
# py_pxi : number of occurrences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of occurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # Make py_pxi and pxi into arrays
    if isinstance(py_pxi, int):
        py_pxi = [py_pxi, py_pxi]
    if isinstance(pxi, int):
        pxi = [pxi, pxi]

    # Calculate base entropy
    base_prob = float(py) / total
    base_entropy = entropy(base_prob)

    # Calculate entropy for 0
    p0 = float(py_pxi[0]) / pxi[0] if pxi[0] else 0
    entropy0 = entropy(p0)

    # Calculate entropy for 1
    p1 = float(py_pxi[1]) / pxi[1] if pxi[1] else 0
    entropy1 = entropy(p1)

    # Calculate weights
    w0 = float(pxi[0]) / total
    w1 = float(pxi[1]) / total

    # Calculate information gain
    igain = base_entropy - ((w0 * entropy0) + (w1 * entropy1))
    return igain


# Find the best variable to split on, according to information gains
def best_split(data, varnames):
    features = len(varnames) - 1
    values = [0, 1]
    max_gain = 0.0
    feature = -1

    total = len(data)
    for i in range(features):

        # Check that the feature has not be used already
        if varnames[i] != "USED":
            # Calculate ig for all features in data
            py_pxi, pxi, py, total = collect_counts(data, i)
            for value in values:
                gain = infogain(py_pxi, pxi, py, total)

            # Update max_gain
            if (gain > max_gain):
                max_gain = gain
                feature = i
    # Return the index of the best splitter
    return feature

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the node class.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a pure leaf or all splits look bad.
def build_tree(data, varnames):

    # Check if data is empty
    if len(data) == 0:
        return node.Leaf(varnames, 0)


    # Check if leaf node,
    attribute = len(data[0]) - 1
    pc, tc, tp, t = collect_counts(data, attribute)
    # Check if all values are 0 or 1
    if tc[0] == 0:
        return node.Leaf(varnames, 1)
    if tc[1] == 0:
        return node.Leaf(varnames, 0)


    # Find best feature to split data on
    max_gain = best_split(data, varnames)
    #print(varnames[max_gain])

    # Split data based on best splitter
    newData = split(data, max_gain)
    left = newData[0]
    right = newData[1]

    # Update data/varnames
    # Change variable name in varnames to notify that that feature has already been used
    varnames[max_gain] == "USED"

    # Build left and right subtrees
    left_subtree = build_tree(left, varnames)
    right_subtree = build_tree(right, varnames)

    # Build tree
    root = node.Split(varnames, max_gain, left_subtree, right_subtree)

    # Return tree
    return root


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in the list is the class value.
def loadAndTrain(trainS,testS,modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with any helper functions needed.
    # It should return the root node of the decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)

def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct)/len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print 'Usage: id3.py <train> <test> <model>'
        sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2])

    acc = runTest()
    print "Accuracy: ",acc

if __name__ == "__main__":
    main(sys.argv[1:])
