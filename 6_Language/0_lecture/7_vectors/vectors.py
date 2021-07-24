from scipy.spatial.distance import cosine
import math
import numpy as np

# With Distributed Representation each word is represented by a vector with multiple values.
# Each vector value expresses a meaning of the word. Similarity between two words can be represented by
# how different the values in their vectors are.
# word2vec is an algorithm that calculates the vector of each word. It uses a neural network with
# the Skip-Gram Architecture. The input layer consists of nodes for multiple target words.
# It has 1 hidden layer with some number of nodes. The output layer has nodes for multiple context words
# which are words that are likely to occur in the context of the target words.
# The weights between an input node and each hidden node is the vector representation of the word.
# The neural network can be trained normally with backpropagation.

# Calculating the difference between 2 vectors, produces a vector that describes what separates one word from the other.
# This difference vector can be added to another vector to get to an area that contains equivalent words from
# the perspective of the other word. e.g. man and king have about the same difference vector than woman and queen.

# The file words.txt already contains the distributed representation of 50000 words.

# Usage: open the python interpreter in the console.
# >>> from vectors import *
# look at the vector representation of a word:
# >>> words["city"]
# look at the distance between 2 words (number between 0 and 1):
# >>> distance(words["japan"], words["anime"])
# look at the closest words of a word:
# >>> closest_words(words["restaurant"])
# get equivalent word by adding a difference vector:
# >>> closest_word(words["sausage"] - words["germany"] + words["japan"])

with open("words.txt") as f:
    words = dict()
    for i in range(50000):
        row = next(f).split()
        word = row[0]
        vector = np.array([float(x) for x in row[1:]])
        words[word] = vector


def distance(w1, w2):
    return cosine(w1, w2)


def closest_words(embedding):
    distances = {
        w: distance(embedding, words[w])
        for w in words
    }
    return sorted(distances, key=lambda w: distances[w])[:10]


def closest_word(embedding):
    return closest_words(embedding)[0]
