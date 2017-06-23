import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfTransformer

X = []
y = []
dim = 300


def getFeature(comment):
    v = np.zeros(dim)
    count = 0
    words = comment.split()
    words = set(words)
    for word in words:
        if word in wv:
            v += wv[word]
            count += 1
    if count > 0:
        v /= count
    return v

wv = KeyedVectors.load_word2vec_format("word_vectors.txt", binary=False)

with open("input.txt") as f:
    for line in f:
        x = getFeature(line)
        X.append(x)

X = np.array(X)
with open("data") as f:
    for line in f:
        score = int((line.split())[0])
        if score >= 7:
            y.append(1)
        else:
            y.append(-1)
y = np.array(y)
