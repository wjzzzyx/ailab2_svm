import os
import string
import numpy as np
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

n = 0
X = []
y = []

class MySentences():
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        l = os.listdir(self.dirname)
        l.sort(key=lambda x:int(x.split('_')[0]))
        for fname in l:
            with open(os.path.join(self.dirname, fname)) as f:
                for line in f:
                    line = line.translate(None, string.punctuation)
                    yield line.split()


def getFeature(comment, wv):
    fv = np.zeros(wv.vector_size)
    for w in comment:
        fv += wv[w]
    return fv / len(comment)


dirname = "/home/zyx/PycharmProjects/original"
try:
    model = Word2Vec.load("features.model")
except:
    sentences = MySentences(dirname)
    model = Word2Vec(sentences, workers=4)
    model.save("features.model")
finally:
    word_vectors = model.wv
    del model

for fname in os.listdir(dirname):
    with open(os.path.join(dirname, fname)) as f:
        for line in f:
            line = line.translate(None, string.punctuation)
            words = line.split()
            x = getFeature(words, word_vectors)
            X.append(x)

X = np.array(X)
with open("data", "r") as f:
    for line in f:
        record = [x for x in line.split(' ')]
        if int(record[0]) >= 7:
            y.append(1)
        else:
            y.append(-1)
        n += 1
y = np.array(y)