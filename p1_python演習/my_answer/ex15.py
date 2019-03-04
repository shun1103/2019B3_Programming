import numpy as np
from ex13 import tf
from ex14 import idf
docs = [["リンゴ", "リンゴ"], ["リンゴ", "レモン"], ["レモン", "ミカン"]]
terms = ["リンゴ", "レモン", "ミカン"]

def tf_idf(terms, docs):
    matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            matrix[j][i] = tf(terms[i], docs[j]) * idf(terms[i], docs)
    return matrix

print(tf_idf(terms, docs))
