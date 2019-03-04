import numpy as np
from ex15 import tf_idf
from ex16 import cosine_sim
docs = [['リンゴ', 'リンゴ', 'リンゴ'], ['リンゴ', 'レモン', 'レモン', 'ミカン'], ['リンゴ', 'イチゴ', 'ミカン'], ['レモン', 'イチゴ', 'ミカン'],['ミカン', 'ミカン', 'ブドウ', 'ブドウ']]

terms = ['リンゴ', 'レモン', 'ミカン', 'イチゴ', 'ブドウ']


matrix = np.zeros((len(docs), len(terms)))

for i in range(len(terms)):
    for j in range(len(docs)):
        matrix[j][i] = cosine_sim(tf_idf(terms, docs)[i], tf_idf(terms, docs)[j])
print(matrix)
