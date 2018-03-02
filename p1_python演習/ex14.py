
# ex14.py
# 2018/02/19 Yuma Saito

import numpy as np

docs = [["リンゴ", "リンゴ"], ["リンゴ", "レモン"], ["レモン", "ミカン"]]
terms = ["リンゴ", "レモン", "ミカン"]

def idf(term, docs):
    """
    単語term、文書リストdocsに対するidf値を計算する。
    idf(term, docs) = log{(総文書数) / (単語termが含まれる文書数)} + 1
    """
    count = 0
    for doc in docs:
        if term in doc:
            count += 1
    return np.log10(len(docs) / count) + 1.

for t in terms:
    print("idf({0}) = {1}".format(t, idf(t, docs)))
