
# ex15.py
# 2018/02/19 Yuma Saito

import numpy as np

docs = [["リンゴ", "リンゴ"], ["リンゴ", "レモン"], ["レモン", "ミカン"]]
terms = ["リンゴ", "レモン", "ミカン"]

def tf(term, doc):
    """
    単語term、文書docに対するtf値を計算する。
    tf(term, doc) = (文書内に現れる単語termの数) / (文書内の総単語数)
    """
    count = 0   # 上式の分子の値を計算
    for word in doc:
        if word == term:
            count += 1
    return count / len(doc)

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

def tf_idf(terms, docs):
    """
    tf_idf[i, j]が文書i、単語jのtf-idf値を表しているような
    tf-idf行列を計算する
    """
    # [docs, terms]サイズの行列を生成
    tfidf = np.zeros([len(docs), len(terms)])
    for i, doc in enumerate(docs):
        for j, term in enumerate(terms):
            tfidf[i, j] = tf(term, doc) * idf(term, docs)
    return tfidf

print(tf_idf(terms, docs))
