
# ex17.py
# 2018/02/19 Yuma Saito

# モジュールのインポート
import numpy as np
from nlp.ex15 import tf_idf
from nlp.ex16 import cosine_sim

docs = []   # 各文書の単語リスト
terms = []  # 文書全体に現れる単語リスト

# ファイルの読み込み
with open("dataset/data.txt", "r") as f:
    for row in f:
        # 各文書を形態素解析（改行コードを除去して"と"で分割）
        wordlist = row.replace("\n", "").split("と")
        docs.append(wordlist)

# termsリストの作成
# まずdocsを一次元リストにflattenして重複ありの単語リストを取得
for doc in docs:
    terms += doc
# 次にいったん集合(set)に変換しすぐリストに戻すことで要素の重複を削除
terms = list(set(terms))

# tf_idf値の行列を取得
tfidf_array = tf_idf(terms, docs)
cosine_array = np.zeros([len(docs), len(docs)])
for doc1 in range(len(docs)):
    for doc2 in range(len(docs)):
        cosine_array[doc1, doc2] = cosine_sim(
            tfidf_array[doc1], tfidf_array[doc2])
print(cosine_array)
