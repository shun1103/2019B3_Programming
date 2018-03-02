
# ex12.py
# 2018/02/19 Yuma Saito

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

print("docs : {0}".format(docs))
print("terms: {0}".format(terms))
