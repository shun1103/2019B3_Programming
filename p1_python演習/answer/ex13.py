
# ex13.py
# 2018/02/19 Yuma Saito

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

for doc in docs:
    for t in terms:
        print("tf({0}, {1}) = {2}\t".format(t, doc, tf(t, doc)), end="")
    print()
