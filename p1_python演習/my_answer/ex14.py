import numpy as np
docs = [["リンゴ", "リンゴ"], ["リンゴ", "レモン"], ["レモン", "ミカン"]]
terms = ["リンゴ", "レモン", "ミカン"]

def idf(term, docs):
    bunbo = 0
    bunshi = len(docs)
    for doc in docs:
        if term in doc:
            bunbo += 1
    
    return np.log10(bunshi / bunbo) + 1

for x in terms:
    print("tf(%s) = %f" % (x, idf(x, docs)))
