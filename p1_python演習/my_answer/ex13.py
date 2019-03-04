docs = [["リンゴ", "リンゴ"], ["リンゴ", "レモン"], ["レモン", "ミカン"]]
terms = ["リンゴ", "レモン", "ミカン"]

def tf(term, doc):
    bunshi = 0
    bunbo = 0
    for x in doc:
        if x == term:
            bunshi += 1
        for y in terms:
            if x == y:
                bunbo += 1
    return bunshi / bunbo

for x in terms:
    for y in docs:
        print("tf(%s, %s) = %f" % (x, y ,tf(x, y)))
