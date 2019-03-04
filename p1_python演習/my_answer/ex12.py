f = open('./dataset/data.txt', 'r')
docs = []
terms = []
for line in f:
    tmp = (line.strip("\n")).split("と")
    docs.append(tmp)
    for x in tmp:
        if x not in terms:
            terms.append(x)

print(docs)
print(terms)
f.close()
