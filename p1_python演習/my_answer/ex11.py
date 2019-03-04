f = open('./dataset/data.txt', 'r', encoding="shiftJIS")

for line in f:
    print(line)
f.close()
