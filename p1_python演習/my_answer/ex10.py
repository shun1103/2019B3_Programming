import sys
import itertools
args = sys.argv
bi_gram1 =[]
for i in range(2, len(args)):
    bi_gram1.append([args[i - 1], args[i]])
print(bi_gram1)

bi_gram2 = []
str = ""
for i in range(1, len(args)):
    str += args[i]
for i in range(1, len(str)):
    bi_gram2.append(str[i - 1] + str[i])
print(bi_gram2)
