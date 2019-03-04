str = "Hi He Lead Because Boron Could Not Oxidize Flourine. New Nations Might Also Sign Peace Security Clause.Arthur King Can."
tmp = str.split()
words = [x.strip(".") for x in tmp]
dict = {}
for i in range(1, len(words)):
    if i == 1 or i == 5 or i == 6 or i == 7 or i == 8 or i == 9 or i == 15 or i == 16 or i == 19:
        dict[words[i][0:1]] = i
    else:
        dict[words[i][0:2]] = i
print(dict)
