import random
import sys

args = sys.argv


import random


def typoglycemia(text):

    def random_word(word):
        if len(word) < 4:
            # 4以下はそのまま
            return word
        # 先頭と末尾以外をランダムに並び替え
        arr = list(word[1:-1])
        random.shuffle(arr)
        # 先頭と末尾を加え返す。
        return word[0] + "".join(arr) + word[-1]

    return " ".join(list(map(random_word, text.split())))


text = args[1]
for i in range(2, len(args)):
    text = text + " " + args[i]
print(text)
print(typoglycemia(text))
