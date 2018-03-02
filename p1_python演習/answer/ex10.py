
# ex10.py
# 2018/02/19 Yuma Saito

# コマンドライン引数の処理に必要なsysモジュールをインポート
import sys

def n_gram(seq, n):
    """
    与えたシーケンスのn-gramをリスト形式で取得
    """
    n_gram = []
    for i in range(len(seq) - n + 1):
        n_gram.append(seq[i:i+n])
    return n_gram

# 内包表記を応用したワンライナー実装
# def n_gram(seq, n):
#     return [seq[i:i+n] for i in range(len(seq) - n + 1)]

args = sys.argv
word_seq = args[1:]          # 単語のリスト
char_seq = "".join(word_seq) # 文字のリスト（文字列）
print("単語bi-gram: {0}".format(n_gram(word_seq, 2)))
print("文字bi-gram: {0}".format(n_gram(char_seq, 2)))
