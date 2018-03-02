
# ex09.py
# 2018/02/19 Yuma Satio

# コマンドライン引数の処理に必要なsysモジュールをインポート
import sys
import random

# コマンドライン引数のリストを取得
args = sys.argv

# args[0]はプログラム自身の名前なのでargs[1]以降が与えた文字列になる
shuffled_list = []
for word in args[1:]:
    # 単語が三文字以上の場合
    if len(word) >= 3:
        # 与えたシーケンスの要素を要素の個数だけ非復元抽出。
        # 文字列の場合random.shuffleよりrandom.sampleの方が扱いやすいです
        inner_shuffled = random.sample(word[1:-1], len(word)-2)
        # inner_shuffledはリストなので、joinメソッドで文字列に連結
        shuffled = word[0] + "".join(inner_shuffled) + word[-1]
    else:
        shuffled = word
    shuffled_list.append(shuffled)

# 表示
print(" ".join(shuffled_list))
