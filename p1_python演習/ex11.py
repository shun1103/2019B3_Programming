
# ex11.py
# 2018/02/19 Yuma Saito

# ファイルの読み込み
with open("dataset/data.txt", "r") as f:
    for row in f:
        # 各文書を表示（改行はすでにrowに含まれているので end="" を指定）
        print(row, end="")
