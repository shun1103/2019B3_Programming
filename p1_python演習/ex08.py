
# ex08.py
# 2018/02/19 Yuma Satio

# 長い文字列は"""で囲うことで複数行に跨がせることが可能
# 改行時に \ を入力すると改行コード(\n)が入らない
str1 = """Hi He Lead Because Boron Could Not Oxidize Flourine. \
New Nations Might Also Sign Peace Security Clause. Arthur King Can."""
# 先頭1文字のみを取得するインデックスをまとめたリスト
one_indices = [1, 5, 6, 7, 8, 9, 15, 16, 19]

# splitメソッドによりスペースで文字列を分割、単語リストを取得
wordlist = str1.split()

worddict = {}   # 空のディクショナリを作成
for i in range(20):
    # pythonではinを条件文に用いることで包含関係による条件判定が可能
    if i + 1 in one_indices:
        # 先頭1文字をキーとして単語の位置をworddictに格納
        worddict[wordlist[i][0]] = i + 1
    else:
        # 先頭2文字をキーとして単語の位置をworddictに格納
        worddict[wordlist[i][0:2]] = i + 1

# ディクショナリの中身の順番はランダムで追加した通りにはならない。
# 追加した順になってほしい場合はOrderedDictを使用する。
print(worddict)
