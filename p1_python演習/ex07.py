
# ex07.py
# 2018/02/19 Yuma Satio

# 長い文字列は"""で囲うことで複数行に跨がせることが可能
# 改行時に \ を入力すると改行コード(\n)が入らない
str1 = """Now I need a drink, alcoholic of course, \
after the heavy lectures involving quantum mechanics."""

# ","と"."をそれぞれ空文字列""に置換することで不要な記号文字を除去
str1_removed = str1.replace(",", "").replace(".", "")

# splitメソッドによりスペースで文字列を分割、単語リストを取得
str1_wordlist = str1_removed.split()

# 内包表記を利用して単語長リストを生成。
# len(シーケンス)で与えたシーケンスの長さ（要素数）を取得できる
str1_lenlist = [len(word) for word in str1_wordlist]
print(str1_lenlist)
