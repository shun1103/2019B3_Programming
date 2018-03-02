
# ex06.py
# 2018/02/19 Yuma Satio

# ex05の処理はリスト内包表記でより簡潔かつ高速に実装できる。
# リスト内包表記は[式 for 変数 in リスト if 条件] の形で記述。
# 関数型言語のmapやfilterに相当する処理が可能
square_list = [i ** 2 for i in range(1, 11)]
print(square_list)
