
# ex2-2.py
# numpyによるndarray単位での活性化関数実装
# 2018/02/20    Yuma Saito

# numpyのインポート。npでエイリアスしておくのが通例。
import numpy as np

# np.maxとかnp.argmaxとか紛らわしい関数があるので注意。
# np.max　　：与えたnumpy配列の中で"最大の要素"を取得
#   np.max([1, 2, 4, 8]) = 8
# np.argmax ：与えたnumpy配列の中で最大の要素となる"インデックス"を取得
#   np.argmax([1, 2, 4, 8]) = 3
# np.maximum："要素ごとに"与えた引数の中で最大のものを取得
#   np.maximum([1, 2, 3], [-1, 1, 4]) = [1, 2, 4]
def relu(x):
    return np.maximum(0, x)

def main():
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    print(relu(x))

if __name__ == "__main__":
    main()
