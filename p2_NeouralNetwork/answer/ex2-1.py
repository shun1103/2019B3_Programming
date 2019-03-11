
# ex2-1.py
# numpyによるndarray単位での活性化関数実装
# 2018/02/20    Yuma Saito

# numpyのインポート。npでエイリアスしておくのが通例。
import numpy as np

# np.***の形の数学関数はデフォルトでnumpy配列に対し要素ごとの計算を行います
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    print(sigmoid(x))

if __name__ == "__main__":
    main()
