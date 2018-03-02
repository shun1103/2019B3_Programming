
# ex3-4.py
# Affineクラス
# Yuma Saito

import numpy as np

# ex3-4: Affineノード
# W, bをコンストラクタで与えるのでなくforwardでその都度与える形式でも良いです
class Affine:
    def __init__(self, W, b):
        self.W = W  # shape = [入力ユニット数、出力ユニット数]
        self.b = b  # shape = [出力ユニット数]
        self.x = None   # dW, dbの計算に必要。順伝播の入力
        self.dW = None  # dW, dbは入力側に逆伝播されることはないが、
        self.db = None  # 次の演習でSGDを実装する際に必要になる
    def forward(self, x):
        """ 順伝播計算：z = xW + b """
        # shape(x) = [バッチサイズ、入力ユニット数]
        self.x = x
        return np.dot(x, self.W) + self.b
    def backprop(self, dz):
        """ 逆伝播計算
            shape(dL/dz) = [バッチサイズ、出力ユニット数]
            dL/dx = dL/dz・W^T   shape = [バッチサイズ、入力ユニット数]
            dL/dW = x^T・dL/dz   shape = [入力ユニット数、出力ユニット数]
            dL/db = dL/dzをバッチ方向に総和をとったもの shape = [出力ユニット数]
        """
        self.dW = np.dot(self.x.T, dz)
        self.db = np.sum(dz, axis=0)
        return np.dot(dz, self.W.T)

def main():
    x = np.array([[1.0, 0.5], [-0.4, 0.1]])
    W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b = np.array([0.1, 0.2, 0.3])
    affine = Affine(W, b)
    print("順伝播出力: \n{0}".format(affine.forward(x)))
    print("逆伝播出力dx: \n{0}".format(affine.backprop(np.ones([2, 3]))))
    print("逆伝播出力dw: \n{0}".format(affine.dW))
    print("逆伝播出力db: \n{0}".format(affine.db))

if __name__ == "__main__":
    main()
