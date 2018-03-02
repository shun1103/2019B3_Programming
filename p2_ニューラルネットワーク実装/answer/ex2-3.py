
# ex2-3.py
# SingleLayerクラスの実装
# 2018/02/20    Yuma Saito

# numpyのインポート。npでエイリアスしておくのが通例。
import numpy as np

# 活性化関数の定義
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)

# SingleLayerクラス
class SingleLayer:
    def __init__(self, W, b):
        """ コンストラクタ：パラメータの初期化 """
        self.W = W  # 重み行列 今回は[2, 3]
        self.b = b  # バイアス [3]

    def forward(self, x):
        """ 順伝播計算 入力x[2] -> 出力y[3]"""
        # 齋藤の個人的な意見ですが、行列関連の処理を実装するときは
        # 常にshape（行列の各軸のサイズ）を意識すると良いです
        # shape(x) = [2], shape(W) = [2, 3]なので、
        # np.dot(self.W, x)ではshapeが合わずエラーが出ます。
        # そのためWを転置させた(self.W.T, x)や順番を逆にした(x, self.W)が正解です
        affine = np.dot(x, self.W) + self.b
        return relu(affine)

def main():
    x = np.array([1.0, 0.5])
    W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b = np.array([0.1, 0.2, 0.3])
    layer1 = SingleLayer(W, b)
    print(layer1.forward(x))

if __name__ == "__main__":
    main()
