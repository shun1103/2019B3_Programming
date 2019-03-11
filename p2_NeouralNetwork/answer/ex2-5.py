
# ex2-4.py
# MLP_3Layerクラスの実装
# 2018/02/20    Yuma Saito

# numpyのインポート。npでエイリアスしておくのが通例。
import numpy as np

# 活性化関数の定義
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    # xは[バッチサイズ, 入力次元数]のshapeになっているという前提で処理しています
    # sumやmaxなどのreduce系の演算はaxisでreduceの対象とする軸を指定できます。
    # numpyやtensorflowだとaxisは0オリジンになることに注意
    # keepdims=Trueとすることでreduceの前後で次元数が減らないようにできます
    # これを指定し忘れるとブロードキャストができないと怒られます
    logit = x - np.max(x, axis=1, keepdims=True) # 各入力ベクトルをその要素の最大値で引く
    exp_x = np.exp(logit)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x

# softmax層を実装するには新たにSoftmaxLayerクラスを定義する方法もありますが
# 今回はSingleLayerのコンストラクタで使用する活性化関数を指定できるように
# 拡張する方針でいきます。
class SingleLayer:
    def __init__(self, W, b, activation=relu):
        """ コンストラクタ：パラメータの初期化 """
        self.W = W  # 重み行列 今回は[2, 3]
        self.b = b  # バイアス [3]
        self.activation = activation    # 使用する活性化関数

    def forward(self, x):
        """ 順伝播計算 入力x[2] -> 出力y[3]"""
        affine = np.dot(x, self.W) + self.b
        if self.activation == "softmax":
            return softmax(affine)
        else:
            return relu(affine)

# 3層MLPのクラス
class MLP_3Layer:
    """ コンストラクタ：パラメータの初期化 """
    def __init__(self, W1, b1, W2, b2, W3, b3):
        # 任意の層数を想定してfor文で回せると理想ですが
        # 今回は3層と決め打ちにしているのでこれで妥協しています
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        # 各層のインスタンスを生成。リストとして管理してforで回せるようにします
        # 工夫すれば内包表記でも書けるのでそこはおまけ問題
        self.layers = []
        self.layers.append(SingleLayer(W1, b1))
        self.layers.append(SingleLayer(W2, b2))
        self.layers.append(SingleLayer(W3, b3, activation="softmax"))

    """ 順伝播処理 """
    # 入力xのshapeは[バッチサイズ、入力ベクトルの次元数]
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

def main():
    # 各層のユニット数は[2, 3, 2, 2]、バッチサイズは4
    x = np.array([[1.0, 0.5], [-0.3, -0.2], [0.0, 0.8], [0.3, -0.4]])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    b3 = np.array([0.1, 0.2])
    model = MLP_3Layer(W1, b1, W2, b2, W3, b3)
    print(model.forward(x))

if __name__ == "__main__":
    main()
