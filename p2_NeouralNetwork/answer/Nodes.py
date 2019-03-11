
# ex3.py
# 各種演算ノードの実装
# Yuma Saito

import numpy as np

# 演算ノード一般の抽象クラス
# 共通してforwardメソッドとbackpropメソッドを持つ
# 各種ノードはこのクラスを継承して各メソッドをオーバーライドする
from abc import ABCMeta, abstractmethod
class OperationNode(metaclass=ABCMeta):
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backprop(self):
        pass

# ex3-2：ReLUノード
class ReLU(OperationNode):
    def __init__(self):
        """ 必要な変数：mask
            順伝播時に0以上だった要素は1、0未満の要素は0とした配列 """
        self.mask = None
    def forward(self, x):
        """ 順伝播計算：z = max(0, x) """
        # mask配列を計算。astype(int)でTrue/Falseを1/0に変換
        self.mask = (x > 0).astype(int)
        # このmaskをかけることでelement-wiseのreluを実装可能
        return x * self.mask
    def backprop(self, dz):
        """ 逆伝播計算：dz/dx = 1 (x >= 0のとき) or 0 (x < 0のとき) """
        # 実は先ほどのmaskがdz/dxをそのまま表している
        return dz * self.mask

# ex3-3：Sigmoidノード
class Sigmoid(OperationNode):
    def __init__(self):
        """ 必要な変数：forwardの出力値 """
        self.z = None
    def forward(self, x):
        """ 順伝播計算: z = 1/(1 + e^-x) """
        self.z = 1. / (1. + np.exp(-x))
        return self.z
    def backprop(self, dz):
        """ 逆伝播計算: dz/dx = z(1-z) """
        return dz * self.z * (1. - self.z)

# ex3-4: Affineノード
class Affine(OperationNode):
    def __init__(self):
        self.W = None   # shape = [入力ユニット数、出力ユニット数]
        self.b = None   # shape = [出力ユニット数]
        self.x = None   # dW, dbの計算に必要。順伝播の入力
    def forward(self, x, W, b):
        """ 順伝播計算：z = xW + b """
        # shape(x) = [バッチサイズ、入力ユニット数]
        self.x = x
        self.W = W
        self.b = b
        return np.dot(x, self.W) + self.b
    def backprop(self, dz):
        """ 逆伝播計算
            shape(dL/dz) = [バッチサイズ、出力ユニット数]
            dL/dx = dL/dz・W^T   shape = [バッチサイズ、入力ユニット数]
            dL/dW = x^T・dL/dz   shape = [入力ユニット数、出力ユニット数]
            dL/db = dL/dzをバッチ方向に総和をとったもの shape = [出力ユニット数]
        """
        dx = np.dot(dz, self.W.T)
        dW = np.dot(self.x.T, dz)
        db = np.sum(dz, axis=0)
        return dx, dW, db

# ex3-5: SoftmaxCrossEntropyノード
def softmax(x):
    """ softmax関数 """
    logit = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(logit)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x

def cross_entropy(y, t):
    """ 交差エントロピー損失の計算
        y: softmax関数の出力        shape = [バッチサイズ、出力層次元数]
        t: one-hot教師ラベルのバッチ shape = [バッチサイズ、出力層次元数]
        出力: 交差エントロピー       shape = []（スカラー）
    """
    # 出力層次元の方向(axis=1)に関して -t*log(y) の総和をとる
    one_sample_loss = np.sum(-t * np.log(y), axis=1)
    # その後バッチ方向(axis=0)に関して 損失の平均をとる
    return np.average(one_sample_loss)

class SoftmaxCrossEntropy(OperationNode):
    def __init__(self):
        """ 必要な変数：softmaxの出力yと教師ラベルt """
        self.y = None   # shape: [バッチサイズ、出力次元数]
        self.t = None   # shape: [バッチサイズ、出力次元数]
    def forward(self, x, t):
        """ 順伝播計算：cross_entropy(softmax(x), t) """
        self.y = softmax(x)
        self.t = t
        return cross_entropy(self.y, t)
    def backprop(self, dz=1):
        """ 逆伝播計算：dL/dx = softmax(x) - t """
        return self.y - self.t
