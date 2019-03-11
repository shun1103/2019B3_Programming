
import numpy as np
from collections import OrderedDict
from Nodes import *

class NeuralNetwork:

    def __init__(self):
        """
        ニューラルネットワークの構築
        今回は簡単のため層数とユニット数は[784, 500, 10]で決め打ちにします
        """
        # 学習パラメータの初期化
        self.param_names = ["W1", "b1", "W2", "b2"]
        self.params = {}
        self.params_grad = {}
        self.params["W1"] = np.random.normal(0., 0.1, [784, 500])
        self.params["b1"] = np.zeros([500])
        self.params["W2"] = np.random.normal(0, 0.1, [500, 10])
        self.params["b2"] = np.zeros([10])

        # 各層のインスタンス生成
        self.affine1 = Affine()
        self.relu1 = ReLU()
        self.affine2 = Affine()
        self.output = SoftmaxCrossEntropy()

    def forward(self, x):
        """
        ニューラルネットワークの順伝播
        推論（未知の数字画像に対する分類）用途のことも考え最終層の直前まで
        """
        z1 = self.affine1.forward(x, self.params["W1"], self.params["b1"])
        z2 = self.relu1.forward(z1)
        z3 = self.affine2.forward(z2, self.params["W2"], self.params["b2"])
        return z3

    def loss(self, x, t):
        """
        損失関数の計算
        """
        return self.output.forward(self.forward(x), t)

    def backprop(self, x, t):
        """
        モデル全体に誤差逆伝播法を実行、パラメータの勾配を取得
        """
        self.loss(x, t)
        dz3 = self.output.backprop()
        dz2, self.params_grad["W2"], self.params_grad["b2"] = \
            self.affine2.backprop(dz3)
        dz1 = self.relu1.backprop(dz2)
        _, self.params_grad["W1"], self.params_grad["b1"] = \
            self.affine1.backprop(dz1)

    def sgd(self, x, t, learning_rate = 0.0001):
        """
        SGDにより勾配を計算。学習時はこの関数のみを実行する
        """
        self.backprop(x, t) # 各パラメータの勾配ディクショナリを計算
        for param in self.param_names:
            self.params[param] -= learning_rate * self.params_grad[param]
