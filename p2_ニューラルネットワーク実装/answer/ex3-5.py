
# ex3-4.py
# Affineクラス
# Yuma Saito

import numpy as np

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

class SoftmaxCrossEntropy:
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


def main():
    x = np.array([[1.0, 0.5], [-0.4, 0.1]])
    t = np.array([[1.0, 0.0], [0.0, 1.0]])
    output = SoftmaxCrossEntropy()
    print("順伝播出力: \n{0}".format(output.forward(x, t)))
    print("逆伝播出力: \n{0}".format(output.backprop(1)))

if __name__ == "__main__":
    main()
