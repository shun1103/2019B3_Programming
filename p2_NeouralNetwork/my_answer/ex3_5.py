import numpy as np


def softmax(x):
    logit = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(logit)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x


def cross_entropy(y, t):
    item = np.sum(-t * np.log(y), axis=1)
    return np.average(item)


class SoftmaxCrossEntropy():
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        return cross_entropy(self.y, t)

    def backprop(self, dz):
        return self.y - self.t


x = np.array([[1.0, 0.5], [-0.4, 0.1]])
t = np.array([[1.0, 0.0], [0.0, 1.0]])
output = SoftmaxCrossEntropy()
print("順伝播出力: \n{0}".format(output.forward(x, t)))
print("逆伝播出力: \n{0}".format(output.backprop(1)))
