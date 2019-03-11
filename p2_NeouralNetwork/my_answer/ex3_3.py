import numpy as np
class Softmax():
    def __init__(self):
        self.z = None
    def forward(self, x):
        self.z = 1. / (1. + np.exp(-x))
        return self.z

    def backprop(self, dz):
        dx = dz * self.z * (1 - self.z)
        return dx

x = np.array([-0.5, 0.0, 1.0, 2.0])
softmax = Softmax()
print("順伝播出力: {0}".format(softmax.forward(x)))
print("逆伝播出力: {0}".format(softmax.backprop(1.)))