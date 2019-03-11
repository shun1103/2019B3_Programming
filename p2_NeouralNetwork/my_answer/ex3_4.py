import numpy as np

class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        L = np.dot(x, self.W) + self.b
        return L
    def backprop(self, dz):
        self.dW = np.dot((self.x).T, dz)
        self.db = np.sum(dz)
        return np.dot(dz, (self.W).T)

x = np.array([[1.0, 0.5], [-0.4, 0.1]])
W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b = np.array([0.1, 0.2, 0.3])
affine = Affine(W, b)
print("順伝播出力: \n{0}".format(affine.forward(x)))
print("逆伝播出力dx: \n{0}".format(affine.backprop(np.ones([2, 3]))))
print("逆伝播出力dw: \n{0}".format(affine.dW))
print("逆伝播出力db: \n{0}".format(affine.db))