import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)

class SingleLayer():
    def __init__(self, W, b):
        self.W = W
        self.b = b
    def forward(self, x):
        z = np.dot((self.W).T, x) + self.b
        return relu(z)

x = np.array([1.0, 0.5])
W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b = np.array([0.1, 0.2, 0.3])

sample = SingleLayer(W, b)
print(sample.forward(x))