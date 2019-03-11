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
        z = np.dot(x, self.W) + self.b
        return relu(z)

class MLP_3Layer():
    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.w1 = W1
        self.b1 = b1
        self.w2 = W2
        self.b2 = b2
        self.w3 = W3
        self.b3 = b3

        self.layers = []
        self.layers.append(SingleLayer(W1, b1))
        self.layers.append(SingleLayer(W2, b2))
        self.layers.append(SingleLayer(W3, b3))


    def forword(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

x = np.array([[1.0, 0.5], [-0.3, -0.2], [0.0, 0.8], [0.3, -0.4]])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

test = MLP_3Layer(W1, b1, W2, b2, W3, b3)
print(test.forword(x))