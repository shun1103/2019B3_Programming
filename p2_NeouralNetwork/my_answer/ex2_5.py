import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    logit = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(logit)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x


class SingleLayer:
    def __init__(self, W, b, activation=relu):
        self.W = W
        self.b = b
        self.activation = activation
    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        if self.activation == "softmax":
            return softmax(z)
        else:
            return relu(z)


class MLP_3Layer():
    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3

        self.layers = []
        self.layers.append(SingleLayer(W1, b1))
        self.layers.append(SingleLayer(W2, b2))
        self.layers.append(SingleLayer(W3, b3,activation="softmax"))


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