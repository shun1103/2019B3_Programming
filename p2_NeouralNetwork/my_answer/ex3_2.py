import numpy as np
class ReLU():
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x > 0).astype(int)
        return x * self.mask
    def backprop(self, dz):
        return dz * self.mask

x = np.array([-0.5, 0.0, 1.0, 2.0])
relu = ReLU()
print("順伝播出力: {0}".format(relu.forward(x)))
print("逆伝播出力: {0}".format(relu.backprop(1.)))
