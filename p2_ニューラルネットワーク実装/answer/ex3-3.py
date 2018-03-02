
# ex3-3.py
# Sigmoidクラス
# Yuma Saito

import numpy as np

# ex3-3：Sigmoidノード
class Sigmoid:
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

def main():
    x = np.array([-0.5, 0.0, 1.0, 2.0])
    sigmoid = Sigmoid()
    print("順伝播出力: {0}".format(sigmoid.forward(x)))
    print("逆伝播出力: {0}".format(sigmoid.backprop(1.)))

if __name__ == "__main__":
    main()
