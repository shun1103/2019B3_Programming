
# ex3-1.py
# MultiplyクラスとAddクラス
# Yuma Saito

import numpy as np

# 例題：乗算ノード
class Multiply():
    def __init__(self):
        """ 逆伝播計算に必要な変数：forwardの入力値 """
        self.x = None
        self.y = None
    def forward(self, x, y):
        """ 順伝播計算：z = x * y """
        self.x = x
        self.y = y
        z = x * y
        return z
    def backprop(self, dz):
        """ 逆伝播計算: dz/dx = y, dz/dy = x """
        dx = dz * self.y
        dy = dz * self.x
        return dx, dy

# ex3-1：加算ノード
class Add():
    def __init__(self):
        """ 必要な変数：なし """
        pass
    def forward(self, x, y):
        """ 順伝播計算：z = x + y """
        return x + y
    def backprop(self, dz):
        """ 逆伝播計算: dz/dx = 1, dz/dy = 1 """
        return dz, dz

def main():
    a = 2
    b = 3
    c = 4
    # ノードのインスタンスを生成
    add_unit = Add()
    mult_unit = Multiply()
    # 順伝播
    v1 = add_unit.forward(a, b)
    v2 = mult_unit.forward(v1, c)
    print("順伝播出力: {0}".format(v2))
    # 各変数の逆伝播。backpropの引数の初期値は1を与えます。
    dv1, dc = mult_unit.backprop(1)
    da, db = add_unit.backprop(dv1)
    print("逆伝播出力 da: {0}, db: {1}, dc: {2}".format(da, db, dc))

if __name__ == "__main__":
    main()
