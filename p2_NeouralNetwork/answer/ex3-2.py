
# ex3-2.py
# ReLUクラス
# Yuma Saito

import numpy as np

# ex3-2：ReLUノード
# 以降、変数は一般のnumpy配列と仮定します
class ReLU():
    def __init__(self):
        """ 必要な変数：mask
            順伝播時に0以上だった要素は1、0未満の要素は0とした配列 """
        self.mask = None
    def forward(self, x):
        """ 順伝播計算：z = max(0, x) """
        # mask配列を計算。astype(int)でTrue/Falseを1/0に変換
        self.mask = (x > 0).astype(int)
        # このmaskをかけることでelement-wiseのreluを実装可能
        return x * self.mask
    def backprop(self, dz):
        """ 逆伝播計算：dz/dx = 1 (x >= 0のとき) or 0 (x < 0のとき) """
        # 実は先ほどのmaskがdz/dxをそのまま表している
        return dz * self.mask

def main():
    x = np.array([-0.5, 0.0, 1.0, 2.0])
    relu = ReLU()
    print("順伝播出力: {0}".format(relu.forward(x)))
    print("逆伝播出力: {0}".format(relu.backprop(1.)))

if __name__ == "__main__":
    main()
