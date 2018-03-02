
# ex2-1.py
# パーセプトロンクラスの実装と論理ゲートの実現
# 2018/02/20    Yuma Saito

import numpy as np

# パーセプトロンのクラス定義
class Perceptron:
    def __init__(self, w1, w2, theta):
        """
        コンストラクタ：各種パラメータ変数を初期化
        今回のモデルでは学習を行わないので最初に決め打ちにしています
        """
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    def forward(self, x1, x2):
        """
        順伝播関数：入力x1, x2から出力yを計算
        """
        # True / Falseはint型にキャストすることで1 / 0になる
        return int(self.w1 * x1 + self.w2 * x2 >= self.theta)

# メイン関数
def main():
    # 入力サンプル
    x1_list = [1, 1, 0, 0]
    x2_list = [1, 0, 1, 0]
    # パラメータ設定はあくまで一例。いろいろあります。
    and_gate  = Perceptron(0.5, 0.5, 0.8)
    nand_gate = Perceptron(-0.5, -0.5, -0.8)
    or_gate   = Perceptron(0.5, 0.5, 0.3)

    # 2つのリストを同時にたどる場合はzip関数が使えます
    for x1, x2 in zip(x1_list, x2_list):
        print("AND({0}, {1}) = {2}\t".format(x1, x2, and_gate.forward(x1, x2)), end="")
        print("NAND({0}, {1}) = {2}\t".format(x1, x2, nand_gate.forward(x1, x2)), end="")
        print("OR({0}, {1}) = {2}\t".format(x1, x2, or_gate.forward(x1, x2)))
    # xorゲートの動作確認
    for x1, x2 in zip(x1_list, x2_list):
        print("XOR({0}, {1}) = {2}\t".format(x1, x2, xor(x1, x2)))

# XORゲート
def xor(x1, x2):
    # XORの実装例としては XOR = AND(NAND, OR) が一番簡単だと思います
    and_gate  = Perceptron(0.5, 0.5, 0.8)
    nand_gate = Perceptron(-0.5, -0.5, -0.8)
    or_gate   = Perceptron(0.5, 0.5, 0.3)
    return and_gate.forward(nand_gate.forward(x1, x2),
                            or_gate.forward(x1, x2))

if __name__ == "__main__":
    main()
