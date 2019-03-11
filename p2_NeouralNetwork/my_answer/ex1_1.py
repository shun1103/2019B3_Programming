class Perceptron():
    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    def forward(self, x1, x2):
        if self.w1 * x1 + self.w2 * x2 >= self.theta:
            return 1
        else:
            return 0


and_gate = Perceptron(1, 1, 2)
nand_gate = Perceptron(-1, -1, -1)
or_gate = Perceptron(1, 1, 1)

x1_list = [1, 1, 0, 0]
x2_list = [1, 0, 1, 0]

for i in range(len(x1)):
    print("AND({0}, {1}) = {2}\t".format(x1[i], x2[i], and_gate.forward(x1[i], x2[i])), end="")
    print("NAND({0}, {1}) = {2}\t".format(x1[i], x2[i], nand_gate.forward(x1[i], x2[i])), end="")
    print("OR({0}, {1}) = {2}\t".format(x1[i], x2[i], or_gate.forward(x1[i], x2[i])))

def xor(x1, x2):
    and_gate = Perceptron(1, 1, 2)
    nand_gate = Perceptron(-1, -1, -1)
    or_gate = Perceptron(1, 1, 1)

    return and_gate.forward(nand_gate.forward(x1, x2), or_gate.forward(x1, x2))
