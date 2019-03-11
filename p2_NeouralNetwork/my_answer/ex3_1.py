class Add():
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        z = x + y
        return z
    def backprop(self, dz):
        return dz, dz

class Multiply():
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        z = x * y
        return z
    def backprop(self, dz):
        dx = dz * self.y
        dy = dz * self.x
        return dx, dy

a = 2
b = 3
c = 4
add = Add()
mult = Multiply()
forward = mult.forward(add.forward(a,b), c)
tmp, dc = mult.backprop(1)
da, db = add.backprop(tmp)
print('順伝播出力: %d' %forward)
print('逆伝播出力: da:{0}, db:{1}, dc:{2}' .format(da, db, dc))