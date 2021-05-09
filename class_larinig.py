class Link:
    def __init__(self):
        self.a = 1
        self.b = 2


class Chain(Link):
    def __init__(self):
        super().__init__()
        self.c = 5

    def sum(self):
        return self.a + self.b + self.c


class MyNetwork(Chain):
    def mul(self):
        return self.a * self.b * self.c


c = Chain()

net = MyNetwork()
print(c.sum())
print(net.mul())
