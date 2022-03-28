import numpy as np


class BaseActivation:
    def forward(self, x, *args):
        return x

    def derivative(self, x):
        return np.ones_like(x)

    def __call__(self, x, *args):
        return self.forward(x, *args)


class Sigmoid(BaseActivation):
    def forward(self, x, *args):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        f = self.forward(x)
        return f - f ** 2


class FastSigmoid(BaseActivation):
    def __init__(self, mins=-6, maxs=6, nums=100):
        super(FastSigmoid, self).__init__()
        self.mins = mins
        self.maxs = maxs
        self.nums = nums

        func = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_table = {i: func(i / nums) for i in range(mins * nums, maxs * nums)}

    def forward(self, x, *args):
        """cache (maxs - mins) * nums numbers
        it will be slow while it's first to use"""
        if x <= self.mins:
            return -1
        elif x >= self.maxs:
            return 1
        else:
            return self.sigmoid_table[int(x * self.nums)]

    def derivative(self, x):
        f = self.forward(x)
        return f - f ** 2


class Tanh(BaseActivation):
    def forward(self, x, *args):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - self.forward(x) ** 2


class Relu(BaseActivation):
    def forward(self, x, *args):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class Softmax(BaseActivation):
    def forward(self, x, axis=1):
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

    def derivative(self, x, axis=1):
        su = np.sum(np.exp(x), axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex * (su - ex) / np.square(su)


class Dropout(BaseActivation):
    def forward(self, x, drop_pro=.3):
        x = np.array(x)
        for i in range(x.size):
            if np.random.random() > drop_pro:
                x[np.unravel_index(i, x.shape)] = 0
        return x

    def derivative(self, x):
        pass


no_activation = BaseActivation()
sigmoid = Sigmoid()
fast_sigmoid = FastSigmoid()
tanh = Tanh()
relu = Relu()
softmax = Softmax()
