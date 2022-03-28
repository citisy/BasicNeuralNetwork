import numpy as np


class BaseOptimizer:
    def __init__(self, lr=.001):
        self.lr = lr

    def update(self, var, g, *args):
        raise NotImplementedError


class GradientDescent(BaseOptimizer):
    def update(self, var, delta, *args):
        return var - self.lr * delta, tuple()


class SGD(BaseOptimizer):
    def update(self, var, delta, *args):
        rand = np.random.randint(delta.shape[0])  # random choice 1-d var be the descent delta
        d = np.full_like(delta, delta[rand])  # todo:

        return var - self.lr * d, tuple()


class Momentum(BaseOptimizer):
    def __init__(self, lr=.01, discount=0.9):
        super(Momentum, self).__init__(lr)
        self.discount = discount

    def update(self, var, delta, mv=None):
        if mv is None:
            mv = np.zeros_like(delta)

        mv = mv * self.discount + self.lr * delta

        return var - mv, (mv, )


class AdaGrad(BaseOptimizer):
    def __init__(self, lr=1., eps=1e-6):
        super(AdaGrad, self).__init__(lr)
        self.eps = eps

    def update(self, var, delta, g=None):
        if g is None:
            g = np.zeros_like(delta)

        g += np.square(delta)

        return var - delta * self.lr / np.sqrt(g + self.eps), (g, )


class Adadelta(BaseOptimizer):
    def __init__(self, lr=1., rho=0.9, eps=1e-6):
        super(Adadelta, self).__init__(lr)
        self.rho = rho
        self.eps = eps

    def update(self, var, delta, g=None, v=None):
        if g is None:
            g = np.zeros_like(delta)

        if v is None:
            v = np.zeros_like(delta)

        v = self.rho * v + (1 - self.rho) * np.square(delta)
        delta = np.sqrt(g + self.eps) / np.sqrt(v + self.eps) * delta

        return var - self.lr * delta, (self.rho * g + (1 - self.rho) * np.square(delta), v)


class RMSProp(BaseOptimizer):
    def __init__(self, lr=.01, rho=0.9, eps=1e-6):
        super(RMSProp, self).__init__(lr)
        self.rho = rho
        self.eps = eps

    def update(self, var, delta, g=None):
        if g is None:
            g = np.zeros_like(delta)

        g = self.rho * g + (1 - self.rho) * np.square(delta)

        return var - self.lr / np.sqrt(g + self.eps) * delta, (g, )


class Adam(BaseOptimizer):
    def __init__(self, lr=.01, beta1=0.9, beta2=0.999, t=1, eps=1e-6):
        super(Adam, self).__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
        self.eps = eps

    def update(self, var, delta, m=None, v=None):
        """momentum + RMSProp"""
        if m is None:
            m = np.zeros_like(delta)

        if v is None:
            v = np.zeros_like(delta)

        m = self.beta1 * m + (1 - self.beta1) * delta  # momentum
        v = self.beta2 * v + (1 - self.beta2) * np.square(delta)  # RMSProp
        m_ = m / (1 - np.power(self.beta1, self.t))
        v_ = v / (1 - np.power(self.beta2, self.t))

        return var - self.lr * m_ / (np.sqrt(v_) + self.eps), (m, v)


class AdaMax(BaseOptimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, t=1, eps=1e-6):
        super(AdaMax, self).__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = t
        self.eps = eps

    def update(self, var, delta, m=None, v=None):
        if m is None:
            m = np.zeros_like(delta)

        if v is None:
            v = np.zeros_like(delta)

        m = self.beta1 * m + (1 - self.beta1) * delta  # momentum
        v = np.maximum(self.beta2 * v, np.abs(delta))
        m_ = m / (1 - np.power(self.beta1, self.t))

        return var - self.lr * m_ / (np.sqrt(v) + self.eps), (m, v)


gd = GradientDescent()
sgd = SGD()
momentum = Momentum()
adagrad = AdaGrad()
adadelta = Adadelta()
rmsprop = RMSProp()
adam = Adam()
adamax = AdaMax()
