import numpy as np


class Methods:
    def __init__(self):
        self.activation = Activation()
        self.loss = Loss()
        self.normalization = Normalization()
        self.optimizer = Optimizer()


class Activation:
    def no_activation(self, x):
        return x

    def dno_activation(self, x):
        return np.ones_like(x)

    def tanh(self, x):
        """[-2, 2] is linear interval"""
        return np.tanh(x)
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def dtanh(self, x):
        return 1 - self.tanh(x) ** 2

    def sigmoid(self, x):
        """[-6, 6] is linear interval"""
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) - self.sigmoid(x) ** 2

    def sigmoid_fast(self, x, mins=-6, maxs=6, nums=100):
        """cache (maxs - mins) * nums numbers
        it will be slow while it's first to use"""
        try:
            self.sigmoid_table = self.sigmoid_table
        except AttributeError:
            self.sigmoid_table = {i: self.sigmoid(i / nums) for i in range(mins * nums, maxs * nums)}
        # refer to the table
        if x <= -6:
            return -1
        elif x >= 6:
            return 1
        else:
            return self.sigmoid_table[int(x * nums)]

    def dsigmoid_fast(self, x, mins=-6, maxs=6, nums=100):
        return self.sigmoid_fast(x, mins, maxs, nums) - self.sigmoid_fast(x, mins, maxs, nums) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def drelu(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

    def softmax(self, x, axis=0):
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

    def dsoftmax(self, x, axis=0):
        su = np.sum(np.exp(x), axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex * (su - ex) / np.square(su)

    def dropout(self, x, drop_pro):
        x = np.array(x)
        for i in range(x.size):
            if np.random.random > drop_pro:
                x[np.unravel_index(i, x.shape)] = 0
        return x


class Loss:
    def l1_loss(self, real, pred):
        return np.abs(pred - real)

    def dl1_loss(self, real, pred):
        d = np.ones_like(pred)
        d[np.where(pred < real)] = -1
        return d

    def l2_loss(self, real, pred):
        return np.square(pred - real)

    def dl2_loss(self, real, pred):
        return pred - real

    def cross_entropy_loss(self, real, pred):
        return -real * np.log(pred)

    def dcross_entropy_loss(self, real, pred):
        return - real / pred

    def dsoftmax_cross_entropy_loss(self, real, pred, axis=0):
        """equals (dcross_entropy_loss * dsoftmax)"""
        pred = np.array(pred)
        pred[np.where(real == np.max(real, axis=axis))] -= 1
        return pred

    def KL_loss(self, real, pred):
        return real * np.log(real - pred)

    def dKL_loss(self, real, pred):
        return real / (pred - real)

    def exp_loss(self, real, pred):
        return np.exp(-real * pred)

    def dexp_loss(self, real, pred):
        return -real * np.exp(-real * pred)

    def Hinge_loss(self, real, pred):
        l = np.zeros_like(real)
        for i in range(len(real)):
            _ = 1 - real[i] * pred[i]
            if _ > 0:
                l[i] = _
        return l

    def dHinge_loss(self, real, pred):
        l = np.zeros_like(real)
        for i in range(len(real)):
            _ = 1 - real[i] * pred[i]
            if _ > 0:
                l[i] = - real[i]
        return l


class Normalization:
    def min_max(self, data, mi=None, ma=None):
        """
        liner changing, if a new data insert in, it will be define again.
        data fall in [0,1], use y = (x-min)/(max-min)
        if wanting to fall in [-1, 1], use y = (x-mean)/(max-min) or y = (y^2 - 1)
        of course, u can fall in any zones what u want.
        """
        data = np.array(data)
        mi = mi or np.min(data, axis=0)
        ma = ma or np.max(data, axis=0)
        data = (data - mi) / (ma - mi)
        return data, mi, ma

    def z_score(self, data, mean=None, std=None):
        """
        after normalization -> dimensionless
        data must fit with Gaussian distribution
        normalization function: y = (x - μ) / σ
        μ: mean of data, after normalization -> 0
        σ: standard deviation of data, after normalization -> 1
        """
        data = np.array(data)
        mean = mean or np.mean(data, axis=0)
        std = std or np.std(data, axis=0)
        data = (data - mean) / std
        return data, mean, std

    def binarizer(self, data, threshold=0.0):
        """
        data >= threshold -> 1
        data < threshold -> 0
        """
        data = np.array(data)
        norm = np.zeros_like(data)
        norm[np.where(data >= threshold)] = 1
        return norm

    def vec(self, data):
        """
        after normalization -> fall in the unit circle
        y = x / ||x||
        """
        data = np.array(data)
        data /= np.linalg.norm(data, axis=1).reshape(-1, 1)
        return data

    def log(self, data, ma=None):
        """
        data must be greater than 1
        y = lg(x) / lg(max) or y = lg(x)
        """
        data = np.array(data)
        ma = ma or np.max(data, axis=0)
        data = np.log10(data) / np.log10(ma)
        return data, ma

    def arctan(self, data):
        data = np.array(data)
        data = np.arctan(data) * 2 / np.pi
        return data

    def fuzzy(self, data, mi=None, ma=None):
        """
        y = 0.5+0.5sin[pi/(max-min)*(x-0.5(max-min))]
        """
        data = np.array(data)
        mi = mi or np.min(data, axis=0)
        ma = ma or np.max(data, axis=0)
        data = 0.5 + 0.5 * np.sin(np.pi / (ma - mi) * (data - 0.5 * (ma - mi)))
        return data, mi, ma


class Optimizer:
    """https://blog.csdn.net/u012328159/article/details/80311892"""

    def GradientDescent(self, vars, delta, lr=0.01):
        return vars - lr * delta

    def SGD(self, vars, delta, lr=0.01):
        try:
            rand = np.random.randint(delta.shape[0])  # random choice 1-d vars be the descent delta
            d = np.full_like(delta, delta[rand])  # todo:
            return vars - lr * d
        except IndexError:
            return self.GradientDescent(vars, delta, lr=lr)

    def Momentum(self, vars, delta, lr=0.01, discount=0.9, mv=None):
        if mv is None:
            mv = np.zeros_like(delta)
        mv = mv * discount + lr * delta
        return vars - mv, mv

    def AdaGrad(self, vars, delta, lr=1., eps=1e-6, g=None):
        if g is None:
            g = np.zeros_like(delta)
        g += np.square(delta)
        return vars - delta * lr / np.sqrt(g + eps), g

    def Adadelta(self, vars, delta, lr=1., rho=0.9, eps=1e-6, g=None, v=None):
        if g is None:
            g = np.zeros_like(delta)
        if v is None:
            v = np.zeros_like(delta)
        v = rho * v + (1 - rho) * np.square(delta)
        delta = np.sqrt(g + eps) / np.sqrt(v + eps) * delta
        return vars - lr * delta, rho * g + (1 - rho) * np.square(delta), v

    def RMSProp(self, vars, delta, lr=0.01, rho=0.9, eps=1e-6, g=None):
        if g is None:
            g = np.zeros_like(delta)
        g = rho * g + (1 - rho) * np.square(delta)
        return vars - lr / np.sqrt(g + eps) * delta, g

    def Adam(self, vars, delta, lr=0.1, beta1=0.9, beta2=0.999, t=1, eps=1e-6, m=None, v=None):
        """momentum + RMSProp"""
        if m is None:
            m = np.zeros_like(delta)
        if v is None:
            v = np.zeros_like(delta)
        m = beta1 * m + (1 - beta1) * delta  # momentum
        v = beta2 * v + (1 - beta2) * np.square(delta)  # RMSProp
        m_ = m / (1 - np.power(beta1, t))
        v_ = v / (1 - np.power(beta2, t))
        return vars - lr * m_ / (np.sqrt(v_) + eps), m, v

    def AdaMax(self, vars, delta, lr=0.01, beta1=0.9, beta2=0.999, t=1, eps=1e-6, m=None, v=None):
        if m is None:
            m = np.zeros_like(delta)
        if v is None:
            v = np.zeros_like(delta)
        m = beta1 * m + (1 - beta1) * delta  # momentum
        v = np.maximum(beta2 * v, np.abs(delta))
        m_ = m / (1 - np.power(beta1, t))
        return vars - lr * m_ / (np.sqrt(v) + eps), m, v


def show_activation():
    method = Methods()

    fig, axs = plt.subplots(3, 4)

    x = np.linspace(-5, 5, 100)
    axs[0][0].plot(x, x)
    axs[0][0].set_title('original')

    axs[1][0].plot(x, method.activation.tanh(x))
    axs[1][0].set_title('tanh')

    axs[2][0].plot(x, method.activation.dtanh(x))
    axs[2][0].set_title('dtanh')

    x = np.linspace(-10, 10, 100)
    axs[0][1].plot(x, x)
    axs[0][1].set_title('original')

    axs[1][1].plot(x, method.activation.sigmoid(x))
    axs[1][1].set_title('sigmoid')

    axs[2][1].plot(x, method.activation.dsigmoid(x))
    axs[2][1].set_title('dsigmoid')

    x = np.linspace(-5, 5, 100)
    axs[0][2].plot(x, x)
    axs[0][2].set_title('original')

    axs[1][2].plot(x, method.activation.relu(x))
    axs[1][2].set_title('relu')

    axs[2][2].plot(x, method.activation.drelu(x))
    axs[2][2].set_title('drelu')

    x = np.linspace(-5, 5, 100)
    y = -x ** 2 + 5
    axs[0][3].plot(x, y)
    axs[0][3].set_title('original')

    y_ = method.activation.softmax(y)
    axs[1][3].plot(x, y_)
    axs[1][3].set_title('softmax')

    axs[2][3].plot(x, method.activation.dsoftmax(y))
    axs[2][3].set_title('dsoftmax')

    fig.set_size_inches(30, 10)
    plt.savefig('./img/activation.png')
    plt.show()


def show_loss():
    method = Methods()

    fig, axs = plt.subplots(3, 6)

    real = np.linspace(-1, 1, 100)
    pred = np.zeros_like(real)
    axs[0][0].plot(list(range(100)), real, label='real')
    axs[0][0].plot(list(range(100)), pred, label='pred')
    axs[0][0].set_title('original')
    axs[0][0].legend()

    axs[1][0].plot(list(range(100)), method.loss.l1_loss(real, pred))
    axs[1][0].set_title('l1_loss')

    axs[2][0].plot(list(range(100)), method.loss.dl1_loss(real, pred))
    axs[2][0].set_title('dl1_loss')

    real = np.linspace(-1, 1, 100)
    pred = np.zeros_like(real)
    axs[0][1].plot(list(range(100)), real, label='real')
    axs[0][1].plot(list(range(100)), pred, label='pred')
    axs[0][1].set_title('original')
    axs[0][1].legend()

    axs[1][1].plot(list(range(100)), method.loss.l2_loss(real, pred))
    axs[1][1].set_title('l2_loss')

    axs[2][1].plot(list(range(100)), method.loss.dl2_loss(real, pred))
    axs[2][1].set_title('dl2_loss')

    real = np.ones(100)
    real[:50] -= 1
    pred = np.linspace(0.1, 0.9, 100)
    axs[0][2].plot(list(range(100)), real, label='real')
    axs[0][2].plot(list(range(100)), pred, label='pred')
    axs[0][2].set_title('original')
    axs[0][2].legend()

    axs[1][2].plot(list(range(100)), method.loss.cross_entropy_loss(real, pred))
    axs[1][2].set_title('cross_entropy_loss')

    axs[2][2].plot(list(range(100)), method.loss.dcross_entropy_loss(real, pred))
    axs[2][2].set_title('dcross_entropy_loss')

    real = np.linspace(0, 1, 100)
    pred = np.zeros_like(real) + 0.5
    axs[0][3].plot(list(range(100)), real, label='real')
    axs[0][3].plot(list(range(100)), pred, label='pred')
    axs[0][3].set_title('original')
    axs[0][3].legend()

    axs[1][3].plot(list(range(100)), method.loss.exp_loss(real, pred))
    axs[1][3].set_title('exp_loss')

    axs[2][3].plot(list(range(100)), method.loss.dexp_loss(real, pred))
    axs[2][3].set_title('dexp_loss')

    real = np.linspace(0, 2, 100)
    pred = np.zeros_like(real) + 1
    axs[0][4].plot(list(range(100)), real, label='real')
    axs[0][4].plot(list(range(100)), pred, label='pred')
    axs[0][4].set_title('original')
    axs[0][4].legend()

    axs[1][4].plot(list(range(100)), method.loss.Hinge_loss(real, pred))
    axs[1][4].set_title('Hinge_loss')

    axs[2][4].plot(list(range(100)), method.loss.dHinge_loss(real, pred))
    axs[2][4].set_title('dHinge_loss')

    real = np.linspace(0, 1, 100)
    pred = np.zeros_like(real) + 0.5
    axs[0][5].plot(list(range(100)), real, label='real')
    axs[0][5].plot(list(range(100)), pred, label='pred')
    axs[0][5].set_title('original')
    axs[0][5].legend()

    axs[1][5].plot(list(range(100)), method.loss.KL_loss(real, pred))
    axs[1][5].set_title('KL_loss')

    axs[2][5].plot(list(range(100)), method.loss.dKL_loss(real, pred))
    axs[2][5].set_title('dKL_loss')

    fig.set_size_inches(30, 15)
    # plt.savefig('./img/loss.png')
    plt.show()


def show_normalization():
    method = Methods()

    fig, axs = plt.subplots(2, 4)

    data = np.random.random(size=(100, 100)) * 10

    axs[0][0].scatter(data[:, 0], data[:, 1])
    axs[0][0].set_title('original')

    norm, _, _ = method.normalization.min_max(data)
    axs[0][1].scatter(norm[:, 0], norm[:, 1])
    axs[0][1].set_title('min_max')

    norm, _, _ = method.normalization.z_score(data)
    axs[0][2].scatter(norm[:, 0], norm[:, 1])
    axs[0][2].set_title('z_score')

    norm = method.normalization.binarizer(data, threshold=5)
    axs[0][3].scatter(norm[:, 0], norm[:, 1])
    axs[0][3].set_title('binarizer')

    norm = method.normalization.vec(data)
    axs[1][0].scatter(norm[:, 0], norm[:, 1])
    axs[1][0].set_title('vec')

    norm, _ = method.normalization.log(data)
    axs[1][1].scatter(norm[:, 0], norm[:, 1])
    axs[1][1].set_title('log')

    norm = method.normalization.arctan(data)
    axs[1][2].scatter(norm[:, 0], norm[:, 1])
    axs[1][2].set_title('arctan')

    norm, _, _ = method.normalization.fuzzy(data)
    axs[1][3].scatter(norm[:, 0], norm[:, 1])
    axs[1][3].set_title('fuzzy')

    fig.set_size_inches(20, 10)
    # plt.savefig('./img/normalization.png')
    plt.show()


def show_optimizer():
    from mpl_toolkits.mplot3d import Axes3D     # although is not used, it's imported necessary

    method = Methods()

    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    # todo: if use this var, the picture is wrong, i don't know why yet
    # z = np.zeros_like(x)
    # z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    xx, yy = np.meshgrid(x, y)
    zz = 2 * np.sin(xx) + np.cos(xx) - xx

    x_ = np.linspace(-10, 10, 5000)
    z_ = np.zeros_like(x_)
    for i in range(len(x_)):
        z_[i] = 2 * np.sin(x_[i]) + np.cos(x_[i]) - x_[i]

    def cal_x(z):
        x = np.zeros_like(z)
        for i in range(len(z)):
            argmin = np.argmin(np.abs(z_ - z[i]))
            x[i] = x_[argmin]
        return x

    def d_fx(x):
        return 2 * np.cos(x) - np.sin(x) + 1

    fig = plt.figure()

    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1] = method.optimizer.GradientDescent(z[i], dy * z[i], lr=0.01)
    px = cal_x(z)
    ax = fig.add_subplot(241, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('GradientDescent')

    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1] = method.optimizer.SGD(z[i], dy * z[i], lr=0.01)
    px = cal_x(z)
    ax = fig.add_subplot(242, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('SGD')

    m = None
    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1], m = method.optimizer.Momentum(z[i], dy * z[i], lr=0.01, mv=m)
    px = cal_x(z)
    ax = fig.add_subplot(243, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('Momentum')

    g = None
    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1], g = method.optimizer.AdaGrad(z[i], dy * z[i], lr=1, g=g)
    px = cal_x(z)
    ax = fig.add_subplot(244, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('AdaGrad')

    g = None
    v = None
    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1], g, v = method.optimizer.Adadelta(z[i], dy * z[i], lr=1, g=g, v=v)
    px = cal_x(z)
    ax = fig.add_subplot(245, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('Adadelta')

    g = None
    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1], g = method.optimizer.RMSProp(z[i], dy * z[i], lr=0.01, g=g)
    px = cal_x(z)
    ax = fig.add_subplot(246, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('RMSProp')

    m = None
    v = None
    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1], m, v = method.optimizer.Adam(z[i], dy * z[i], lr=0.1, m=m, v=v)
    px = cal_x(z)
    ax = fig.add_subplot(247, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('Adam')

    m = None
    v = None
    z = np.zeros_like(x)
    z[0] = 2 * np.sin(x[0]) + np.cos(x[0]) - x[0]
    for i in range(999):
        dy = d_fx(x[i])
        z[i + 1], m, v = method.optimizer.AdaMax(z[i], dy * z[i], lr=0.01, m=m, v=v)
    px = cal_x(z)
    ax = fig.add_subplot(248, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot(px, y, z, color='b')
    ax.set_zlim([-10, 10])
    ax.set_title('AdaMax')

    fig.set_size_inches(20, 10)
    # plt.savefig('./img/optimizer.png')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    show_optimizer()
