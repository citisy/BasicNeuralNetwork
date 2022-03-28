import numpy as np
from scipy import signal
from . import activations


class BaseLayer:
    def __init__(self, units=None):
        self.units = units
        self.input_units = None
        self.activation = None
        self.optimizer = None
        self.optimizer_args = {}
        self.variable = {}

    def initializer(self, input_units, optimizer=None):
        pass

    def forward(self, x):
        """前向传播"""
        raise NotImplementedError

    def backward(self, dh):
        """后向传播"""
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Input(BaseLayer):
    def __init__(self, units):
        super().__init__()
        self.units = units


class Dense(BaseLayer):
    def __init__(self,
                 units: int,
                 activation: activations.BaseActivation = activations.no_activation,
                 w_initializer=None,
                 b_initializer=None
                 ):
        super().__init__(units)
        self.activation = activation
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer

    def initializer(self, input_units, optimizer=None):
        self.variable = {
            'w': self.w_initializer if self.w_initializer is not None else np.random.normal(.0, .1, size=(input_units, self.units)),
            'b': self.b_initializer if self.b_initializer is not None else np.zeros(shape=(1, self.units)),
        }
        self.optimizer = optimizer
        self.optimizer_args = {
            'w': tuple(),
            'b': tuple()
        }

    def forward(self, x):
        """
        N = batch_size, m = input_units, n = units

        x: [N, m]
        w: [m, n]
        b: [1, n]
        h: [N, n]
        """
        w, b = self.variable['w'], self.variable['b']
        a = x @ w + b
        h = self.activation(a)

        self.variable['x'] = x
        self.variable['a'] = a
        self.variable['h'] = h

        return h

    def backward(self, dh):
        """
        dh: [N, n]
        dx: [N, m]
        g: [N, n]
        db: [N, 1]
        dw: [m, n]
        """
        gradient = {}

        g = dh * self.activation.derivative(self.variable['a'])
        gradient['b'] = np.mean(g, axis=0, keepdims=True)
        gradient['w'] = self.variable['x'].T @ g
        gradient['x'] = g @ self.variable['w'].T

        for k in ['w', 'b']:
            self.variable[k], self.optimizer_args[k] = self.optimizer.update(self.variable[k], gradient[k], *self.optimizer_args[k])

        return gradient


class Conv2D(BaseLayer):
    def __init__(self,
                 units: int,
                 kernel_size=(3, 3),
                 activation: activations.BaseActivation = activations.relu,
                 w_initializer=None
                 ):
        super().__init__(units)
        self.kernel_size = kernel_size
        self.activation = activation
        self.w_initializer = w_initializer

    def initializer(self, input_units, optimizer=None):
        self.variable = {
            'w': self.w_initializer if self.w_initializer is not None else np.random.normal(0.0, 0.1, size=(*self.kernel_size, self.units, input_units[-1]))
        }

        self.units = list(input_units[:-1]) + [self.units]
        self.optimizer = optimizer
        self.optimizer_args = {'w': tuple()}

    def forward(self, x):
        """
        N = batch_size, (m, n) = input_size, (c1, c2) = kernel_size, k = input_unit, kc = units

        x: [N, m, n, k]
        w: [c1, c2, kc, k]
        h: [N, m, n, kc]
        """
        a = np.zeros((*x.shape[:-1], self.units[-1]))

        for k in range(self.units[-1]):
            a_ = a[:, :, :, k]
            for j in range(x.shape[-1]):
                kernel = self.variable['w'][:, :, k, j]
                for i in range(x.shape[0]):
                    a_[i, :, :] += signal.convolve2d(x[i, :, :, j], kernel, 'same')

                a[:, :, :, k] = a_

        h = self.activation(a)
        self.variable['x'] = x
        self.variable['a'] = a
        self.variable['h'] = h

        return h

    def backward(self, dh):
        """
        dh: [N, m, n, kc]
        dx: [N, m, n, k]
        g: [N, m, n, kc]
        dw: [c1, c2, kc, k]
        """

        x = self.variable['x']
        g = dh * self.activation.derivative(self.variable['a'])
        dx = np.zeros_like(x)
        dw = np.zeros_like(self.variable['w'])

        for k in range(self.units[-1]):
            for j in range(x.shape[-1]):
                kernel = self.variable['w'][:, :, k, j]

                for i in range(x.shape[0]):
                    dx[i, :, :, j] = signal.convolve2d(g[i, :, :, k], np.rot90(kernel, 2), 'same')
                    dw[:, :, k, j] = signal.convolve2d(g[i, :, :, k], x[i, :, :, j], 'valid')

        gradient = {'x': dx, 'w': dw}

        self.variable['w'], self.optimizer_args['w'] = self.optimizer.update(self.variable['w'], gradient['w'], *self.optimizer_args['w'])

        return gradient


class AvgPool2D(BaseLayer):
    def __init__(self, pool_size=(2, 2)):
        super().__init__()
        self.pool_size = pool_size

    def initializer(self, input_units, optimizer=None):
        self.units = (
            int(np.ceil(input_units[0] / self.pool_size[0])),
            int(np.ceil(input_units[1] / self.pool_size[1])),
            input_units[2]
        )

    def forward(self, x):
        """
        (p1, p2) = pool_size
        x: [N, m, n, kc]
        h: [N, m/p1, n/p2, kc]
        """
        h = np.zeros((x.shape[0], *self.units))

        for i in range(h.shape[1]):
            for j in range(h.shape[2]):
                avg = np.mean(x[:,
                              i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                              j * self.pool_size[1]:(j + 1) * self.pool_size[1],
                              :],
                              axis=(1, 2), keepdims=True)
                h[:, i:i + 1, j:j + 1, :] = avg

        self.variable['x'] = x
        self.variable['h'] = h

        return h

    def backward(self, dh):
        """
        dh: [N, m/p1, n/p2, kc]
        dx: [N, m, n, kc]
        """
        dx = np.zeros_like(self.variable['x'])
        for i in range(dh.shape[1]):
            for j in range(dh.shape[2]):
                dx[:,
                i * self.pool_size[0]:(i + 1) * self.pool_size[0],
                j * self.pool_size[1]:(j + 1) * self.pool_size[1],
                :] = dh[:, i:i + 1, j:j + 1, :] / (self.pool_size[0] * self.pool_size[1])

        return {'x': dx}


class Flatten(BaseLayer):
    def initializer(self, input_units, optimizer=None):
        self.input_units = input_units
        self.units = np.cumprod(input_units)[-1]

    def forward(self, x):
        """
        x: [N, m, n, kc]
        h: [N, m * n * kc]
        """
        h = x.reshape((x.shape[0], -1))
        self.variable['x'] = x
        self.variable['h'] = h

        return h

    def backward(self, dh):
        """
        dh: [N, m * n * kc]
        dx: [N, m, n, kc]
        """

        return {'x': dh.reshape((-1, *self.input_units))}


class SimpleRNN(BaseLayer):
    def __init__(self,
                 units: int,
                 activation: activations.BaseActivation = activations.relu,
                 return_sequences=True,
                 u_initializer=None,
                 v_initializer=None,
                 b_initializer=None,
                 ):
        super().__init__(units)
        self.activation = activation
        self.return_sequences = return_sequences
        self.u_initializer = u_initializer
        self.v_initializer = v_initializer
        self.b_initializer = b_initializer

    def initializer(self, input_units, optimizer=None):
        self.variable = {
            'v': self.v_initializer if self.v_initializer is not None else np.random.normal(.0, .1, size=(input_units[-1], self.units)),
            'u': self.u_initializer if self.u_initializer is not None else np.random.normal(.0, .1, size=(self.units, self.units)),
            'b': self.b_initializer if self.b_initializer is not None else np.zeros(shape=(1, self.units)),
        }
        self.optimizer = optimizer
        self.optimizer_args = {
            'v': tuple(),
            'u': tuple(),
            'b': tuple()
        }
        self.input_units = input_units[-1]

    def forward(self, x):
        """
        N = batch_size, m = time_steps, n = input_unit, nc = unit

        x: [N, m, n]
        h: [N * m, nc]
        v: [n, nc]
        u: [nc, nc]
        b: [1, nc]
        """
        u, v, b = self.variable['u'], self.variable['v'], self.variable['b']
        a = np.zeros((*x.shape[:-1], self.units))
        h = np.zeros_like(a)

        for i in range(x.shape[1]):
            x_ = x[:, i, :]

            if i == 0:
                a_ = x_ @ v + b
            else:
                a_ = x_ @ v + h[:, i - 1, :] @ u + b

            h_ = self.activation(a_)

            a[:, i, :] = a_
            h[:, i, :] = h_

        self.variable['x'] = x
        self.variable['a'] = a
        self.variable['h'] = h

        return h.reshape((-1, self.units))

    def backward(self, dh):
        """
        dx: [N, m, n]
        dh: [N * m, nc]
        g: [N, nc]
        dv: [n, nc]
        du: [nc, nc]
        db: [1, nc]
        """
        a = self.variable['a']
        x = self.variable['x']
        h = self.variable['h']

        dh = dh.reshape(h.shape)
        dx = np.zeros((*dh.shape[:-1], self.input_units))

        for i in range(dh.shape[1]):
            dh_t = dh[:, i, :]
            a_t = a[:, i, :]
            x_t = x[:, i, :]
            g = dh_t * self.activation.derivative(a_t)

            if i == 0:
                dv = x_t.T @ g
            else:
                u_ = np.repeat(np.diag(self.variable['u']).reshape((1, -1)), dh.shape[0], axis=0)
                dv = (x_t + u_ @ dv.T).T @ g

            self.variable['v'], self.optimizer_args['v'] = self.optimizer.update(self.variable['v'], dv, *self.optimizer_args['v'])

            if i > 0:
                h_t = h[:, i - 1, :]
                if i == 1:
                    du = h_t.T @ g
                else:
                    u_ = np.repeat(np.diag(self.variable['u']).reshape((1, -1)), dh.shape[0], axis=0)

                    du = (h_t + u_ @ du.T).T @ g

                self.variable['u'], self.optimizer_args['u'] = self.optimizer.update(self.variable['u'], du, *self.optimizer_args['u'])

            db = np.mean(g, axis=0, keepdims=True)
            self.variable['b'], self.optimizer_args['b'] = self.optimizer.update(self.variable['b'], db, *self.optimizer_args['b'])

        return {'x': dx}
