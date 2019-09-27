"""
http://www.cnblogs.com/pinard/p/6422831.html
https://blog.csdn.net/z962013489/article/details/80113960
https://blog.csdn.net/qq_30666517/article/details/80238729
"""

import numpy as np
from basic_neural_network.methods import Methods


class DNN:
    """
    DNN is the most simple neural network,
    it only contains 3 layer types, input layer, hidden layer and hidden layer
    two propagation parts, forward propagation and backward propagation
    """

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        self.method = Methods()

        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim

        self.w1 = np.random.normal(0.0, 0.1, size=(self.hidden1_dim, self.input_dim))
        self.b1 = np.zeros(shape=(self.hidden1_dim, 1))
        self.w2 = np.random.normal(0.0, 0.2, size=(self.hidden2_dim, self.hidden1_dim))
        self.b2 = np.ones(shape=(self.hidden2_dim, 1))
        self.w3 = np.random.normal(0.0, 0.5, size=(self.output_dim, self.hidden2_dim))
        self.b3 = np.zeros(shape=(self.output_dim, 1))

    def train(self, inputs, labels, lr=0.5, max_iter=5000):
        """
        assume that:
            there is neural network with 3 hidden layers
            we can express that:
                input     hidden1             hidden1             hidden3              output
                inputs -> w1: (h1_d, in_d) -> w2: (h2_d, h1_d) -> w3: (out_d, h1_d) -> outputs

        forward propagation:
            h1 = w1 .* inputs + b1
            a1 = af(h1)
            h2 = w2 .* a1 + b2
            a2 = af(h2)
            h3 = w3 .* a2 + b3
            a3 = af(h3)
            a3 is the output what we predict, it's over, very easy!

        backward propagation:
            we define a loss function L(theta), there, we use the l2 loss,
            we define that L(theta) = 0.5 * sum(a3 - outputs)
            our goal is making the L(theta) smaller and smaller
            so we need to calculate the dL/dtheta=0
            there, dL/dtheta = dL/dw3 * dw3/da2 * dw2/da1 * da1/dtheta
            so, we get that:
                dL/da3 = a3 - outputs
                da3/dh3 = daf(h3) = a3 * (1- a3)
                    ->  z3 = dL/da3 * da3/dh3 = (a3 - outputs) * (a3 * (1- a3))
                after we get the z3, we will get z2 and z1 (the proof process is skipped)
                    ->  z2 = w3.T .* daf(h2)
                    ->  z1 = w2.T .* daf(h1)
            let's update weights and bias:
                dh3/dw3 = a2.T
                g3 = dL/dw3 = dL/da3 * da3/dh3 * dh3/dw3 = z3 .* a2.T
                    ->  w3 -= lr / n(a3) * z3 .* a2.T
                    ->  b3 -= lr * mean(z3)
                    ->  w2 -= lr / n(a2) * z2 .* a1.T
                    ->  b2 -= lr * mean(z2)
                    ->  w1 -= lr / n(a1) * z1 .* inputs.T
                    ->  b1 -= lr * mean(z1)
            step by step, finally, we update all the weights and bias
            one turn training is over.
        """
        inputs = np.array(inputs).T
        labels = np.array(labels).T
        a3 = np.zeros_like(labels)
        loss = np.nan
        for i in range(max_iter):
            # forward
            h1 = np.matmul(self.w1, inputs) + self.b1
            a1 = self.method.activation.sigmoid(h1)
            h2 = np.matmul(self.w2, a1) + self.b2
            a2 = self.method.activation.sigmoid(h2)
            h3 = np.matmul(self.w3, a2) + self.b3
            a3 = self.method.activation.no_activation(h3)

            # backward
            loss = 0.5 * np.mean(self.method.loss.l2_loss(labels, a3))
            da = self.method.loss.dl2_loss(labels, a3)
            z3 = da * self.method.activation.dsigmoid(h3)
            z2 = self.w3.T.dot(z3) * self.method.activation.dsigmoid(h2)
            z1 = self.w2.T.dot(z2) * self.method.activation.dsigmoid(h1)

            self.w3 -= lr / a3.shape[1] * z3.dot(a2.T)
            self.b3 -= lr * np.mean(z3, axis=1, keepdims=True)

            self.w2 -= lr / a2.shape[1] * z2.dot(a1.T)
            self.b2 -= lr * np.mean(z2, axis=1, keepdims=True)

            self.w1 -= lr / a1.shape[1] * z1.dot(inputs.T)
            self.b1 -= lr * np.mean(z1, axis=1, keepdims=True)
        return a3.T, loss

    def train_(self, inputs, labels, lr=0.5, max_iter=5000):
        """another backward training ways"""
        inputs = np.array(inputs).T
        labels = np.array(labels).T
        a3 = np.zeros_like(labels)
        loss = np.inf
        for i in range(max_iter):
            # forward
            h1 = np.matmul(self.w1, inputs) + self.b1
            a1 = self.method.activation.sigmoid(h1)
            h2 = np.matmul(self.w2, a1) + self.b2
            a2 = self.method.activation.sigmoid(h2)
            h3 = np.matmul(self.w3, a2) + self.b3
            a3 = self.method.activation.no_activation(h3)

            # backward
            loss = 0.5 * np.mean(self.method.loss.l2_loss(labels, a3))
            da = self.method.loss.dl2_loss(labels, a3)

            da_2 = self.backward_propagation(da, h3, a2, self.w3, self.b3, lr, True)
            da_1 = self.backward_propagation(da_2, h2, a1, self.w2, self.b2, lr)
            da_0 = self.backward_propagation(da_1, h1, inputs, self.w1, self.b1, lr)
        return a3.T, loss

    def backward_propagation(self, da, h_1, a_1, w, b, lr, last=False):
        """
        da:current layer activation output partial devirate result
        a:current layer activation output
        a_1:previous layer of current layer activation output
        w:current parameter
        b:current bias
        """
        if last:
            dz = self.method.activation.dno_activation(h_1) * da
        else:
            dz = self.method.activation.dsigmoid(h_1) * da
        dw = np.matmul(dz, a_1.T)
        db = np.mean(dz, axis=1, keepdims=True)
        da_1 = np.matmul(w.T, dz)

        w -= lr / da.shape[1] * dw
        b -= lr * db

        return da_1


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import seaborn as sns

    sns.set(style="white", palette="muted", color_codes=True)

    ims = []
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    np.random.seed(1)  # seed is 4, the model can't work, why? maybe u will increase the hidden dim
    x = np.linspace(-2, 2, 100)
    noise = np.random.normal(0.0, 0.5, size=100)
    y = x ** 2 + noise
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    dnn = DNN(1, 8, 8, 1)

    sca = ax.scatter(x, y, c='r', animated=True)
    for i in range(200):
        predict, loss = dnn.train(x, y, lr=0.5, max_iter=100)
        line, = ax.plot(x, predict, c='b', animated=True)
        text = ax.text(1.25, 0.5, 'loss=%f' % loss)
        ims.append([sca, line, text])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000, repeat=True)
    # ani.save('../img/dnn_1_8_8_1.gif', writer='imagemagick')
    plt.show()
