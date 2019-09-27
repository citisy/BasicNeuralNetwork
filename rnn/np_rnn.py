"""
http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
https://blog.csdn.net/n9nzjx57bf/article/details/71747294
"""

import numpy as np
from basic_neural_network.methods import Methods


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.method = Methods()
        self.U = np.random.normal(0.0, 0.1, size=(self.hidden_dim, self.input_dim))
        self.W = np.random.normal(0.0, 0.1, size=(self.hidden_dim, self.hidden_dim))
        self.V = np.random.normal(0.0, 0.1, size=(self.output_dim, self.hidden_dim))
        self.b_h = np.zeros(shape=(hidden_dim, 1))
        self.b_o = np.zeros(shape=(output_dim, 1))

    def train(self, inputs, labels, hs=None, max_iter=1000, lr=1e-3):
        """
        forward:
            h_t = U .* x_t + W .* h_(t-1) +b_h
            a_t = af1(h_t)
            o_t = V .* a_t + b_o
            pred = af2(h_t)
            x is inputs, h is hidden unit, o is outputs,
            b is bias, af is activity function, af1 always tanh, af2 always softmax

        backward:
            we define the loss function, rnn usually used for classifier problem, so it usually use cross entropy loss
            but there, we use l2 loss is enough
                L(theta) = l2_loss(outputs, pred)
                dL = pred - outputs
            then, it's easy to get gV and gb_o
                gb_o = dL/db_o = (pred - outputs)
                gV = dL/dV = (pred - outputs) .* a_t.T
                d_t = dL/do_t * do_t/dh_t + dL/dh_(t+1) * dh_(t+1)/dh_t
            'cause it haven't t+1 times, so,
                d_t = dL/do_t * do_t/dh_t = daf(h_t) * V.t .* dL
                gb_h = dL/db_h = d_t
                gW = dL/dW = d .* h_(t-1).T
                gU = dL/dU = d .* inputs.T
            update the weights and bias, over.
        """
        inputs = np.array(inputs)  # [batch, time_step_dim, input_dim]
        labels = np.array(labels)
        # if old_a is None:
        #     old_a = np.zeros((self.hidden_dim, inputs.shape[1]))
        loss = np.nan
        preds = np.zeros_like(labels.T)  # [output_dim, time_step_dim, batch]
        if hs is None:
            hs = np.zeros((inputs.shape[1], self.hidden_dim, inputs.shape[0]))  # [time_step_dim, hidden_dim, batch]
        old_a = self.method.activation.tanh(hs[0, :, :].reshape((self.hidden_dim, -1)))

        for _ in range(max_iter):
            # forward
            for i in range(inputs.shape[1]):
                input = inputs[:, i, :].reshape((-1, self.input_dim)).T
                label = labels[:, i, :].reshape((-1, self.output_dim)).T
                h = self.U.dot(input) + self.W.dot(old_a) + self.b_h  # [hidden_dim, batch]
                a1 = self.method.activation.tanh(h)
                o = self.V.dot(a1) + self.b_o
                pred = self.method.activation.no_activation(o)
                loss = 0.5 * np.mean(self.method.loss.l2_loss(label, pred))

                hs[i, :, :] = h[np.newaxis, :, :]
                preds[:, i, :] = pred[:, np.newaxis, :]

            # backward
            for i in range(inputs.shape[1])[::-1]:
                input = inputs[:, i, :].reshape((-1, self.input_dim)).T
                label = labels[:, i, :].reshape((-1, self.output_dim)).T
                pred = preds[:, i, :].reshape((self.output_dim, -1))
                h = hs[i, :, :].reshape((self.hidden_dim, -1))
                a1 = self.method.activation.tanh(h)

                dL = self.method.loss.dl2_loss(label, pred)
                gb_o = dL
                gV = dL.dot(h.T)

                d_t = self.method.activation.dtanh(h) * self.V.T.dot(dL)

                gW = d_t.dot(old_a.T)
                gb_h = d_t
                gU = d_t.dot(input.T)

                self.U -= lr / input.shape[1] * gU
                self.W -= lr / old_a.shape[1] * gW
                self.V -= lr / a1.shape[1] * gV
                self.b_h -= lr * np.mean(gb_h, axis=1, keepdims=True)
                self.b_o -= lr * np.mean(gb_o, axis=1, keepdims=True)
                old_a = a1

        return preds.T, loss, hs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import seaborn as sns

    sns.set(style="white", palette="muted", color_codes=True)

    ims = []
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    plt.ylim((-1.2, 1.2))

    np.random.seed(2)

    time_step_dim = 10
    batch_size = 10
    input_dim = output_dim = 1
    hidden_dim = 32

    dnn = RNN(input_dim, hidden_dim, output_dim)

    hs = None
    for step in range(4900):
        # use sin predicts cos
        steps = np.linspace(0.25 * step * np.pi, (0.25 * step + 2) * np.pi, time_step_dim * batch_size)
        x = np.sin(steps).reshape((-1, time_step_dim, input_dim))  # shape (batch, time_step, input_size)
        y = np.cos(steps).reshape((-1, time_step_dim, output_dim))

        preds, loss, hs = dnn.train(x, y, hs=hs, lr=0.01, max_iter=1)

        if step % 49 == 0:
            line1, = ax.plot(list(range(time_step_dim * batch_size)), y.flatten(), 'r-', animated=True)
            line2, = ax.plot(list(range(time_step_dim * batch_size)), preds.flatten(), 'b-', animated=True)
            text = ax.text(12.75, -0.75, 'step=%d\nloss=%f' % (step, loss))
            ims.append([line1, line2, text])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000, repeat=True)
    # ani.save('../img/rnn_more_iter.gif', writer='imagemagick')
    plt.show()
