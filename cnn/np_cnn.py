"""
https://github.com/ahmedfgad/NumPyCNN
"""

import numpy as np
from scipy import signal
from basic_neural_network.methods import Methods
from tqdm import tqdm
import sys


class CNN:
    def __init__(self, height, width, channel, kernel_num, kernel_dim, pool_size, output_dim):
        self.kernel_num = kernel_num
        self.kernel_dim = kernel_dim
        self.height = height
        self.width = width
        self.channel = channel
        self.pool_size = pool_size
        self.pool_dim = (height // pool_size, width // pool_size)
        self.f_dim = self.pool_dim[0] * self.pool_dim[1] * kernel_num
        self.output_dim = output_dim
        self.method = Methods()

        self.kernels = np.random.normal(0.0, 0.1, size=(kernel_dim, kernel_dim, kernel_num))
        self.f_weights = np.random.normal(0.0, 0.1, size=(self.f_dim, self.f_dim))
        self.b_f = np.zeros(shape=(self.f_weights.shape[0], 1))
        self.o_weights = np.random.normal(0.0, 0.1, size=(output_dim, self.f_dim))
        self.b_o = np.zeros(shape=(self.o_weights.shape[0], 1))

    def train(self, inputs, labels, max_iter=200, lr=0.05):
        """
        forward:
            assume that
                the inputs matrix([batch, height, width, channel]) is 1 * 28 * 28 * 1
                the kernel matrix([kernel_dim, kernel_dim, kernel_num]) as k is 5 * 5 * 8
                    (in fact, 6 kernels is enough, because the numbers has 6 direction in all)
            convolution layer:
                hc = conv(k, inputs)    (same mode)hc -> 1 * 28 * 28 * 8
                ac = af(hc)             (af usually is relu)ac -> 1 * 28 * 28 * 8
            pool layer:
                hp = pool(ac)           (pool_size = 4)hp -> 7 * 7 * 8
                p_flat = flat(hp.T)       p_flat -> [8 * 7 * 7] * 1
            full connection:
                hf = (wf.dot(p_flat) + bf)   hf -> [8 * 7 * 7] * 1
                af = af(hf)                 (af usually is tanh)af -> [8 * 7 * 7] * 1
            output layer:
                ho = (wo.dot(af) + bo)      ho -> 1 * 1
                ao = af(ho)                 (af usually is softmax)ao -> 1 * 1

        backward:
            we use cross entropy loss function.
            zo = dcross_entropy_loss(real, ao) * dsoftmax(ao) = dsoftmax_cross_entropy_loss(real, ao)   zo -> 1 * 1
            zf = wo.T.dot(zo) * daf(hf)      zf -> [8 * 7 * 7] * 1
            dp = upsample(zf).T              zp -> 1 * 28 * 28 * 8
            zc = dp * daf(hc)                zc -> 1 * 28 * 28 * 8
            d = conv(zc, rot180(k))         (use valid mode)d -> 24 * 24 * 8
            update weight:
                k -= lr * conv(rot180(inputs), d)     (use valid mode) conv(rot180(inputs), d) -> 5 * 5 * 8
                wo, wf, bo, bf's updating same as dnn
        """
        inputs = np.array(inputs)  # [batch, height, width, channel]
        labels = np.array(labels).T

        # [batch, height, width, kernel_num]
        hs_c = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], self.kernel_num))
        as_c = np.zeros_like(hs_c)
        pools = np.zeros((inputs.shape[0], self.pool_dim[0], self.pool_dim[1], self.kernel_num))
        for it in tqdm(range(max_iter)):
            for k in range(self.kernel_num):
                # forward
                kernel = self.kernels[:, :, k]
                h_c = hs_c[:, :, :, k]
                a_c = as_c[:, :, :, k]
                pool = pools[:, :, :, k]
                for i in range(inputs.shape[0]):
                    # conv
                    h_c[i, :, :] = signal.convolve2d(inputs[i, :, :, 0], kernel, 'same')
                    a_c[i, :, :] = self.method.activation.relu(h_c[i, :, :])

                    # pooling
                    for ii in range(pool.shape[1]):
                        for jj in range(pool.shape[2]):
                            pool[i, ii, jj] = np.mean(a_c[i, ii * self.pool_size:(ii + 1) * self.pool_size,
                                                      jj * self.pool_size:(jj + 1) * self.pool_size])

                self.kernels[:, :, k] = kernel
                hs_c[:, :, :, k] = h_c
                as_c[:, :, :, k] = a_c
                pools[:, :, :, k] = pool

            p_flat = pools.T.reshape((-1, inputs.shape[0]))

            # full connection
            h_f = self.f_weights.dot(p_flat) + self.b_f
            a_f = self.method.activation.tanh(h_f)

            # output layer
            h_o = self.o_weights.dot(a_f) + self.b_o
            a_o = self.method.activation.softmax(h_o)

            loss = np.mean(self.method.loss.cross_entropy_loss(labels, a_o))
            if it % 10 == 0:
                tqdm.write(str(loss), sys.stderr)

            # backward
            # zo = self.method.loss.dcross_entropy_loss(labels, a_o) * self.method.activation.dsoftmax(h_o)
            zo = self.method.loss.dsoftmax_cross_entropy_loss(labels, a_o)
            zf = self.o_weights.T.dot(zo) * self.method.activation.dtanh(h_f)

            # upsample
            zs_f_ = zf.reshape((self.kernel_num, self.pool_dim[1], self.pool_dim[0], inputs.shape[0])).T
            as_c_ = np.zeros_like(as_c)
            for k in range(self.kernel_num):
                kernel = self.kernels[:, :, k]
                z_f_ = zs_f_[:, :, :, k]
                a_c_ = as_c_[:, :, :, k]
                h_c = hs_c[:, :, :, k]
                for i in range(inputs.shape[0]):
                    for ii in range(z_f_.shape[1]):
                        for jj in range(z_f_.shape[2]):
                            a_c_[i, ii * self.pool_size:(ii + 1) * self.pool_size,
                            jj * self.pool_size:(jj + 1) * self.pool_size] \
                                = z_f_[i, ii, jj] / (self.pool_size * self.pool_size)

                zc = a_c_ * self.method.activation.drelu(h_c)

                kernel_ = np.rot90(kernel, 2)
                d = np.zeros((zc.shape[0], zc.shape[1] - kernel_.shape[0] + 1, zc.shape[2] - kernel_.shape[1] + 1))
                for i in range(inputs.shape[0]):
                    d[i, :, :] = signal.convolve2d(zc[i, :, :], kernel_, 'valid')

                for i in range(inputs.shape[0]):
                    self.kernels[:, :, k] -= lr / a_c_.shape[1] * \
                                             signal.convolve2d(np.rot90(inputs[i, :, :, 0], 2), d[i, :, :], 'valid')

            self.b_o -= lr * np.mean(zo, axis=1, keepdims=True)
            self.b_f -= lr * np.mean(zf, axis=1, keepdims=True)

            self.o_weights -= lr / a_o.shape[1] * zo.dot(a_f.T)
            self.f_weights -= lr / a_f.shape[1] * zf.dot(p_flat.T)

        return a_o.T, loss, as_c


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt

    np.random.seed(1)
    mnist = input_data.read_data_sets('./mnist', one_hot=True, validation_size=55000,
                                      reshape=False)  # they has been normalized to range (0,1)

    batch_size = 20
    cnn = CNN(height=mnist.train.images.shape[1],
              width=mnist.train.images.shape[2],
              channel=mnist.train.images.shape[3],
              kernel_num=8,
              kernel_dim=5,
              pool_size=4,
              output_dim=mnist.train.labels.shape[1])

    x, y = mnist.train.next_batch(batch_size)
    pred, loss, conv = cnn.train(x, y, max_iter=200)
    r = np.argmax(y, axis=-1)
    p = np.argmax(pred, axis=-1)

    fig, axs = plt.subplots(8, 8)
    for i in range(8):
        axs[0][i].set_title(np.argmax(y[i]))
        for j in range(8):
            axs[j][i].set_xticks([])
            axs[j][i].set_yticks([])
            axs[j][i].imshow(conv[i][:, :, j])

    fig.set_size_inches(16, 16)
    # plt.savefig('../img/cnn_conv_layer.png')
    plt.show()
