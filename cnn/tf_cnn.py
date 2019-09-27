import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys

np.random.seed(2)


class CNN:
    def __init__(self, height, width, channel, kernel_num, kernel_dim, pool_size, output_dim, lr=1e-3):
        self.lr = lr

        tf.reset_default_graph()

        self.inputs = tf.placeholder(tf.float32, [None, height, width, channel]) / 255.
        self.labels = tf.placeholder(tf.int32, [None, output_dim])

        # CNN
        self.conv = tf.layers.conv2d(
            inputs=self.inputs,
            filters=kernel_num,
            kernel_size=kernel_dim,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        pool = tf.layers.max_pooling2d(
            self.conv,
            pool_size=pool_size,
            strides=pool_size,
        )

        flat = tf.reshape(pool, [-1, pool.shape[1] * pool.shape[2] * pool.shape[3]])  # -> (14*14*16, 1)
        self.output = tf.layers.dense(flat, output_dim)  # output layer

        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.output)  # compute cost
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    mnist = input_data.read_data_sets('./mnist', one_hot=True, validation_size=55000,
                                      reshape=False)

    batch_size = 20
    cnn = CNN(height=mnist.train.images.shape[1],
                width=mnist.train.images.shape[2],
                channel=mnist.train.images.shape[3],
                kernel_num=8,
                kernel_dim=5,
                pool_size=4,
                output_dim=mnist.train.labels.shape[1])
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)  # initialize var in graph

    fig, axs = plt.subplots(8, 8)
    ims = []
    x, y = mnist.train.next_batch(batch_size)
    for step in tqdm(range(100)):
        _, loss, conv = sess.run([cnn.train_op, cnn.loss, cnn.conv],
                                 {cnn.inputs: x, cnn.labels: y})
        if step % 10 == 0:
            tqdm.write(str(loss), sys.stderr)
            ims.append([])
            for i in range(8):
                axs[0][i].set_title(np.argmax(y[i]))
                for j in range(8):
                    axs[j][i].set_xticks([])
                    axs[j][i].set_yticks([])
                    ims[-1].append(axs[j][i].imshow(conv[i][:, :, j]))


    fig.set_size_inches(16, 16)
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000, repeat=True)
    ani.save('../img/tf_cnn_conv_layer.gif', writer='imagemagick')
    plt.show()
