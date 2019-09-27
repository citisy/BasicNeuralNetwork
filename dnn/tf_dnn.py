import numpy as np
import tensorflow as tf


class DNN:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, lr=0.05):
        self.inputs = tf.placeholder(tf.float32, [None, input_dim])
        self.labels = tf.placeholder(tf.int16, [None, output_dim])

        w1 = self.weight_variable([input_dim, hidden1_dim], 'w1')
        b1 = self.bias_variable([hidden1_dim, ], 'b1')
        w2 = self.weight_variable([hidden1_dim, hidden2_dim], 'w2')
        b2 = self.bias_variable([hidden2_dim, ], 'b2')
        w3 = self.weight_variable([hidden2_dim, output_dim], 'w3')
        b3 = self.bias_variable([output_dim, ], 'b3')

        h1 = tf.nn.xw_plus_b(self.inputs, w1, b1)   # xw or wx? seems that xw is easy to calculate.
        a1 = tf.nn.sigmoid(h1)

        h2 = tf.nn.xw_plus_b(a1, w2, b2)
        a2 = tf.nn.sigmoid(h2)

        h3 = tf.nn.xw_plus_b(a2, w3, b3)
        self.a3 = h3

        # # another easy way to build the network
        # a1 = tf.layers.dense(self.inputs, hidden1_dim, tf.nn.sigmoid)
        # a2 = tf.layers.dense(a1, hidden2_dim, tf.nn.sigmoid)
        # self.a3 = tf.layers.dense(a2, output_dim)

        self.loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.a3)
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

    def weight_variable(self, shape, name):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def bias_variable(self, shape, name):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import seaborn as sns

    sns.set(style="white", palette="muted", color_codes=True)

    ims = []
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    np.random.seed(2)  # seed is 4, the model can't work, why? maybe u will increase the hidden dim
    x = np.linspace(-2, 2, 100)
    noise = np.random.normal(0.0, 0.5, size=100)
    y = x ** 2 + noise
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    dnn = DNN(1, 4, 4, 1)
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())  # the local var is for accuracy_op
    sess.run(init_op)  # initialize var in graph

    sca = ax.scatter(x, y, c='r', animated=True)
    for i in range(5000):
        _, predict, loss = sess.run([dnn.train_op, dnn.a3, dnn.loss],
                                    feed_dict={dnn.inputs: x, dnn.labels: y})
        if i % 50 == 0:
            line, = ax.plot(x, predict, c='b', animated=True)
            text = ax.text(1.25, 0.5, 'loss=%f' % loss)
            ims.append([sca, line, text])
    ani = animation.ArtistAnimation(fig, ims, interval=2000 // len(ims), blit=True,
                                    repeat_delay=1000, repeat=True)
    # ani.save('../img/tf_dnn_1_4_4_1.gif', writer='imagemagick')
    plt.show()
