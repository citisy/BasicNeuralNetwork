import numpy as np
import tensorflow as tf


class RNN:
    def __init__(self, input_dim, time_step_dim, hidden_dim, output_dim, batch_size, lr=0.002):
        self.inputs = tf.placeholder(tf.float32, [None, time_step_dim, input_dim])  # shape(batch, 5, 1)
        self.labels = tf.placeholder(tf.float32, [None, time_step_dim, output_dim])  # input y

        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim)
        self.init_s = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)  # very first hidden state
        a1, self.final_s = tf.nn.dynamic_rnn(
            rnn_cell,
            self.inputs,
            initial_state=self.init_s,
            time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
        )
        a1_2D = tf.reshape(a1, [-1, hidden_dim])  # reshape 3D output to 2D for fully connected layer
        a2 = tf.layers.dense(a1_2D, output_dim)
        self.outputs = tf.reshape(a2, [-1, time_step_dim, output_dim])  # reshape back to 3D

        self.loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.outputs)  # compute cost
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import seaborn as sns

    sns.set(style="white", palette="muted", color_codes=True)

    ims = []
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    plt.ylim((-1.2, 1.2))

    time_step_dim = 10
    batch_size = 10
    input_dim = output_dim = 1
    hidden_dim = 32
    model = RNN(input_dim, time_step_dim, hidden_dim, output_dim, batch_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # initialize var in graph
    for step in range(900):
        # use sin predicts cos
        steps = np.linspace(0.25 * step * np.pi, (0.25 * step + 2) * np.pi, time_step_dim*batch_size)
        x = np.sin(steps).reshape((-1, time_step_dim, input_dim))  # shape (batch, time_step, input_size)
        y = np.cos(steps).reshape((-1, time_step_dim, output_dim))

        if 'final_s' not in globals():  # first state, no any hidden state
            feed_dict = {model.inputs: x, model.labels: y}
        else:  # has hidden state, so pass it to rnn
            feed_dict = {model.inputs: x, model.labels: y, model.init_s: final_s}
        _, pred, final_s, loss = sess.run([model.train_op, model.outputs, model.final_s, model.loss], feed_dict)  # train
        if step % 9 == 0:
            line1, = ax.plot(list(range(time_step_dim*batch_size)), y.flatten(), 'r-', animated=True)
            line2, = ax.plot(list(range(time_step_dim*batch_size)), pred.flatten(), 'b-', animated=True)
            text = ax.text(12.75, -0.75, 'step=%d\nloss=%f' % (step, loss))
            ims.append([line1, line2, text])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000, repeat=True)
    # ani.save('../img/tf_rnn.gif', writer='imagemagick')
    plt.show()
