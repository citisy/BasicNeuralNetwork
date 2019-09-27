from keras.layers import Input, Dense, SimpleRNN
from keras.activations import tanh
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from keras.callbacks import LambdaCallback

sns.set(style="white", palette="muted", color_codes=True)


class RNN:
    def __init__(self, input_dim, time_step_dim, hidden_dim, output_dim, lr=0.01):
        inputs = Input((time_step_dim, input_dim,))
        a1 = SimpleRNN(hidden_dim,
                       activation=tanh,
                       return_sequences=True)(inputs)
        outputs = Dense(output_dim, )(a1)

        self.model = Model(inputs=inputs, outputs=outputs)

        optimizers = Adam(lr)

        self.model.compile(optimizer=optimizers,
                           loss='mean_squared_error')
        # self.model.summary()

    def train(self, inputs, labels, epochs=2000, is_show=False, save_path=None):
        inputs = np.array(inputs)
        labels = np.array(labels)

        if is_show:
            self.ims = []
            fig, self.ax = plt.subplots()
            fig.set_tight_layout(True)
            plt.ylim((-1.2, 1.2))
            batch_print_callback = [LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.draw(epoch, logs['loss'], inputs, labels))]
        else:
            batch_print_callback = None

        self.model.fit(inputs, labels, epochs=epochs, verbose=0, callbacks=batch_print_callback)

        if is_show:
            ani = animation.ArtistAnimation(fig, self.ims, interval=100, blit=True,
                                            repeat_delay=1000, repeat=True)
            if save_path:
                ani.save(save_path, writer='imagemagick')
            plt.show()

    def draw(self, epoch, loss, inputs, labels):
        if epoch % 20 == 0:
            pred = self.model.predict(inputs)
            size = inputs.shape[0] * inputs.shape[1]
            line1, = plt.plot(list(range(size)), labels.flatten(), 'r-', animated=True)
            line2, = plt.plot(list(range(size)), pred.flatten(), 'b-', animated=True)
            text = self.ax.text(1.25, 0.5, 'loss=%f' % loss)
            self.ims.append([line1, line2, text])


if __name__ == '__main__':
    time_step_dim = 10
    batch_size = 10
    input_dim = output_dim = 1
    hidden_dim = 32

    steps = np.linspace(0, 2 * np.pi, time_step_dim * batch_size)
    x = np.sin(steps).reshape((-1, time_step_dim, input_dim))  # shape (batch, time_step, input_size)
    y = np.cos(steps).reshape((-1, time_step_dim, output_dim))

    rnn = RNN(input_dim, time_step_dim, hidden_dim, output_dim)
    # save_path = '../img/keras_rnn.gif'
    save_path = None
    rnn.train(x, y, is_show=True, save_path=save_path)

