from keras.layers import Input, Dense
from keras.activations import sigmoid
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from keras.callbacks import LambdaCallback

sns.set(style="white", palette="muted", color_codes=True)


class DNN:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, lr=0.05):
        inputs = Input(shape=(input_dim, ))
        a1 = Dense(hidden1_dim, activation=sigmoid)(inputs)
        a2 = Dense(hidden2_dim, activation=sigmoid)(a1)
        a3 = Dense(output_dim)(a2)

        self.model = Model(inputs=inputs, outputs=a3)

        optimizer = SGD(lr)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error')
        # self.model.summary()

    def train(self, inputs, labels, epochs=2000, is_show=False, save_path=None):
        inputs = np.array(inputs)
        labels = np.array(labels)

        if is_show:
            self.ims = []
            fig, self.ax = plt.subplots()
            fig.set_tight_layout(True)
            self.sca = self.ax.scatter(inputs, labels, c='r')
            batch_print_callback = [LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.draw(epoch, logs['loss'], inputs))]
        else:
            batch_print_callback = None

        self.model.fit(inputs, labels, epochs=epochs, callbacks=batch_print_callback, verbose=0)

        if is_show:
            ani = animation.ArtistAnimation(fig, self.ims, interval=100, blit=True,
                                            repeat_delay=1000, repeat=True)
            if save_path:
                ani.save(save_path, writer='imagemagick')
            plt.show()

    def draw(self, epoch, loss, inputs):
        if epoch % 20 == 0:
            line, = self.ax.plot(inputs, self.model.predict(inputs), c='b', animated=True)
            text = self.ax.text(1.25, 0.5, 'loss=%f' % loss)
            self.ims.append([self.sca, line, text])


if __name__ == '__main__':
    np.random.seed(2)  # seed is 4, the model can't work, why? maybe u will increase the hidden dim
    x = np.linspace(-2, 2, 100)
    noise = np.random.normal(0.0, 0.5, size=100)
    y = x ** 2 + noise
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    dnn = DNN(1, 4, 4, 1)
    # save_path = '../img/keras_dnn_1_4_4_1.gif'
    save_path = None
    dnn.train(x, y, is_show=True, save_path=save_path)




