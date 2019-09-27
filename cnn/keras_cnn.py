from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.activations import relu, tanh, softmax
from keras.callbacks import LambdaCallback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)


class CNN:
    def __init__(self, height, width, channel, kernel_num, kernel_dim, pool_size, output_dim, lr=0.01):
        inputs = Input((height, width, channel))
        self.conv = Conv2D(kernel_num, kernel_dim, activation=relu)(inputs)
        pool = MaxPooling2D((pool_size, pool_size))(self.conv)
        flat = Flatten()(pool)
        dense = Dense(256, activation=tanh)(flat)
        outputs = Dense(output_dim, activation=softmax)(dense)
        optimizer = SGD(lr)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy')
        self.model.summary()

    def train(self, inputs, labels, epochs=100, is_show=False, save_path=None):
        inputs = np.array(inputs)
        labels = np.array(labels)

        if is_show:
            self.ims = []
            fig, self.axs = plt.subplots(8, 8)
            batch_print_callback = [LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.draw(epoch, logs['loss'], inputs, labels))]
        else:
            batch_print_callback = None

        self.model.fit(inputs, labels, epochs=epochs, verbose=3, callbacks=batch_print_callback)

        if is_show:
            fig.set_size_inches(16, 16)
            ani = animation.ArtistAnimation(fig, self.ims, interval=100, blit=True,
                                            repeat_delay=1000, repeat=True)
            if save_path:
                ani.save(save_path, writer='imagemagick')
            plt.show()

    def draw(self, epoch, loss, inputs, labels):
        if epoch % 10 == 0:
            layer_1 = K.function([self.model.layers[0].input], [self.model.layers[1].output])
            conv = layer_1([inputs, labels])[0]
            self.ims.append([])
            for i in range(8):
                self.axs[0][i].set_title(np.argmax(y[i]))
                for j in range(8):
                    self.axs[j][i].set_xticks([])
                    self.axs[j][i].set_yticks([])
                    self.ims[-1].append(self.axs[j][i].imshow(conv[i][:, :, j]))


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('./mnist', one_hot=True, validation_size=55000,
                                      reshape=False)

    batch_size = 20
    x, y = mnist.train.next_batch(batch_size)

    cnn = CNN(height=mnist.train.images.shape[1],
              width=mnist.train.images.shape[2],
              channel=mnist.train.images.shape[3],
              kernel_num=8,
              kernel_dim=5,
              pool_size=4,
              output_dim=mnist.train.labels.shape[1])

    # save_path = None
    save_path = '../img/keras_cnn_conv_layer.gif'
    cnn.train(x, y, epochs=100, is_show=True, save_path=save_path)
