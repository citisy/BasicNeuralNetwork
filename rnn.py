import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.animation import FFMpegWriter as Writer

sns.set(style="white", palette="muted", color_codes=True)

time_step = 10
batch_size = 10
input_size = 1
output_size = 1


def get_data():
    """
    data: [10, 10, 1]
    label: [10, 10, 1]
    """

    step = 0

    while True:
        # use sin predicts cos
        steps = np.linspace(0.25 * step * np.pi, (0.25 * step + 2) * np.pi, time_step * batch_size)
        x = np.sin(steps).reshape((-1, time_step, input_size))
        y = np.cos(steps).reshape((-1, time_step, output_size))

        yield x, y

        step += 1


def np_rnn():
    from utils.models import Sequential
    from utils.layers import Input, Dense, SimpleRNN
    from utils.activations import tanh
    from utils.optimizers import GradientDescent
    from utils.losses import mse

    ims = []
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    plt.ylim((-1.2, 1.2))

    gen_data = get_data()

    def callback(model, epoch):
        x, y = next(gen_data)
        if epoch % 2 == 0:
            pred = model.predict(x)
            loss = model.loss(pred, y)
            line1, = ax.plot(list(range(time_step * batch_size)), y.flatten(), 'r-', animated=True)
            line2, = ax.plot(list(range(time_step * batch_size)), pred.flatten(), 'b-', animated=True)
            text = ax.text(12.75, -0.75, 'step=%d\nloss=%f' % (epoch, loss))
            ims.append([line1, line2, text])

    model = Sequential()

    model.add(Input((batch_size, input_size)))
    model.add(SimpleRNN(32,
                        activation=tanh,
                        return_sequences=True))
    model.add(Dense(output_size))

    model.compile(optimizer=GradientDescent(lr=0.005),
                  loss=mse)

    model.fit_generator(get_data(),
                        steps_per_epoch=1,
                        epochs=100,
                        callbacks=lambda epoch: callback(model, epoch),
                        )

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000, repeat=True)
    # img_save_path = 'img/np_rnn.mp4'
    # fps = 5
    # ani.save(img_save_path, writer=Writer(fps=fps))
    plt.show()


def keras_rnn():
    from tensorflow.keras.layers import Input, Dense, SimpleRNN
    from tensorflow.keras.activations import tanh
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import LambdaCallback
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import mse

    ims = []
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    plt.ylim((-1.2, 1.2))

    gen_data = get_data()

    def callback(model, epoch, loss):
        x, y = next(gen_data)
        if epoch % 2 == 0:
            pred = model.predict(x)
            line1, = ax.plot(list(range(time_step * batch_size)), y.flatten(), 'r-', animated=True)
            line2, = ax.plot(list(range(time_step * batch_size)), pred.flatten(), 'b-', animated=True)
            text = ax.text(12.75, -0.75, 'step=%d\nloss=%f' % (epoch, loss))
            ims.append([line1, line2, text])

    model = Sequential()

    model.add(Input((batch_size, input_size)))

    model.add(SimpleRNN(32,
                        activation=tanh,
                        return_sequences=True,
                        ))

    model.add(Dense(output_size))

    model.compile(optimizer=Adam(),
                  loss=mse)

    model.summary()

    batch_print_callback = [
        LambdaCallback(on_epoch_end=lambda epoch, logs: callback(model, epoch, logs['loss']))
    ]

    model.fit_generator(get_data(),
                        steps_per_epoch=1,
                        epochs=200,
                        callbacks=batch_print_callback,
                        verbose=0,
                        )

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=1000, repeat=True)
    # img_save_path = 'img/keras_rnn.mp4'
    # fps = 5
    # ani.save(img_save_path, writer=Writer(fps=fps))
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    # np_rnn()
    keras_rnn()
