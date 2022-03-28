import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.animation import FFMpegWriter as Writer

sns.set(style="white", palette="muted", color_codes=True)


def get_dataset():
    """
    data: [100, 1]
    label: [100, 1]
    """
    size = 100
    data = np.linspace(-2, 2, size)
    noise = np.random.normal(0.0, 0.5, size=size)
    label = data ** 2 + noise
    data = data.reshape((-1, 1))
    label = label.reshape((-1, 1))

    return data, label


def np_dnn():
    from utils.models import Sequential
    from utils.layers import Input, Dense
    from utils.activations import sigmoid
    from utils.optimizers import GradientDescent
    from utils.losses import mse

    def callback(model, epoch, x, y):
        if not epoch % 100:
            predict = model.predict(x)

            loss = model.loss(predict, y)

            line, = ax.plot(x, predict, c='b', animated=True)
            text = ax.text(1.25, 0.5, 'loss=%f' % loss)
            ims.append([sca, line, text])

    data, label = get_dataset()

    # draw pictures
    fig, ax = plt.subplots()
    ims = []
    fig.set_tight_layout(True)
    sca = ax.scatter(data, label, c='r', animated=True)

    unit_list = [8, 8]

    model = Sequential()

    # input layer
    model.add(Input(data.shape[1]))

    # hidden layer
    for unit in unit_list:
        model.add(Dense(unit, activation=sigmoid))

    # output layer
    model.add(Dense(label.shape[1]))

    model.compile(
        optimizer=GradientDescent(lr=.005),
        loss=mse,
    )

    model.fit(data, label,
              batch_size=64,
              epochs=10000,
              callbacks=lambda epoch: callback(model, epoch, data, label)
              )

    # draw pictures
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000, repeat=True)
    img_save_path = 'img/np_dnn.mp4'
    fps = 10
    ani.save(img_save_path, writer=Writer(fps=fps))
    plt.show()


def keras_dnn():
    """
    keras == 2.8.0
    tensorflow == 2.8.0
    """
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.activations import sigmoid
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.losses import mse
    from tensorflow.keras.callbacks import LambdaCallback

    def callback(model, epoch, x, loss):
        if not epoch % 20:
            predict = model.predict(x)
            line, = ax.plot(x, predict, c='b', animated=True)
            text = ax.text(1.25, 0.5, 'loss=%f' % loss)
            ims.append([sca, line, text])

    data, label = get_dataset()

    # draw pictures
    fig, ax = plt.subplots()
    ims = []
    fig.set_tight_layout(True)
    sca = ax.scatter(data, label, c='r', animated=True)

    batch_print_callback = [
        LambdaCallback(on_epoch_end=lambda epoch, logs: callback(model, epoch, data, logs['loss']))
    ]

    unit_list = [8, 8]

    model = Sequential()

    # input layer
    model.add(Input(shape=(data.shape[1],)))

    # hidden layer
    for unit in unit_list:
        model.add(Dense(unit, activation=sigmoid))

    # output layer
    model.add(Dense(label.shape[1]))

    model.compile(
        optimizer=SGD(learning_rate=0.05),
        loss=mse,
    )
    # model.summary()

    model.fit(data, label,
              epochs=2000,
              batch_size=64,
              callbacks=batch_print_callback,
              verbose=0,
              )

    # draw pictures
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000, repeat=True)
    img_save_path = 'img/keras_dnn.mp4'
    fps = 10
    ani.save(img_save_path, writer=Writer(fps=fps))
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    np_dnn()
    # keras_dnn()
