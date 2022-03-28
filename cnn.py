import numpy as np
import matplotlib.pyplot as plt


def get_data():
    """
    data: [20, 28, 28, 1]
    label: [20, 10]
    """
    from tensorflow import keras

    num_classes = 10

    (data, label), _ = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    data = data[:20].astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    data = np.expand_dims(data, -1)

    # convert class vectors to binary class matrices
    label = keras.utils.to_categorical(label[:20], num_classes)

    return data, label


def np_cnn():
    from utils.activations import relu
    from utils.layers import Input, Conv2D, AvgPool2D, Flatten, Dense
    from utils.models import Sequential
    from utils.activations import softmax
    from utils.losses import softmax_cross_entropy_loss
    from utils.optimizers import GradientDescent

    data, label = get_data()
    units_list = [8]

    model = Sequential()

    model.add(Input(data.shape[1:]))

    for unit in units_list:
        model.add(Conv2D(unit, kernel_size=(5, 5), activation=relu))
        model.add(AvgPool2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dense(label.shape[1]))

    model.compile(
        loss=softmax_cross_entropy_loss,
        optimizer=GradientDescent(0.005)
    )
    model.fit(data, label, epochs=200)

    conv = model.layers[1].variable['h']
    pred = softmax(model.predict(data))

    fig, axs = plt.subplots(9, 8)
    for i in range(8):
        axs[0][i].set_title(np.argmax(label[i]))

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].imshow(data[i])

        for j in range(8):
            axs[j + 1][i].set_xticks([])
            axs[j + 1][i].set_yticks([])
            axs[j + 1][i].imshow(conv[i][:, :, j])

        axs[-1][i].set_xlabel(np.argmax(pred[i]))

    fig.set_size_inches(16, 16)
    plt.savefig('img/np_cnn.png')
    plt.show()


def keras_cnn():
    from tensorflow.keras.activations import relu, softmax
    from tensorflow.keras.layers import Input, Conv2D, AvgPool2D, Flatten, Dropout, Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.losses import categorical_crossentropy
    from tensorflow.keras.optimizers import Adam

    data, label = get_data()

    units_list = [8]

    model = Sequential()

    model.add(Input(shape=data.shape[1:]))

    for unit in units_list:
        model.add(Conv2D(unit, kernel_size=(5, 5), activation=relu))
        model.add(AvgPool2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dense(label.shape[1], activation=softmax))

    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam()
    )
    model.summary()

    model.fit(data, label,
              epochs=100,
              verbose=0,
              )

    from keras import backend as K

    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])
    conv = get_layer_output([data])[0]
    pred = model.predict(data)

    fig, axs = plt.subplots(9, 8)
    for i in range(8):
        axs[0][i].set_title(np.argmax(label[i]))

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].imshow(data[i])

        for j in range(8):
            axs[j + 1][i].set_xticks([])
            axs[j + 1][i].set_yticks([])
            axs[j + 1][i].imshow(conv[i][:, :, j])

        axs[-1][i].set_xlabel(np.argmax(pred[i]))

    fig.set_size_inches(16, 16)
    plt.savefig('img/keras_cnn.png')
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    np_cnn()
    # keras_cnn()
