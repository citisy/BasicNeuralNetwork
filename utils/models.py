from . import layers
from . import losses
from . import optimizers


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self,
            layer: layers.BaseLayer
            ):
        self.layers.append(layer)

    def compile(self,
                optimizer: optimizers.BaseOptimizer = optimizers.gd,
                loss: losses.BaseLoss = losses.mse
                ):
        self.optimizer = optimizer
        self.loss = loss

        for i, layer in enumerate(self.layers):
            if i:
                layer.initializer(self.layers[i-1].units, optimizer)

    def fit(self, x, y, epochs=100, batch_size=100, callbacks=None):
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                self._fit(x[i: i + batch_size], y[i: i + batch_size], epoch, callbacks)

    def fit_generator(self, generator, steps_per_epoch=1,
                      epochs=100, callbacks=None):
        for epoch in range(epochs):
            if not epoch % steps_per_epoch:
                x, y = next(generator)
                self._fit(x, y, epoch, callbacks)

    def _fit(self, x, y, epoch, callbacks):
        # forward
        for layer in self.layers:
            if isinstance(layer, layers.Input):
                continue

            x = layer.forward(x)

        # backward
        dh = self.loss.backward(x, y)
        for layer in self.layers[::-1]:
            if isinstance(layer, layers.Input):
                continue

            dh = layer.backward(dh)['x']

        if callbacks:
            callbacks(epoch)

    def predict(self, x):
        h = x

        for layer in self.layers:
            if isinstance(layer, layers.Input):
                continue

            h = layer.forward(h)

        return h
