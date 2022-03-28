import numpy as np
from . import activations


class BaseLoss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class MAE(BaseLoss):
    def forward(self, y_pred, y_true):
        return np.abs(y_pred - y_true)

    def backward(self, y_pred, y_true):
        d = np.ones_like(y_pred)
        d[np.where(y_pred < y_true)] = -1
        return d


class MSE(BaseLoss):
    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true)) / 2

    def backward(self, y_pred, y_true):
        return np.mean(y_pred - y_true, axis=1, keepdims=True)


class CrossEntropyLoss(BaseLoss):
    def forward(self, y_pred, y_true):
        return np.mean(-y_true * np.log(y_pred))

    def backward(self, y_pred, y_true):
        return np.mean(-y_true / y_pred, axis=1, keepdims=True)


class SoftmaxCrossEntropyLoss(BaseLoss):
    def backward(self, y_pred, y_true, axis=1):
        """equals (dcross_entropy_loss * dsoftmax)"""
        y_pred = activations.softmax(y_pred, axis)
        y_pred[np.where(y_true == np.max(y_true))] -= 1
        return y_pred


class KLLoss(BaseLoss):
    def forward(self, y_pred, y_true):
        return y_true * np.log(y_true - y_pred)

    def backward(self, y_pred, y_true):
        return y_true / (y_pred - y_true)


class ExpLoss(BaseLoss):
    def forward(self, y_pred, y_true):
        return np.exp(-y_true * y_pred)

    def backward(self, y_pred, y_true):
        return -y_true * np.exp(-y_true * y_pred)


class HingeLoss(BaseLoss):
    def forward(self, y_pred, y_true):
        l = np.zeros_like(y_true)
        for i in range(len(y_true)):
            _ = 1 - y_true[i] * y_pred[i]
            if _ > 0:
                l[i] = _
        return l

    def backward(self, y_pred, y_true):
        l = np.zeros_like(y_true)
        for i in range(len(y_true)):
            _ = 1 - y_true[i] * y_pred[i]
            if _ > 0:
                l[i] = - y_true[i]
        return l


default_loss = BaseLoss()
mae = MAE()
mse = MSE()
cross_entropy_loss = CrossEntropyLoss()
softmax_cross_entropy_loss = SoftmaxCrossEntropyLoss()
kl_loss = KLLoss()
exp_loss = ExpLoss()
hinge_loss = HingeLoss()
