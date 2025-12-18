"""
Optimizers
==========
Parametre guncelleme algoritmalari - modelden AYRI.

Kullanim:
    optimizer = SGD(model, lr=0.1)

    # Egitim dongusunde
    y_pred = model.forward(x)
    loss = loss_fn.forward(y_pred, y_true)
    dout = loss_fn.backward()
    model.backward(dout)
    optimizer.step()  # Parametreleri guncelle
"""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Tum optimizer'lar icin temel sinif."""

    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    @abstractmethod
    def step(self):
        """Parametreleri guncelle."""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent.

    Formul:
        W = W - lr * dW
        b = b - lr * db

    Args:
        model: Sequential model (veya get_params/get_grads olan herhangi bir model)
        lr: Learning rate (ogrenme orani)

    Ornek:
        optimizer = SGD(model, lr=0.1)

        # Egitim dongusunde
        model.backward(dout)
        optimizer.step()
    """

    def __init__(self, model, lr=0.01):
        super().__init__(model, lr)

    def step(self):
        """
        Gradient descent adimi.

        Tum katmanlarin parametrelerini gunceller:
            param = param - lr * grad
        """
        for layer in self.model.layers:
            params = layer.get_params()
            grads = layer.get_grads()

            if params and grads:
                # W guncelle
                if 'W' in params and 'dW' in grads:
                    layer.W = layer.W - self.lr * grads['dW']

                # b guncelle
                if 'b' in params and 'db' in grads:
                    layer.b = layer.b - self.lr * grads['db']

    def __repr__(self):
        return f"SGD(lr={self.lr})"
