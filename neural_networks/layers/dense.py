"""
Dense (Fully Connected) Layer
=============================
Tam bagli katman implementasyonu.

Formul:
    Forward:  z = W · x + b
    Backward: dW = (1/m) * dout · x.T
              db = (1/m) * sum(dout, axis=1)
              dx = W.T · dout

Boyutlar:
    x:    (n_in, m)   - m ornek, n_in ozellik
    W:    (n_out, n_in)
    b:    (n_out, 1)
    z:    (n_out, m)
    dout: (n_out, m)  - sonraki katmandan gelen gradyan
    dx:   (n_in, m)   - onceki katmana iletilecek gradyan
"""

import numpy as np
from .base import Layer


class Dense(Layer):
    """
    Fully Connected (Dense) katman.

    Args:
        n_in: Girdi boyutu
        n_out: Cikti boyutu (noron sayisi)
        seed: Random seed (tekrarlanabilirlik icin)

    Ornek:
        layer = Dense(784, 128)
        out = layer.forward(x)      # x: (784, m)
        dx = layer.backward(dout)   # dout: (128, m)
    """

    def __init__(self, n_in, n_out, seed=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        # He initialization
        if seed is not None:
            np.random.seed(seed)

        self.W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((n_out, 1))

        # Gradyanlar (backward'da hesaplanacak)
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Ileri yayilim: z = W · x + b

        Args:
            x: Girdi (n_in, m)

        Returns:
            z: Cikti (n_out, m)
        """
        self.cache['x'] = x

        z = np.dot(self.W, x) + self.b
        return z

    def backward(self, dout):
        """
        Geri yayilim.

        Args:
            dout: Sonraki katmandan gelen gradyan (n_out, m)

        Returns:
            dx: Onceki katmana iletilecek gradyan (n_in, m)
        """
        x = self.cache['x']
        m = x.shape[1]

        # Parametre gradyanlari
        self.dW = (1 / m) * np.dot(dout, x.T)
        self.db = (1 / m) * np.sum(dout, axis=1, keepdims=True)

        # Onceki katmana iletilecek gradyan
        dx = np.dot(self.W.T, dout)

        return dx

    def get_params(self):
        """Eğitilebilir parametreleri döndür."""
        return {'W': self.W, 'b': self.b}

    def get_grads(self):
        """Hesaplanan gradyanlari döndür."""
        return {'dW': self.dW, 'db': self.db}

    def __repr__(self):
        return f"Dense({self.n_in}, {self.n_out})"
