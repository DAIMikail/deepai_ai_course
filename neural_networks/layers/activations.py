"""
Aktivasyon Katmanlari
=====================
ReLU, Softmax gibi aktivasyon fonksiyonlari.

Bu katmanlarin egitilabilir parametresi yoktur (W, b yok).
Sadece element-wise donusum uygularlar.
"""

import numpy as np
from .base import Layer


class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) aktivasyonu.

    Formul:
        Forward:  a = max(0, z)
        Backward: dz = dout * (z > 0)

    Ornek:
        layer = ReLU()
        a = layer.forward(z)       # z: (n, m)
        dz = layer.backward(dout)  # dout: (n, m)
    """

    def forward(self, z):
        """
        Ileri yayilim: a = max(0, z)

        Args:
            z: Girdi (herhangi bir boyut)

        Returns:
            a: Cikti (ayni boyut)
        """
        self.cache['z'] = z
        return np.maximum(0, z)

    def backward(self, dout):
        """
        Geri yayilim: dz = dout * ReLU'(z)

        Args:
            dout: Sonraki katmandan gelen gradyan

        Returns:
            dz: Onceki katmana iletilecek gradyan
        """
        z = self.cache['z']
        dz = dout * (z > 0).astype(float)
        return dz

    def __repr__(self):
        return "ReLU()"


class Softmax(Layer):
    """
    Softmax aktivasyonu.

    Formul:
        Forward: a_i = exp(z_i) / sum(exp(z_j))

    Not:
        Softmax genelde CrossEntropyLoss ile birlikte kullanilir.
        Bu durumda backward hesabi basitlestigi icin (a - y),
        backward burada bos birakilabilir veya loss ile birlesik kullanilir.

    Ornek:
        layer = Softmax()
        probs = layer.forward(z)  # z: (n_classes, m)
    """

    def forward(self, z):
        """
        Ileri yayilim: softmax(z)

        Numerical stability icin max deger cikarilir.

        Args:
            z: Raw skorlar (n_classes, m)

        Returns:
            a: Olasiliklar (n_classes, m), her sutun toplami = 1
        """
        # Numerical stability
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        a = exp_z / np.sum(exp_z, axis=0, keepdims=True)

        self.cache['a'] = a
        return a

    def backward(self, dout):
        """
        Geri yayilim.

        Not: CrossEntropyLoss ile birlikte kullanildiginda,
        gradyan hesabi basitlesir: dz = a - y
        Bu nedenle bu metod genelde dogrudan cagrilmaz.

        Args:
            dout: Sonraki katmandan gelen gradyan

        Returns:
            dz: Onceki katmana iletilecek gradyan
        """
        # Softmax + CrossEntropy birlikte kullanildiginda
        # bu metod atlanir, dogrudan (a - y) kullanilir
        # Burada sadece pass-through yapiyoruz
        return dout

    def __repr__(self):
        return "Softmax()"
