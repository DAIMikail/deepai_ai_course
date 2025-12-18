"""
Loss Fonksiyonlari
==================
Kayip fonksiyonlari - modelden AYRI.

PyTorch tarzi kullanim:
    loss_fn = CrossEntropyLoss()

    # Egitim dongusunde
    y_pred = model.forward(x)
    loss = loss_fn.forward(y_pred, y_true)
    dout = loss_fn.backward()
    model.backward(dout)
"""

import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """Tum loss fonksiyonlari icin temel sinif."""

    def __init__(self):
        self.cache = {}

    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        Loss degerini hesapla.

        Args:
            y_pred: Model tahmini
            y_true: Gercek etiketler

        Returns:
            loss: Skaler loss degeri
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Baslangic gradyanini hesapla.

        Returns:
            dout: Model'e iletilecek gradyan
        """
        pass


class CrossEntropyLoss(Loss):
    """
    Categorical Cross-Entropy Loss.

    Softmax ciktisi ile kullanilir.

    Formul:
        Forward:  L = -(1/m) * sum(Y * log(y_pred))
        Backward: dout = y_pred - Y  (Softmax + CE birlesimiyle)

    Ornek:
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(y_pred, y_true)  # y_pred: softmax ciktisi
        dout = loss_fn.backward()
    """

    def forward(self, y_pred, y_true):
        """
        Cross-entropy loss hesapla.

        Args:
            y_pred: Softmax ciktisi (n_classes, m)
            y_true: One-hot encoded etiketler (n_classes, m)

        Returns:
            loss: Ortalama loss (skaler)
        """
        m = y_true.shape[1]
        epsilon = 1e-8  # log(0) onleme

        # Loss hesapla
        log_probs = np.log(y_pred + epsilon)
        loss = -np.sum(y_true * log_probs) / m

        # Backward icin sakla
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true

        return loss

    def backward(self):
        """
        Baslangic gradyanini hesapla.

        Softmax + CrossEntropy kombinasyonunda:
            dL/dz = y_pred - y_true

        Returns:
            dout: (n_classes, m) boyutunda gradyan
        """
        y_pred = self.cache['y_pred']
        y_true = self.cache['y_true']

        # Softmax + CE turevi basitlesir
        dout = y_pred - y_true

        return dout

    def __repr__(self):
        return "CrossEntropyLoss()"
