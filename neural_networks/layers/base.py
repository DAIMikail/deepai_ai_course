"""
Layer Base Class
================
Tum katmanlarin miras alacagi soyut temel sinif.

Her katman su arayuzu uygulamalidir:
- forward(x): Ileri yayilim
- backward(dout): Geri yayilim (gradyan hesabi)
- get_params(): Parametreleri dondur (W, b)
- get_grads(): Gradyanlari dondur (dW, db)
"""

from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Tum katmanlar icin temel sinif.

    Ornek kullanim:
        class Dense(Layer):
            def forward(self, x):
                ...
            def backward(self, dout):
                ...
    """

    def __init__(self):
        self.cache = {}  # Forward sirasinda backward icin saklanacak degerler

    @abstractmethod
    def forward(self, x):
        """
        Ileri yayilim.

        Args:
            x: Girdi verisi

        Returns:
            Katman ciktisi
        """
        pass

    @abstractmethod
    def backward(self, dout):
        """
        Geri yayilim.

        Args:
            dout: Bir sonraki katmandan gelen gradyan

        Returns:
            Bir onceki katmana iletilecek gradyan (dx)
        """
        pass

    def get_params(self):
        """
        Katmanin egitilabilir parametrelerini dondur.

        Returns:
            dict: {'W': ..., 'b': ...} veya bos dict
        """
        return {}

    def get_grads(self):
        """
        Hesaplanan gradyanlari dondur.

        Returns:
            dict: {'dW': ..., 'db': ...} veya bos dict
        """
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}()"
