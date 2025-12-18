"""
Sequential Model
================
Katmanlari sirali sekilde birlestiren container sinifi.

Ornek:
    model = Sequential([
        Dense(784, 128),
        ReLU(),
        Dense(128, 10),
        Softmax()
    ])

    # Ileri yayilim
    out = model.forward(x)

    # Geri yayilim
    model.backward(dout)
"""

class Sequential:
    """
    Katmanlari sirali tutan model container'i.

    Args:
        layers: Layer listesi

    Ornek:
        model = Sequential([
            Dense(784, 128),
            ReLU(),
            Dense(128, 10),
            Softmax()
        ])
    """

    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        """
        Modele katman ekle.

        Args:
            layer: Eklenecek Layer nesnesi
        """
        # Duck typing: forward ve backward metodlari varsa kabul et
        if not (hasattr(layer, 'forward') and hasattr(layer, 'backward')):
            raise TypeError(f"Layer forward/backward metodlarina sahip olmali: {type(layer)}")
        self.layers.append(layer)

    def forward(self, x):
        """
        Ileri yayilim - tum katmanlardan sirayla gecer.

        Args:
            x: Girdi verisi

        Returns:
            Modelin ciktisi
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        Geri yayilim - ters sirada gradyan yayilimi.

        Args:
            dout: Loss'tan gelen baslangic gradyani

        Returns:
            Girdiye gore gradyan (genelde kullanilmaz)
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_params(self):
        """
        Tum katmanlarin parametrelerini topla.

        Returns:
            list: Her katmanin parametrelerini iceren dict listesi
        """
        params = []
        for layer in self.layers:
            layer_params = layer.get_params()
            if layer_params:  # Bos degilse ekle
                params.append({
                    'layer': layer,
                    'params': layer_params
                })
        return params

    def get_grads(self):
        """
        Tum katmanlarin gradyanlarini topla.

        Returns:
            list: Her katmanin gradyanlarini iceren dict listesi
        """
        grads = []
        for layer in self.layers:
            layer_grads = layer.get_grads()
            if layer_grads:  # Bos degilse ekle
                grads.append({
                    'layer': layer,
                    'grads': layer_grads
                })
        return grads

    def __repr__(self):
        lines = ["Sequential(["]
        for layer in self.layers:
            lines.append(f"    {layer},")
        lines.append("])")
        return "\n".join(lines)

    def __len__(self):
        return len(self.layers)
