# Neural Networks from Scratch - Implementation Guide

Bu dokumanin amaci, ogrencilere sifirdan bir neural network framework'u implement etmeyi ogretmektir.
Tum kodlar pure NumPy ile yazilacaktir.

---

## Genel Yapi

```
neural_networks/
├── layers/
│   ├── __init__.py
│   ├── base.py          # (1) Soyut temel sinif
│   ├── dense.py         # (2) Fully Connected katman
│   ├── activations.py   # (3) ReLU, Softmax
│   └── conv.py          # (7) Conv2D, MaxPool2D, Flatten
├── losses/
│   ├── __init__.py
│   └── losses.py        # (4) CrossEntropyLoss
├── optimizers/
│   ├── __init__.py
│   └── optimizers.py    # (5) SGD
├── models/
│   ├── __init__.py
│   └── sequential.py    # (6) Model container
├── test_fashion_mnist.py     # ANN testi
└── test_cnn_fashion_mnist.py # CNN testi
```

---

# BOLUM 1: ANN (Artificial Neural Network)

## Adim 1: Base Layer (layers/base.py)

**Amac:** Tum katmanlarin miras alacagi soyut temel sinif.

**Temel Kavramlar:**
- Abstract Base Class (ABC)
- forward/backward interface
- cache mekanizmasi

**Implement Edilecekler:**
```python
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.cache = {}  # Forward sirasinda backward icin degerler saklanir

    @abstractmethod
    def forward(self, x):
        """Ileri yayilim"""
        pass

    @abstractmethod
    def backward(self, dout):
        """Geri yayilim - gradyan hesabi"""
        pass

    def get_params(self):
        """Egitilabilir parametreler (W, b)"""
        return {}

    def get_grads(self):
        """Hesaplanan gradyanlar (dW, db)"""
        return {}
```

**Onemli Noktalar:**
- `cache`: Forward'da hesaplanan degerleri saklar (backward'da lazim olacak)
- `get_params()` ve `get_grads()`: Default olarak bos dict dondurur (aktivasyonlarin parametresi yok)

---

## Adim 2: Dense Layer (layers/dense.py)

**Amac:** Fully connected (tam bagli) katman.

**Matematiksel Formul:**
```
Forward:  z = W · x + b
Backward: dW = (1/m) * dout · x.T
          db = (1/m) * sum(dout, axis=1)
          dx = W.T · dout
```

**Boyutlar:**
```
x:    (n_in, m)     - m ornek, n_in ozellik
W:    (n_out, n_in)
b:    (n_out, 1)
z:    (n_out, m)
dout: (n_out, m)    - sonraki katmandan gelen gradyan
dx:   (n_in, m)     - onceki katmana iletilecek gradyan
```

**Implement Edilecekler:**
```python
import numpy as np
from .base import Layer

class Dense(Layer):
    def __init__(self, n_in, n_out, seed=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        # He initialization
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((n_out, 1))

        self.dW = None
        self.db = None

    def forward(self, x):
        self.cache['x'] = x
        z = np.dot(self.W, x) + self.b
        return z

    def backward(self, dout):
        x = self.cache['x']
        m = x.shape[1]

        self.dW = (1 / m) * np.dot(dout, x.T)
        self.db = (1 / m) * np.sum(dout, axis=1, keepdims=True)
        dx = np.dot(self.W.T, dout)

        return dx

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def get_grads(self):
        return {'dW': self.dW, 'db': self.db}
```

**Onemli Noktalar:**
- **He initialization:** ReLU ile kullanildiginda iyi calisir: `W * sqrt(2/n_in)`
- **Neden 1/m?**: Batch uzerinden ortalama gradyan almak icin
- **keepdims=True:** Broadcasting icin boyutu korur

---

## Adim 3: Aktivasyon Katmanlari (layers/activations.py)

**Amac:** Non-linearity eklemek (ReLU, Softmax).

### 3.1 ReLU

**Formul:**
```
Forward:  a = max(0, z)
Backward: dz = dout * (z > 0)
```

**Implement:**
```python
class ReLU(Layer):
    def forward(self, z):
        self.cache['z'] = z
        return np.maximum(0, z)

    def backward(self, dout):
        z = self.cache['z']
        dz = dout * (z > 0).astype(float)
        return dz
```

### 3.2 Softmax

**Formul:**
```
Forward: a_i = exp(z_i) / sum(exp(z_j))
```

**Numerical Stability:**
- `z - max(z)` yaparak overflow onlenir

**Implement:**
```python
class Softmax(Layer):
    def forward(self, z):
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        a = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        self.cache['a'] = a
        return a

    def backward(self, dout):
        # Softmax + CrossEntropy birlikte kullanildiginda
        # gradyan basitlesir, dogrudan pass-through
        return dout
```

**Onemli Noktalar:**
- **Numerical stability:** Buyuk sayilarin exp'i overflow yapar
- **Softmax backward:** CrossEntropy ile birlikte kullanildiginda `a - y` olur (loss'ta hesaplanir)

---

## Adim 4: Loss Fonksiyonu (losses/losses.py)

**Amac:** Kayip hesaplama ve baslangic gradyani uretme.

### CrossEntropyLoss

**Formul:**
```
Forward:  L = -(1/m) * sum(Y * log(y_pred))
Backward: dout = y_pred - Y  (Softmax + CE birlesimi)
```

**Implement:**
```python
class CrossEntropyLoss:
    def __init__(self):
        self.cache = {}

    def forward(self, y_pred, y_true):
        m = y_true.shape[1]
        epsilon = 1e-8  # log(0) onleme

        log_probs = np.log(y_pred + epsilon)
        loss = -np.sum(y_true * log_probs) / m

        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true

        return loss

    def backward(self):
        y_pred = self.cache['y_pred']
        y_true = self.cache['y_true']
        dout = y_pred - y_true
        return dout
```

**Onemli Noktalar:**
- **epsilon:** log(0) = -inf onlemek icin
- **Softmax + CE turevi:** Cok guzel basitlesir: `dL/dz = a - y`

---

## Adim 5: Optimizer (optimizers/optimizers.py)

**Amac:** Parametreleri guncellemek.

### SGD (Stochastic Gradient Descent)

**Formul:**
```
W = W - lr * dW
b = b - lr * db
```

**Implement:**
```python
class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            params = layer.get_params()
            grads = layer.get_grads()

            if params and grads:
                if 'W' in params and 'dW' in grads:
                    layer.W = layer.W - self.lr * grads['dW']
                if 'b' in params and 'db' in grads:
                    layer.b = layer.b - self.lr * grads['db']
```

**Onemli Noktalar:**
- Optimizer modeli alir, katmanlar uzerinde iterate eder
- Sadece parametresi olan katmanlar guncellenir

---

## Adim 6: Sequential Model (models/sequential.py)

**Amac:** Katmanlari sirali birlestiren container.

**Implement:**
```python
class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
```

**Onemli Noktalar:**
- Forward: Sirayla ileri git
- Backward: TERS sirada geri git (chain rule)

---

## TEST: ANN ile Fashion MNIST (test_fashion_mnist.py)

```python
# Model
model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])

loss_fn = CrossEntropyLoss()
optimizer = SGD(model, lr=0.1)

# Training loop
for epoch in range(200):
    # Forward
    y_pred = model.forward(X_train)

    # Loss
    loss = loss_fn.forward(y_pred, Y_train)

    # Backward
    dout = loss_fn.backward()
    model.backward(dout)

    # Update
    optimizer.step()
```

**Beklenen Sonuc:** ~85-87% test accuracy

---

# BOLUM 2: CNN (Convolutional Neural Network)

## Adim 7: Convolutional Layers (layers/conv.py)

### 7.1 im2col Optimizasyonu

**Neden im2col?**
- Naive nested loop cok yavas
- im2col: Konvolusyonu matris carpimina donusturur
- ~10-50x hiz artisi

**Fikir:**
```
Her konvolusyon penceresi -> bir sutun
Tum pencereler -> buyuk matris
Sonra tek np.dot ile hesapla
```

**Implement:**
```python
def im2col(x, kH, kW, stride=1, padding=0):
    """
    x: (m, C, H, W) -> col: (C*kH*kW, H_out*W_out*m)
    """
    m, C, H, W = x.shape

    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)))

    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1

    # Stride tricks - cok verimli
    shape = (m, C, kH, kW, H_out, W_out)
    s = x.strides
    strides = (s[0], s[1], s[2], s[3], s[2]*stride, s[3]*stride)

    cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    cols = cols.transpose(1,2,3,4,5,0).reshape(C*kH*kW, -1)

    return cols
```

### 7.2 Conv2D

**Boyutlar:**
```
Input:  (m, C_in, H, W)
Kernel: (C_out, C_in, kH, kW)
Output: (m, C_out, H_out, W_out)

H_out = (H - kH + 2*padding) // stride + 1
```

**Forward:**
```python
def forward(self, x):
    x_col = im2col(x, kH, kW, stride, padding)
    W_row = self.W.reshape(C_out, -1)
    out = np.dot(W_row, x_col) + self.b
    out = out.reshape(C_out, H_out, W_out, m).transpose(3,0,1,2)
    return out
```

**Backward:**
```python
def backward(self, dout):
    # dW = dout @ x_col.T
    # dx = col2im(W.T @ dout)
    ...
```

### 7.3 MaxPool2D

**Formul:**
```
Her pencerede max deger al
Backward: Sadece max konumuna gradyan ilet
```

**Implement:**
```python
def forward(self, x):
    # Her pencerede max bul
    for i, j in pencereler:
        out[:,:,i,j] = np.max(x_slice, axis=(2,3))
    return out

def backward(self, dout):
    # Max konumuna gradyan ilet (mask kullan)
    mask = (x_slice == max_val)
    dx += mask * dout
```

### 7.4 Flatten

**Amac:** Conv ciktisini Dense'e baglamak.

**Boyut Donusumu:**
```
Input:  (m, C, H, W)  - CNN format
Output: (C*H*W, m)    - Dense format
```

**Implement:**
```python
def forward(self, x):
    self.cache['input_shape'] = x.shape
    m = x.shape[0]
    return x.reshape(m, -1).T

def backward(self, dout):
    return dout.T.reshape(self.cache['input_shape'])
```

---

## TEST: CNN ile Fashion MNIST (test_cnn_fashion_mnist.py)

```python
model = Sequential([
    Conv2D(1, 16, kernel_size=3),    # (m, 1, 28, 28) -> (m, 16, 26, 26)
    ReLU(),
    MaxPool2D(pool_size=2),          # -> (m, 16, 13, 13)
    Flatten(),                        # -> (2704, m)
    Dense(2704, 64),
    ReLU(),
    Dense(64, 10),
    Softmax()
])
```

**Beklenen Sonuc:** ~80-85% test accuracy (kucuk dataset ile)

---

# IMPLEMENTATION CHECKLIST

## Bolum 1: ANN Temelleri
- [ ] **1. base.py** - Layer soyut sinifi
- [ ] **2. dense.py** - Dense katman (forward + backward)
- [ ] **3. activations.py** - ReLU ve Softmax
- [ ] **4. losses.py** - CrossEntropyLoss
- [ ] **5. optimizers.py** - SGD
- [ ] **6. sequential.py** - Model container
- [ ] **TEST: test_fashion_mnist.py** - ANN calistir

## Bolum 2: CNN Eklentileri
- [ ] **7. conv.py** - im2col, col2im helpers
- [ ] **8. conv.py** - Conv2D katmani
- [ ] **9. conv.py** - MaxPool2D katmani
- [ ] **10. conv.py** - Flatten katmani
- [ ] **TEST: test_cnn_fashion_mnist.py** - CNN calistir

---

# APPENDIX: Boyut Kontrol Rehberi

## ANN Boyutlari
```
Input: X (784, m)
       |
Dense(784, 128): W(128, 784) @ X(784, m) + b(128, 1) = Z(128, m)
       |
ReLU: max(0, Z) = A(128, m)
       |
Dense(128, 10): W(10, 128) @ A(128, m) + b(10, 1) = Z(10, m)
       |
Softmax: softmax(Z) = Y_pred(10, m)
```

## CNN Boyutlari
```
Input: X (m, 1, 28, 28)
       |
Conv2D(1, 16, k=3): (m, 16, 26, 26)
       |
ReLU: (m, 16, 26, 26)
       |
MaxPool2D(2): (m, 16, 13, 13)
       |
Flatten: (2704, m)
       |
Dense(2704, 64): (64, m)
       |
ReLU: (64, m)
       |
Dense(64, 10): (10, m)
       |
Softmax: (10, m)
```

---

# DEBUGGING TIPS

1. **Shape mismatch:** Her adimda `print(x.shape)` ekle
2. **NaN/Inf:** Learning rate cok yuksek olabilir, dusur
3. **Loss artiyorsa:** Learning rate cok yuksek veya gradyan yonunde hata
4. **Loss sabit:** Learning rate cok dusuk veya gradyanlar sifir

---

*Basarilar!*
