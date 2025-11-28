# Batch Gradient Descent ile Linear Regression

## Genel Bakış

Bu dosyada, **Batch Gradient Descent** yöntemiyle basit bir linear regression modeli eğitiyoruz. Amaç: çalışma saati (X) ile sınav notu (y) arasındaki doğrusal ilişkiyi öğrenmek.

```
y = wx + b
```

## Algoritma Adımları

### 1. Veri Normalizasyonu

Gradient descent'in daha kararlı ve hızlı çalışması için veriler normalize edilir:

```
X_norm = (X - mean) / std
```

Bu sayede:
- Gradyanlar daha dengeli olur
- Learning rate seçimi kolaylaşır
- Yakınsama hızlanır

### 2. Batch Gradient Descent Döngüsü

Her epoch'ta şu işlemler **tüm veri seti üzerinde** yapılır:

```
for epoch in range(epochs):
    1. Tahmin yap:        y_pred = w * X + b
    2. Loss hesapla:      MSE = mean((y_pred - y)²)
    3. Gradyan hesapla:   dw = (2/n) * Σ((y_pred - y) * X)
                          db = (2/n) * Σ(y_pred - y)
    4. Parametreleri güncelle:
                          w = w - learning_rate * dw
                          b = b - learning_rate * db
```

### 3. Orijinal Ölçeğe Dönüşüm

Normalize edilmiş parametreler, orijinal veri ölçeğine geri dönüştürülür:

```
w_original = w_norm * (y_std / X_std)
b_original = y_mean - w_original * X_mean
```

## Batch vs Diğer Yöntemler

| Özellik | Batch GD |
|---------|----------|
| Gradyan Hesabı | Tüm veri |
| Güncelleme/Epoch | 1 |
| Kararlılık | Yüksek |
| Bellek Kullanımı | Yüksek |
| Büyük Veri | Yavaş |

## Matematiksel Temel

**Loss Fonksiyonu (MSE):**
```
L = (1/n) * Σ(y_pred - y)²
```

**Gradyanlar (Zincir Kuralı ile):**
```
∂L/∂w = (2/n) * Σ((wx + b - y) * x)
∂L/∂b = (2/n) * Σ(wx + b - y)
```

**Güncelleme Kuralı:**
```
w_new = w_old - α * ∂L/∂w
b_new = b_old - α * ∂L/∂b
```

Burada `α` (alpha) = learning rate

## Sonuç

Kod çalıştırıldığında, model çalışma saati ile sınav notu arasındaki doğrusal ilişkiyi öğrenir. Örneğin 5.5 saat çalışan bir öğrencinin tahmini sınav notu hesaplanabilir.
