# Stochastic Gradient Descent (SGD) ile Linear Regression

## Genel Bakış

Bu dosyada, **Stochastic Gradient Descent** yöntemiyle linear regression modeli eğitiyoruz. Batch GD'den temel farkı: her veri noktası için **ayrı ayrı** parametre güncellemesi yapılması.

```
y = wx + b
```

## Algoritma Adımları

### 1. Veri Normalizasyonu

Batch GD ile aynı şekilde veriler normalize edilir:

```
X_norm = (X - mean) / std
```

### 2. Stochastic Gradient Descent Döngüsü

Her epoch'ta:

```
for epoch in range(epochs):
    1. Verileri karıştır (shuffle):  indices = random.permutation(n)

    2. Her veri noktası için TEK TEK:
       for i in indices:
           - Tahmin:    y_pred = w * X[i] + b
           - Gradyan:   dw = 2 * (y_pred - y[i]) * X[i]
                        db = 2 * (y_pred - y[i])
           - Güncelle:  w = w - lr * dw
                        b = b - lr * db

    3. Epoch sonunda loss hesapla
```

### 3. Kritik Fark: Shuffle (Karıştırma)

```python
indices = np.random.permutation(n)
```

Her epoch'ta veriler rastgele sıralanır. Bu sayede:
- Model belirli bir sıraya ezberlemez
- Yakınsama daha iyi olur
- Local minima'dan kaçış şansı artar

## Batch GD vs Stochastic GD Karşılaştırması

| Özellik | Batch GD | Stochastic GD |
|---------|----------|---------------|
| Gradyan Hesabı | Tüm veri | Tek veri noktası |
| Güncelleme/Epoch | 1 | n (veri sayısı) |
| Kararlılık | Yüksek | Düşük (gürültülü) |
| Hız | Yavaş | Hızlı |
| Bellek | Yüksek | Düşük |
| Local Minima | Takılabilir | Kaçabilir |

## Görsel Karşılaştırma

```
Batch GD:      ─────────────→  (düz, kararlı)

Stochastic GD: ∿∿∿∿∿∿∿∿∿∿∿∿→  (zikzak, gürültülü ama hedefe ulaşır)
```

## Matematiksel Temel

**Tek Nokta İçin Gradyan:**
```
dw = 2 * (wx_i + b - y_i) * x_i
db = 2 * (wx_i + b - y_i)
```

Not: Batch GD'deki `(1/n) * Σ` yerine sadece tek nokta kullanılır.

## Hiperparametreler

- **learning_rate = 0.05**: Batch GD'ye göre daha düşük tutulur (gürültü nedeniyle)
- **epochs = 100**: Her epoch'ta n güncelleme yapıldığı için toplam güncelleme sayısı: 100 × 8 = 800

## Ne Zaman SGD Kullanmalı?

- Büyük veri setlerinde (belleğe sığmayan)
- Online learning senaryolarında
- Local minima'dan kaçmak istediğinizde
- Daha hızlı iterasyon istediğinizde

## Sonuç

SGD, Batch GD'ye göre daha gürültülü bir öğrenme süreci sunar ama büyük veri setlerinde çok daha verimlidir. Modern deep learning'de SGD'nin bir varyantı olan **Mini-Batch GD** en yaygın kullanılan yöntemdir.
