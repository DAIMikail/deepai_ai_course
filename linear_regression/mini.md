# Mini-Batch Gradient Descent ile Linear Regression

## Genel Bakış

**Mini-Batch GD**, Batch ve Stochastic yöntemlerinin **ortası**dır. Ne tüm veriyi ne de tek noktayı kullanır; belirli sayıda veri noktasından oluşan **küçük gruplar (batch)** üzerinden güncelleme yapar.

```
y = wx + b
```

## Algoritma Adımları

### 1. Veri Karıştırma ve Gruplama

```python
indices = np.random.permutation(n)
X_shuffled = X[indices]
y_shuffled = y[indices]
```

### 2. Mini-Batch Döngüsü

```
for epoch in range(epochs):
    1. Verileri karıştır

    2. Her mini-batch için:
       for i in range(0, n, batch_size):
           - Batch al:   X_batch = X[i : i+batch_size]
           - Tahmin:     y_pred = w * X_batch + b
           - Gradyan:    dw = (2/batch_n) * Σ((y_pred - y_batch) * X_batch)
                         db = (2/batch_n) * Σ(y_pred - y_batch)
           - Güncelle:   w = w - lr * dw
                         b = b - lr * db

    3. Epoch sonunda loss hesapla
```

## Somut Örnek (batch_size=2)

Veri: `X = [1, 2, 3, 4, 5, 6, 7, 8]` (8 nokta)

```
Epoch 1:
  Karıştır → [3, 7, 1, 5, 8, 2, 6, 4]

  Batch 1: [3, 7]     → gradyan hesapla → güncelle
  Batch 2: [1, 5]     → gradyan hesapla → güncelle
  Batch 3: [8, 2]     → gradyan hesapla → güncelle
  Batch 4: [6, 4]     → gradyan hesapla → güncelle

  Toplam: 4 güncelleme / epoch
```

## Üç Yöntemin Karşılaştırması

| Özellik | Batch | Mini-Batch | Stochastic |
|---------|-------|------------|------------|
| Batch Size | n (tüm veri) | k (örn: 2, 32, 64) | 1 |
| Güncelleme/Epoch | 1 | n/k | n |
| Kararlılık | En yüksek | Orta | En düşük |
| Hız | Yavaş | Dengeli | Hızlı |
| GPU Kullanımı | Verimsiz | Verimli | Verimsiz |

## Neden Mini-Batch?

```
Batch:       ●●●●●●●● → 1 güncelleme (çok kararlı, yavaş)
Mini-Batch:  ●●|●●|●●|●● → 4 güncelleme (dengeli)
Stochastic:  ●|●|●|●|●|●|●|● → 8 güncelleme (gürültülü, hızlı)
```

**Avantajları:**
- GPU/vektör işlemlerinden faydalanır (paralel hesaplama)
- SGD'den daha kararlı gradyanlar
- Batch'ten daha hızlı yakınsama
- Bellek kullanımı kontrol altında

## Batch Size Seçimi

| Batch Size | Özellik |
|------------|---------|
| Küçük (2-8) | SGD'ye yakın, gürültülü |
| Orta (32-128) | En yaygın, dengeli |
| Büyük (256-512) | Batch'e yakın, kararlı |

Modern deep learning'de genellikle **32, 64, 128** gibi değerler kullanılır.

## Kod Örneğindeki Değerler

```python
batch_size = 2      # Her grupta 2 veri
n = 8               # Toplam veri sayısı
güncelleme/epoch = 8/2 = 4
```

## Sonuç

Mini-Batch GD, pratikte **en çok kullanılan** gradient descent yöntemidir. Hem hesaplama verimliliği hem de öğrenme kalitesi açısından optimal dengeyi sağlar.
