# Logistic Regression Algoritması

## Genel Bakış

Logistic Regression, ikili sınıflandırma problemleri için kullanılan bir makine öğrenmesi algoritmasıdır. Bu örnekte öğrencilerin çalışma saatlerine göre sınavı geçip geçmeyeceğini tahmin ediyoruz.

---

## Algoritma Adımları

### 1. Veri Hazırlama

- **X (Girdi)**: Öğrencilerin çalışma saatleri (1-7 saat arası)
- **y (Çıktı)**: Sınav sonucu (0 = Başarısız, 1 = Başarılı)

---

### 2. Sigmoid Fonksiyonu

```
σ(z) = 1 / (1 + e^(-z))
```

**Ne işe yarar?**
- Herhangi bir sayıyı 0 ile 1 arasına sıkıştırır
- Bu sayede çıktı bir olasılık olarak yorumlanabilir
- Örnek: σ(0) = 0.5, σ(2) ≈ 0.88, σ(-2) ≈ 0.12

---

### 3. Tahmin (Forward Pass)

```
z = X · w + b
ŷ = σ(z)
```

**Ne işe yarar?**
- **w (ağırlık)**: Her özelliğin ne kadar önemli olduğunu belirler
- **b (bias)**: Modelin temel eğilimini ayarlar
- **z**: Lineer kombinasyon (ham skor)
- **ŷ**: Sigmoid'den geçirilmiş olasılık tahmini

---

### 4. Kayıp Fonksiyonu (Binary Cross-Entropy Loss)

```
L = -1/n * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Ne işe yarar?**
- Modelin tahminlerinin gerçek değerlerden ne kadar uzak olduğunu ölçer
- Yanlış tahminleri cezalandırır:
  - Gerçek değer 1 iken düşük olasılık tahmin edersek → yüksek ceza
  - Gerçek değer 0 iken yüksek olasılık tahmin edersek → yüksek ceza
- Amaç: Bu değeri minimize etmek

---

### 5. Gradient Hesabı (Backpropagation)

```
dL/dw = (1/n) * X^T · (ŷ - y)
dL/db = (1/n) * Σ(ŷ - y)
```

**Ne işe yarar?**
- Kayıp fonksiyonunun w ve b'ye göre türevini hesaplar
- Bu türevler, parametrelerin hangi yönde ve ne kadar değişmesi gerektiğini söyler
- **(ŷ - y)**: Tahmin hatası - pozitifse tahmin çok yüksek, negatifse çok düşük

---

### 6. Parametre Güncelleme (Gradient Descent)

```
w = w - α · dL/dw
b = b - α · dL/db
```

**Ne işe yarar?**
- **α (learning rate)**: Adım büyüklüğü - ne kadar hızlı öğreneceğimizi belirler
- Parametreler, gradient'ın tersi yönünde güncellenir
- Her iterasyonda kayıp azalır ve model iyileşir

---

### 7. Karar Sınırı

```
Karar Sınırı: x = -b / w
```

**Ne işe yarar?**
- σ(wx + b) = 0.5 olduğu nokta
- Bu noktanın üstünde → Sınıf 1 (Geçer)
- Bu noktanın altında → Sınıf 0 (Kalır)

---

## Özet Akış

```
Girdi (X) → Lineer Kombinasyon (z = Xw + b) → Sigmoid (ŷ) → Tahmin
                          ↑                                    ↓
                   Parametre Güncelle  ←  Gradient  ←  Kayıp Hesapla
```

Model bu döngüyü binlerce kez tekrarlayarak en iyi w ve b değerlerini öğrenir.
