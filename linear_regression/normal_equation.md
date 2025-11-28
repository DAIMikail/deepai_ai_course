# Normal Equation (Analitik Çözüm) ile Linear Regression

## Genel Bakış

Normal Equation, gradient descent'ten tamamen farklı bir yaklaşımdır. İterasyon yapmaz, **tek bir matematiksel formülle** optimal parametreleri direkt hesaplar.

```
θ = (XᵀX)⁻¹ Xᵀy
```

## Gradient Descent vs Normal Equation

| Özellik | Gradient Descent | Normal Equation |
|---------|------------------|-----------------|
| Yaklaşım | İteratif | Analitik |
| Hiperparametre | learning_rate, epochs | Yok |
| Normalizasyon | Gerekli | Gereksiz |
| Hesaplama | O(kn²) | O(n³) - matris tersi |
| Büyük n | Verimli | Yavaş |
| Büyük özellik | Verimli | Çok yavaş |

## Algoritma Adımları

### 1. Design Matrix Oluşturma

```python
X_b = np.column_stack([np.ones(len(X)), X])
```

Bias terimi için 1'lerden oluşan sütun eklenir:

```
X = [1, 2, 3, 4]  →  X_b = [[1, 1],
                            [1, 2],
                            [1, 3],
                            [1, 4]]
```

### 2. Normal Equation Formülü

```python
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

Adım adım:
```
1. Xᵀ         → X'in transpozu
2. XᵀX        → Matris çarpımı
3. (XᵀX)⁻¹    → Matris tersi
4. (XᵀX)⁻¹Xᵀy → Sonuç: [b, w]
```

### 3. Parametreleri Çıkarma

```python
b, w = theta[0], theta[1]
```

## Matematiksel Türetim

MSE loss fonksiyonunun gradyanını sıfıra eşitleyerek:

```
L = (Xθ - y)ᵀ(Xθ - y)

∂L/∂θ = 2Xᵀ(Xθ - y) = 0

XᵀXθ = Xᵀy

θ = (XᵀX)⁻¹Xᵀy  ✓
```

## Somut Hesaplama Örneği

```
X = [1, 2, 3, 4, 5, 6, 7, 8]
y = [35, 45, 50, 60, 65, 70, 80, 85]

X_b = [[1, 1], [1, 2], [1, 3], ... [1, 8]]

XᵀX = [[8, 36],      (XᵀX)⁻¹ = [[0.607, -0.107],
       [36, 204]]               [-0.107, 0.024]]

Xᵀy = [490, 2590]

θ = (XᵀX)⁻¹ @ Xᵀy = [27.5, 7.14]

Sonuç: y = 7.14x + 27.5
```

## Ne Zaman Normal Equation Kullanmalı?

**Kullan:**
- Küçük/orta veri setleri (n < 10,000)
- Az sayıda özellik (feature < 1000)
- Hızlı prototipleme
- Hiperparametre ayarlamak istemiyorsan

**Kullanma:**
- Çok büyük veri setleri
- Çok sayıda özellik
- XᵀX tekil (singular) matris ise
- Online learning gerekiyorsa

## Avantajlar ve Dezavantajlar

**Avantajlar:**
- Tek seferde sonuç
- Learning rate seçimi yok
- Normalizasyon gereksiz
- Garantili global minimum

**Dezavantajlar:**
- O(n³) karmaşıklık (matris tersi)
- Bellek kullanımı yüksek
- XᵀX terslenemezse çalışmaz
- Sadece linear modeller için

## Sonuç

Normal Equation, küçük veri setlerinde **en hızlı ve en basit** çözümdür. Ancak veri büyüdükçe gradient descent tercih edilir.
