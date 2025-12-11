# Fashion MNIST - NumPy ile Sıfırdan Sinir Ağı

Bu dokümantasyon, `ann.py` dosyasındaki yapay sinir ağı implementasyonunu detaylı olarak açıklamaktadır.

---

## 1. Genel Bakış

Bu proje, Fashion MNIST veri seti üzerinde **sıfırdan** (NumPy kullanarak) bir yapay sinir ağı oluşturmaktadır. Herhangi bir derin öğrenme kütüphanesi (TensorFlow, PyTorch vb.) kullanılmamıştır.

### Veri Seti: Fashion MNIST
- 28x28 piksellik gri tonlamalı giysi görüntüleri
- 60,000 eğitim + 10,000 test örneği
- 10 sınıf:

| Sınıf | Etiket |
|-------|--------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 2. Ağ Mimarisi

```
Input Layer (784) → Hidden Layer (128, ReLU) → Output Layer (10, Softmax)
```

### Katman Detayları

| Katman | Nöron Sayısı | Aktivasyon | Açıklama |
|--------|--------------|------------|----------|
| Input | 784 | - | 28×28 = 784 piksel (flatten edilmiş görüntü) |
| Hidden | 128 | ReLU | Gizli katman |
| Output | 10 | Softmax | 10 sınıf için olasılık dağılımı |

### Matris Boyutları

```
X:  (784, m)    → m örnek, her biri 784 piksel
W1: (128, 784)  → 128 hidden nöron, her biri 784 input'a bağlı
b1: (128, 1)    → Hidden layer bias
z1: (128, m)    → Pre-activation (linear çıktı)
a1: (128, m)    → ReLU sonrası aktivasyon

W2: (10, 128)   → 10 output nöron, her biri 128 hidden'a bağlı
b2: (10, 1)     → Output layer bias
z2: (10, m)     → Raw skorlar (logits)
a2: (10, m)     → Softmax sonrası (olasılıklar)
```

### Toplam Parametre Sayısı
```
W1: 128 × 784 = 100,352
b1: 128 × 1   = 128
W2: 10 × 128  = 1,280
b2: 10 × 1    = 10
─────────────────────
Toplam: 101,770 parametre
```

---

## 3. Veri Hazırlama

### 3.1 Veri Yükleme
```python
train_data = pd.read_csv(TRAIN_CSV)  # (60000, 785) - ilk sütun etiket
test_data = pd.read_csv(TEST_CSV)    # (10000, 785)

y_train = train_data.iloc[:,0].values   # Etiketler
X_train = train_data.iloc[:,1:].values  # Piksel değerleri
```

### 3.2 Normalizasyon
Piksel değerleri 0-255 aralığından 0-1 aralığına ölçeklenir:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

**Neden Normalizasyon?**
- Gradyan hesaplamalarını stabilize eder
- Eğitimi hızlandırır
- Farklı ölçeklerdeki özellikleri dengeye getirir

### 3.3 Transpose İşlemi
```python
X_train = X_train.T  # (60000, 784) → (784, 60000)
```

**Neden Transpose?**
Matris çarpımı formülü `z = W · X + b` şeklindedir:
```
W: (n_h, n_x)  →  (128, 784)
X: (n_x, m)    →  (784, 60000)  ← Her sütun bir örnek
z: (n_h, m)    →  (128, 60000)
```

---

## 4. One-Hot Encoding

Etiketleri kategorik forma dönüştürür.

### Formül
Eğer `y[i] = k` ise, one-hot vektörün k. elemanı 1, diğerleri 0'dır.

### Örnek
```
y = [2, 0, 5]

One-hot matrix (10, 3):
         örnek0  örnek1  örnek2
sınıf 0 [   0       1       0   ]  ← T-shirt
sınıf 1 [   0       0       0   ]
sınıf 2 [   1       0       0   ]  ← Pullover
sınıf 3 [   0       0       0   ]
sınıf 4 [   0       0       0   ]
sınıf 5 [   0       0       1   ]  ← Sandal
...
```

### Implementasyon
```python
def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot
```

**Trick:** `one_hot[y, np.arange(m)] = 1` satırı, her sütunda doğru satıra 1 yerleştirir.

---

## 5. Parametre Başlatma (He Initialization)

### Formül
```
W = randn(n_out, n_in) × √(2 / n_in)
b = zeros(n_out, 1)
```

### Neden He Initialization?
- ReLU aktivasyonu için optimize edilmiştir
- Vanishing/exploding gradient problemini önler
- Xavier initialization: `√(1/n_in)` - sigmoid/tanh için
- He initialization: `√(2/n_in)` - ReLU için (çünkü ReLU girdilerin yarısını sıfırlar)

### Implementasyon
```python
def initialize_parameters(n_x, n_h, n_y):
    # Katman 1: Input → Hidden
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h, 1))

    # Katman 2: Hidden → Output
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2.0 / n_h)
    b2 = np.zeros((n_y, 1))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
```

---

## 6. Aktivasyon Fonksiyonları

### 6.1 ReLU (Rectified Linear Unit)

#### Formül
```
g(z) = max(0, z)
```

#### Grafik
```
        ▲ g(z)
        │      ╱
        │     ╱
        │    ╱
        │   ╱
────────┼──╱──────► z
        │╱
        │
```

#### Türev (Backpropagation için)
```
g'(z) = 1  if z > 0
      = 0  if z ≤ 0
```

#### Implementasyon
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

#### Avantajları
- Hesaplama açısından verimli
- Vanishing gradient problemini azaltır
- Sparse activation (seyrek aktivasyon) sağlar

---

### 6.2 Softmax

#### Formül
```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

#### Özellikler
- Tüm çıktılar 0-1 arasında
- Tüm çıktıların toplamı = 1 (olasılık dağılımı)
- En yüksek değeri orantılı olarak büyütür

#### Numerical Stability
Büyük değerlerde overflow önlemek için max değeri çıkarılır:
```python
z_shifted = z - max(z)  # Overflow önleme
```

Bu matematiksel olarak sonucu değiştirmez çünkü:
```
exp(z - c) / Σexp(z - c) = exp(z)×exp(-c) / Σexp(z)×exp(-c) = exp(z) / Σexp(z)
```

#### Implementasyon
```python
def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

---

## 7. Forward Propagation (İleri Yayılım)

### Formüller

**Katman 1 (Input → Hidden):**
```
z₁ = W₁ · X + b₁
a₁ = ReLU(z₁)
```

**Katman 2 (Hidden → Output):**
```
z₂ = W₂ · a₁ + b₂
a₂ = Softmax(z₂) = ŷ
```

### Akış Diyagramı
```
X (784, m)
    │
    ▼
┌─────────────────────────────────┐
│  z₁ = W₁ · X + b₁               │  Linear transformation
│  W₁: (128, 784), b₁: (128, 1)   │
└─────────────────────────────────┘
    │
    ▼ z₁ (128, m)
    │
┌─────────────────────────────────┐
│  a₁ = ReLU(z₁) = max(0, z₁)     │  Non-linear activation
└─────────────────────────────────┘
    │
    ▼ a₁ (128, m)
    │
┌─────────────────────────────────┐
│  z₂ = W₂ · a₁ + b₂              │  Linear transformation
│  W₂: (10, 128), b₂: (10, 1)     │
└─────────────────────────────────┘
    │
    ▼ z₂ (10, m)
    │
┌─────────────────────────────────┐
│  a₂ = Softmax(z₂)               │  Probability distribution
└─────────────────────────────────┘
    │
    ▼ a₂ (10, m) = ŷ (tahminler)
```

### Implementasyon
```python
def forward_propagation(X, parameters):
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)

    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache
```

---

## 8. Loss Fonksiyonu (Categorical Cross-Entropy)

### Formül
```
L = -(1/m) × Σⱼ Σᵢ Yᵢⱼ × log(ŷᵢⱼ)
```

**Matris formunda:**
```
L = -(1/m) × sum(Y ⊙ log(a2))
```

Burada:
- `Y`: Gerçek etiketler (one-hot encoded)
- `ŷ` (a2): Tahmin edilen olasılıklar
- `⊙`: Element-wise çarpım
- `m`: Örnek sayısı

### Sezgisel Açıklama
- Doğru sınıfın olasılığının log'unu alır
- Yüksek olasılık (1'e yakın) → Düşük loss
- Düşük olasılık (0'a yakın) → Yüksek loss

| Tahmin (ŷ) | -log(ŷ) | Yorum |
|------------|---------|-------|
| 1.0 | 0.0000 | Mükemmel tahmin |
| 0.9 | 0.1054 | İyi tahmin |
| 0.5 | 0.6931 | Orta tahmin |
| 0.1 | 2.3026 | Kötü tahmin |
| 0.01 | 4.6052 | Çok kötü tahmin |

### Implementasyon
```python
def compute_loss(a2, Y):
    m = Y.shape[1]
    epsilon = 1e-8  # log(0) önlemek için
    log_probs = np.log(a2 + epsilon)
    loss = -np.sum(Y * log_probs) / m
    return loss
```

---

## 9. Backward Propagation (Geri Yayılım)

### Temel Fikir
Chain rule kullanarak loss'un her parametreye göre kısmi türevini hesaplar.

### Formüller

**Katman 2 (Output Layer):**
```
dz₂ = a₂ - Y                    # Softmax + Cross-Entropy türevi
dW₂ = (1/m) × dz₂ · a₁ᵀ
db₂ = (1/m) × Σ dz₂             # Sütunlar boyunca toplam
```

**Katman 1 (Hidden Layer):**
```
da₁ = W₂ᵀ · dz₂                 # Gradyanı geriye taşı
dz₁ = da₁ ⊙ ReLU'(z₁)           # ReLU türevini uygula
dW₁ = (1/m) × dz₁ · Xᵀ
db₁ = (1/m) × Σ dz₁
```

### Akış Diyagramı (Geriye Doğru)
```
Loss (L)
    │
    ▼
┌─────────────────────────────────────┐
│  dz₂ = a₂ - Y                       │  Softmax + CE combined
│  (10, m)                            │
└─────────────────────────────────────┘
    │
    ├──► dW₂ = (1/m) × dz₂ · a₁ᵀ     (10, 128)
    ├──► db₂ = (1/m) × Σ dz₂          (10, 1)
    │
    ▼
┌─────────────────────────────────────┐
│  da₁ = W₂ᵀ · dz₂                    │  Gradient flow back
│  (128, m)                           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  dz₁ = da₁ ⊙ ReLU'(z₁)              │  Element-wise
│  (128, m)                           │
└─────────────────────────────────────┘
    │
    ├──► dW₁ = (1/m) × dz₁ · Xᵀ      (128, 784)
    └──► db₁ = (1/m) × Σ dz₁          (128, 1)
```

### Neden `dz₂ = a₂ - Y` ?
Softmax + Cross-Entropy loss kombinasyonunun türevi basitleşir:

```
∂L/∂z₂ = softmax(z₂) - Y = a₂ - Y
```

Bu, matematiksel olarak türetilmiş bir sonuçtur ve hesaplamayı çok basitleştirir.

### Implementasyon
```python
def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]

    z1, a1, a2 = cache['z1'], cache['a1'], cache['a2']
    W2 = parameters['W2']

    # Katman 2
    dz2 = a2 - Y
    dW2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    # Katman 1
    da1 = np.dot(W2.T, dz2)
    dz1 = da1 * relu_derivative(z1)
    dW1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
```

---

## 10. Gradient Descent (Parametre Güncelleme)

### Formül
```
W := W - α × dW
b := b - α × db
```

Burada:
- `α` (alpha): Learning rate (öğrenme oranı)
- `dW`, `db`: Hesaplanan gradyanlar

### Sezgisel Açıklama
- Gradyan, loss'un en hızlı arttığı yönü gösterir
- Negatif gradyan yönünde hareket ederek loss'u azaltırız
- Learning rate, adım büyüklüğünü kontrol eder

### Implementasyon
```python
def update_parameters(parameters, gradients, learning_rate):
    parameters['W1'] -= learning_rate * gradients['dW1']
    parameters['b1'] -= learning_rate * gradients['db1']
    parameters['W2'] -= learning_rate * gradients['dW2']
    parameters['b2'] -= learning_rate * gradients['db2']
    return parameters
```

---

## 11. Accuracy Hesaplama

### Adımlar
1. Forward pass ile tahmin yap
2. `argmax` ile en olası sınıfı bul
3. Gerçek etiketlerle karşılaştır
4. Doğru tahmin yüzdesini hesapla

### Implementasyon
```python
def compute_accuracy(X, Y, parameters):
    a2, _ = forward_propagation(X, parameters)
    predictions = np.argmax(a2, axis=0)    # Tahmin edilen sınıflar
    labels = np.argmax(Y, axis=0)          # Gerçek sınıflar
    correct = np.sum(predictions == labels)
    accuracy = (correct / Y.shape[1]) * 100
    return accuracy
```

---

## 12. Eğitim Döngüsü

### Genel Akış
```
Her epoch için:
    1. Forward Pass    → Tahmin yap
    2. Loss Hesapla    → Hata ölç
    3. Backpropagation → Gradyanları hesapla
    4. Update          → Parametreleri güncelle
    5. Log             → İlerlemeyi kaydet
```

### Hiperparametreler
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Hidden neurons | 128 | Gizli katman nöron sayısı |
| Learning rate | 0.1 | Öğrenme oranı |
| Epochs | 200 | Toplam eğitim döngüsü |

### Eğitim Süreci Özeti
```
Epoch →  Loss ↓   →  Train Acc ↑  →  Test Acc ↑
  1       2.xx          ~15%           ~15%
  50      0.xx          ~80%           ~80%
  100     0.xx          ~85%           ~83%
  200     0.xx          ~88%           ~85%
```

---

## 13. Görselleştirme

### 13.1 Training History
İki grafik çizilir:
1. **Loss vs Epoch**: Cross-entropy loss'un düşüşü
2. **Accuracy vs Epoch**: Train ve test accuracy'nin artışı

### 13.2 Tahmin Görselleştirme
- Rastgele test örnekleri seçilir
- Model tahmini yapılır
- **Yeşil başlık**: Doğru tahmin
- **Kırmızı başlık**: Yanlış tahmin
- Güven yüzdesi gösterilir

---

## 14. Özet: Tam Eğitim Akışı

```
┌─────────────────────────────────────────────────────────────┐
│                     VERİ HAZIRLAMA                          │
├─────────────────────────────────────────────────────────────┤
│  1. CSV'den veri yükle                                      │
│  2. Normalizasyon: X / 255                                  │
│  3. Transpose: (m, 784) → (784, m)                          │
│  4. One-hot encode: y → Y                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PARAMETRE BAŞLATMA                         │
├─────────────────────────────────────────────────────────────┤
│  He Initialization: W = randn × √(2/n_in)                   │
│  Bias: b = zeros                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EĞİTİM DÖNGÜSÜ                           │
├─────────────────────────────────────────────────────────────┤
│  for epoch in range(num_epochs):                            │
│                                                             │
│    ┌──────────────────────────────────┐                     │
│    │ A. FORWARD PROPAGATION           │                     │
│    │    z₁ = W₁·X + b₁                │                     │
│    │    a₁ = ReLU(z₁)                 │                     │
│    │    z₂ = W₂·a₁ + b₂               │                     │
│    │    a₂ = Softmax(z₂)              │                     │
│    └──────────────────────────────────┘                     │
│                      │                                      │
│                      ▼                                      │
│    ┌──────────────────────────────────┐                     │
│    │ B. LOSS HESAPLA                  │                     │
│    │    L = -mean(Y × log(a₂))        │                     │
│    └──────────────────────────────────┘                     │
│                      │                                      │
│                      ▼                                      │
│    ┌──────────────────────────────────┐                     │
│    │ C. BACKPROPAGATION               │                     │
│    │    dz₂ = a₂ - Y                  │                     │
│    │    dW₂, db₂ hesapla              │                     │
│    │    da₁ = W₂ᵀ · dz₂               │                     │
│    │    dz₁ = da₁ ⊙ ReLU'(z₁)         │                     │
│    │    dW₁, db₁ hesapla              │                     │
│    └──────────────────────────────────┘                     │
│                      │                                      │
│                      ▼                                      │
│    ┌──────────────────────────────────┐                     │
│    │ D. GRADIENT DESCENT              │                     │
│    │    W := W - α × dW               │                     │
│    │    b := b - α × db               │                     │
│    └──────────────────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DEĞERLENDİRME                            │
├─────────────────────────────────────────────────────────────┤
│  - Loss grafiği çiz                                         │
│  - Accuracy grafiği çiz                                     │
│  - Örnek tahminleri görselleştir                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 15. Matematiksel Formüller Özeti

| İşlem | Formül |
|-------|--------|
| Linear Transform | z = W · X + b |
| ReLU | a = max(0, z) |
| ReLU Türevi | g'(z) = 1 if z > 0, else 0 |
| Softmax | σ(zᵢ) = exp(zᵢ) / Σexp(zⱼ) |
| Cross-Entropy | L = -(1/m) × Σ Y × log(ŷ) |
| Gradient Descent | θ := θ - α × ∂L/∂θ |
| He Init | W ~ N(0, √(2/n_in)) |

---

## 16. Kod Dosya Yapısı

```
artificial_neural_networks/
├── ann.py                  # Ana sinir ağı implementasyonu
├── ann.md                  # Bu dokümantasyon
├── fashion-mnist_train.csv # Eğitim verisi (60,000 örnek)
└── fashion-mnist_test.csv  # Test verisi (10,000 örnek)
```
