import numpy as np

np.random.seed(42)

X = np.array([
    [1.0], [1.5], [2.0], [2.5], [3.0],   # Düşük çalışma saatleri
    [5.0], [5.5], [6.0], [6.5], [7.0]    # Yüksek çalışma saatleri
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0: Başarısız, 1: Başarılı

def sigmoid(z):
    """
    Sigmoid aktivasyon fonksiyonu

    NEDEN SİGMOID?
    - Linear Regression: Çıktı (-∞, +∞) arasında herhangi bir değer
    - Logistic Regression: Çıktı [0, 1] arasında olasılık olmalı
    - Sigmoid bu dönüşümü sağlar

    Grafik:
         1 ┤          ●●●●●●●●
           │        ●●
       0.5 ┤      ●      ← z=0 noktasında 0.5
           │    ●●
         0 ┤●●●●
           └──────────────────
              -∞    0    +∞
    """
    # σ(z) = 1 / (1 + e^(-z))
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss (Log Loss)

    NEDEN MSE DEĞİL BCE?
    - MSE + Sigmoid = Non-convex (birçok yerel minimum)
    - BCE + Sigmoid = Convex (tek global minimum, gradient descent çalışır)

    GRADIENT ASCENT vs GRADIENT DESCENT:
    ┌─────────────────┬────────────────────────┬────────────────────────┐
    │                 │ Gradient Ascent        │ Gradient Descent       │
    ├─────────────────┼────────────────────────┼────────────────────────┤
    │ Amaç            │ Maximize               │ Minimize               │
    │ Fonksiyon       │ Log-Likelihood (LL)    │ BCE = -LL              │
    │ Güncelleme      │ θ = θ + α * ∇θ         │ θ = θ - α * ∇θ         │
    └─────────────────┴────────────────────────┴────────────────────────┘

    Matematiksel olarak aynı sonuç:
    - Ascent:  w = w + α * (y - ŷ)   "olasılığı artır"
    - Descent: w = w - α * (ŷ - y)   "hatayı azalt"
    - Çünkü: -(ŷ - y) = (y - ŷ)
    """
    # log(0) = -∞ hatasını önle
    # 1e-15 = 0.000000000000001 (çok küçük pozitif sayı)
    epsilon = 1e-15
    # Tahminleri [ε, 1-ε] aralığına sıkıştır
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Binary Cross-Entropy formülü:
    # L = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    #
    # y=1 iken: -log(ŷ)     → ŷ=1'e yakınsa loss düşük
    # y=0 iken: -log(1-ŷ)   → ŷ=0'a yakınsa loss düşük
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def predict_proba(X,w,b):
    """
    Olasılık tahmini

    İşlem sırası:
    1. z = X·w + b  (linear kombinasyon)
    2. ŷ = σ(z)     (sigmoid ile 0-1 aralığına dönüştür)
    """
    # Linear kısım (Linear Regression ile aynı)
    z = np.dot(X, w) + b
    # Sigmoid ile olasılığa dönüştür
    return sigmoid(z)

def predict(X, w, b, threshold=0.5):
    """
    Sınıf tahmini (0 veya 1)

    threshold (eşik değeri):
    - ŷ >= 0.5 → 1 (pozitif sınıf)
    - ŷ <  0.5 → 0 (negatif sınıf)

    Örnek:
    probas = [0.2, 0.7, 0.5, 0.9]
    preds  = [0,   1,   1,   1  ]
    """
    probas = predict_proba(X, w, b)
    # True/False → 1/0 dönüşümü
    return (probas >= threshold).astype(int)

def train(X, y, learning_rate=0.1, epochs=1000, print_every=100):
    """
    Logistic Regression eğitimi (Gradient Descent)

    LINEAR REGRESSION İLE KARŞILAŞTIRMA:
    ┌────────────────┬───────────────────┬───────────────────┐
    │                │ Linear Reg.       │ Logistic Reg.     │
    ├────────────────┼───────────────────┼───────────────────┤
    │ Tahmin         │ ŷ = wx + b        │ ŷ = σ(wx + b)     │
    │ Loss           │ MSE               │ BCE               │
    │ Gradyan (dw)   │ (2/n)*Σ(ŷ-y)*x   │ (1/n)*Xᵀ·(ŷ-y)   │
    │ Gradyan (db)   │ (2/n)*Σ(ŷ-y)     │ (1/n)*Σ(ŷ-y)     │
    └────────────────┴───────────────────┴───────────────────┘
    Not: Gradyan formülleri matematiksel olarak benzer çıkar!
    """
    n_samples, n_features = X.shape

    # Parametreleri sıfırla
    w = np.zeros(n_features)
    b = 0.0

    print("\n" + "=" * 50)
    print("EĞİTİM BAŞLADI")
    print("=" * 50)
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print("-" * 50)

    losses = []

    for epoch in range(epochs):
        # 1. İleri yayılım (Forward Pass)
        y_pred = predict_proba(X,w,b)

        # 2. Loss hesapla (BCE)
        loss = compute_loss(y,y_pred)
        losses.append(loss)

        # 3. Gradyanları hesapla
        # error = ŷ - y (tahmin hatası)
        error = y_pred - y
        # ∂L/∂w = (1/n) * Xᵀ · (ŷ - y)
        dw = (1 / n_samples) * np.dot(X.T, error)
        # ∂L/∂b = (1/n) * Σ(ŷ - y)
        db = (1 / n_samples) * np.sum(error)

        # 4. Parametreleri güncelle
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {loss:.4f}")


    print("-" * 50)
    print("EĞİTİM TAMAMLANDI")

    return w, b, losses

w, b, losses = train(X, y, learning_rate=0.5, epochs=500, print_every=100)

print("\n" + "=" * 50)
print("ÖĞRENILEN PARAMETRELER")
print("=" * 50)
print(f"Ağırlık (w): {w[0]:.4f}")
print(f"Bias (b): {b:.4f}")

print("\n" + "=" * 50)
print("YENİ VERİLER İÇİN TAHMİN")
print("=" * 50)

new_hours = np.array([[3.5], [4.0], [4.5]])

for hours in new_hours:
    prob = predict_proba(hours.reshape(1, -1), w, b)[0]
    pred = "Geçer" if prob >= 0.5 else "Kalır"
    print(f"{hours[0]:.1f} saat çalışan öğrenci: {pred} (Olasılık: {prob:.2%})")


# σ(wx + b) = 0.5 olduğunda x = -b/w
decision_boundary = -b / w[0]
print(f"\nKarar Sınırı: {decision_boundary:.2f} saat")
print("Bu değerin üzerinde çalışan öğrenci geçer tahmin edilir.")