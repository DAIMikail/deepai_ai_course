import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# VERİ SETİ: Çalışma Saati vs Sınav Notu
# =============================================================================
X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)  # Çalışma saati
y = np.array([35, 45, 50, 60, 65, 70, 80, 85], dtype=float)  # Sınav notu

# Normalizasyon (gradient descent için önemli)
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

def minibatch_gradient_descent(X,y,batch_size=2,learning_rate=0.1,epochs=100):
    """
    Mini-Batch Gradient Descent

    BATCH VE SGD'NİN ORTASI - İKİSİNİN AVANTAJLARINI BİRLEŞTİRİR:
    ┌─────────────┬─────────────┬─────────────┬─────────────────────┐
    │ Yöntem      │ Örnek Sayısı│ Güncelleme  │ Özellik             │
    ├─────────────┼─────────────┼─────────────┼─────────────────────┤
    │ Batch       │ n (tümü)    │ 1/epoch     │ Kararlı ama yavaş   │
    │ SGD         │ 1           │ n/epoch     │ Hızlı ama gürültülü │
    │ Mini-Batch  │ k (1<k<n)   │ n/k /epoch  │ Dengeli yaklaşım    │
    └─────────────┴─────────────┴─────────────┴─────────────────────┘
    """
    n= len(X)
    w,b = 0.0, 0.0
    history = []

    for epoch in range(epochs):
        # Verileri karıştır
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # VERİYİ KÜÇÜK GRUPLARA (MINI-BATCH) BÖLEREK İŞLE
        for i in range(0,n,batch_size):
            # Mini-batch oluştur
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_n = len(X_batch)

            # Mini-batch üzerinden tahmin
            y_pred = w * X_batch + b

            # Gradyanlar (MINI-BATCH ÜZERİNDEN - n yerine batch_n)
            # ∂L/∂w = (2/k) * Σ(ŷ - y) * x  (k = batch_size)
            # ∂L/∂b = (2/k) * Σ(ŷ - y)
            dw = (2/batch_n) * np.sum((y_pred - y_batch) * X_batch)
            db = (2/batch_n) * np.sum(y_pred - y_batch)

            # Güncelleme (HER MINI-BATCH SONRASI)
            w = w - learning_rate * dw
            b = b - learning_rate * db

        # Loss (epoch sonunda tüm veri üzerinden)
        # MSE = (1/n) * Σ(ŷ - y)²
        loss = np.mean((w* X +b - y) ** 2)
        print(f"Epoch: {epoch} loss: {loss}")
        history.append(loss)

    return w, b, history

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("3. MINI-BATCH GRADIENT DESCENT (batch_size=2)")
    print("=" * 50)
    np.random.seed(42)
    w_mini, b_mini, hist_mini = minibatch_gradient_descent(X_norm, y_norm, batch_size=2)
    print(f"Normalize edilmiş: w = {w_mini:.4f}, b = {b_mini:.4f}")

    w_original_mini = w_mini * (y_std / X_std)
    b_original_mini = y_mean - w_original_mini * X_mean
    print(f"Orijinal ölçek: y = {w_original_mini:.2f}x + {b_original_mini:.2f}")

    print("\n" + "=" * 50)
    print("TAHMİN ÖRNEĞİ")
    print("=" * 50)
    test_hours = 5.5
    predicted_score = w_original_mini * test_hours + b_original_mini
    print(f"Bir öğrenci {test_hours} saat çalışırsa, tahmini sınav notu: {predicted_score:.1f}")