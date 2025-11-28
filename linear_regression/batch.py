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


def batch_gradient_descent(X, y, learning_rate=0.1, epochs=100):
    n = len(X)
    w, b = 0.0, 0.0
    history = []
    
    for epoch in range(epochs):
        # Tüm veri ile tahmin
        y_pred = w * X + b
        
        # Loss (MSE)
        loss = np.mean((y_pred - y) ** 2)
        history.append(loss)
        
        # Gradyanlar (TÜM VERİ ÜZERİNDEN)
        dw = (2/n) * np.sum((y_pred - y) * X)
        db = (2/n) * np.sum(y_pred - y)
        
        # Güncelleme
        w = w - learning_rate * dw
        b = b - learning_rate * db

        print(f"Epoch: {epoch} loss: {loss}")
    
    return w, b, history

if __name__ == "__main__":
    print("=" * 50)
    print("1. BATCH GRADIENT DESCENT")
    print("=" * 50)
    w_batch, b_batch, hist_batch = batch_gradient_descent(X_norm, y_norm)
    print(f"Normalize edilmiş: w = {w_batch:.4f}, b = {b_batch:.4f}")

    # Orijinal ölçeğe dönüştür
    w_original = w_batch * (y_std / X_std)
    b_original = y_mean - w_original * X_mean
    print(f"Orijinal ölçek: y = {w_original:.2f}x + {b_original:.2f}")

    print("\n" + "=" * 50)
    print("TAHMİN ÖRNEĞİ")
    print("=" * 50)
    test_hours = 5.5
    predicted_score = w_original * test_hours + b_original
    print(f"Bir öğrenci {test_hours} saat çalışırsa, tahmini sınav notu: {predicted_score:.1f}")