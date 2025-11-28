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

def stochastic_gradient_descent(X,y,learning_rate=0.1,epochs=100):
    n = len(X)
    w , b = 0.0, 0.0
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)

        for i in indices:
            y_pred = w * X[i] + b
            dw = 2 * (y_pred - y[i]) * X[i]
            db = 2 * (y_pred - y[i])

            w = w - learning_rate * dw
            b = b - learning_rate * db

        loss = np.mean((w * X + b - y) ** 2)
        print(f"Epoch: {epoch} loss {loss}")
        history.append(loss)

    return w, b, history

if __name__ == "__main__":
    np.random.seed(42)
    w_sgd, b_sgd, hist_sgd = stochastic_gradient_descent(X_norm, y_norm, learning_rate=0.05)
    print(f"Normalize edilmiş: w = {w_sgd:.4f}, b = {b_sgd:.4f}")

    w_original_sgd = w_sgd * (y_std / X_std)
    b_original_sgd = y_mean - w_original_sgd * X_mean
    print(f"Orijinal ölçek: y = {w_original_sgd:.2f}x + {b_original_sgd:.2f}")

    print("\n" + "=" * 50)
    print("TAHMİN ÖRNEĞİ")
    print("=" * 50)
    test_hours = 5.5
    predicted_score = w_original_sgd * test_hours + b_original_sgd
    print(f"Bir öğrenci {test_hours} saat çalışırsa, tahmini sınav notu: {predicted_score:.1f}")