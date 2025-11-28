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

def normal_equation(X,y):
    """
    Normal Equation (Analitik Çözüm)

    GRADIENT DESCENT'TEN FARKI:
    - GD: İteratif, adım adım optimum'a yaklaşır
    - Normal Eq: Tek seferde direkt çözüm, iterasyon yok

    Avantajları: Hızlı, learning rate seçimi yok
    Dezavantajları: Büyük veride yavaş (matris tersi O(n³)),
                    XᵀX tekil olabilir (terslenemez)
    """
    # Design matrix: X'e 1'ler sütunu ekle (bias terimi için)
    # [1, x₁]
    # [1, x₂]  →  y = b·1 + w·x  şeklinde yazılabilsin
    # [1, x₃]
    X_b = np.column_stack([np.ones(len(X)), X])

    # Normal Denklem Formülü:
    # θ = (XᵀX)⁻¹ Xᵀy
    #
    # Adımlar:
    # 1. X_b.T @ X_b  →  XᵀX (özellik matrisi)
    # 2. np.linalg.inv(...)  →  (XᵀX)⁻¹ (matris tersi)
    # 3. @ X_b.T @ y  →  sonucu Xᵀy ile çarp
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    # theta[0] = b (bias), theta[1] = w (weight)
    b,w = theta[0], theta[1]

    return w, b

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("4. NORMAL EQUATION (Analitik Çözüm)")
    print("=" * 50)
    w_normal, b_normal = normal_equation(X, y)
    print(f"Orijinal ölçek: y = {w_normal:.2f}x + {b_normal:.2f}")
    print("(İterasyon yok, tek seferde hesaplandı!)")

    print("\n" + "=" * 50)
    print("TAHMİN ÖRNEĞİ")
    print("=" * 50)
    test_hours = 5.5
    predicted_score = w_normal * test_hours + b_normal
    print(f"Bir öğrenci {test_hours} saat çalışırsa, tahmini sınav notu: {predicted_score:.1f}")