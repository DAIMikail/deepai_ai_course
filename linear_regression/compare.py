# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from batch import batch_gradient_descent
from stochastic import stochastic_gradient_descent
from mini import minibatch_gradient_descent
from normal_equation import normal_equation

# =============================================================================
# VERİ SETİ: Çalışma Saati vs Sınav Notu
# =============================================================================
X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([35, 45, 50, 60, 65, 70, 80, 85], dtype=float)

# Normalizasyon
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# =============================================================================
# TÜM ALGORİTMALARI ÇALIŞTIR
# =============================================================================
np.random.seed(42)

# 1. Batch GD
w_batch, b_batch, hist_batch = batch_gradient_descent(X_norm, y_norm)
w_original = w_batch * (y_std / X_std)
b_original = y_mean - w_original * X_mean

# 2. Stochastic GD
w_sgd, b_sgd, hist_sgd = stochastic_gradient_descent(X_norm, y_norm, learning_rate=0.05)
w_original_sgd = w_sgd * (y_std / X_std)
b_original_sgd = y_mean - w_original_sgd * X_mean

# 3. Mini-Batch GD
w_mini, b_mini, hist_mini = minibatch_gradient_descent(X_norm, y_norm, batch_size=2)
w_original_mini = w_mini * (y_std / X_std)
b_original_mini = y_mean - w_original_mini * X_mean

# 4. Normal Equation
w_normal, b_normal = normal_equation(X, y)

# =============================================================================
# SONUÇLARIN KARŞILAŞTIRILMASI
# =============================================================================
print("\n" + "=" * 50)
print("SONUÇLARIN KARŞILAŞTIRILMASI")
print("=" * 50)
print(f"{'Yöntem':<25} {'w (eğim)':<12} {'b (bias)':<12}")
print("-" * 50)
print(f"{'Batch GD':<25} {w_original:<12.2f} {b_original:<12.2f}")
print(f"{'Stochastic GD':<25} {w_original_sgd:<12.2f} {b_original_sgd:<12.2f}")
print(f"{'Mini-Batch GD':<25} {w_original_mini:<12.2f} {b_original_mini:<12.2f}")
print(f"{'Normal Equation':<25} {w_normal:<12.2f} {b_normal:<12.2f}")

# =============================================================================
# VİZUALİZASYON
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: Regresyon çizgisi
ax1 = axes[0]
ax1.scatter(X, y, color='blue', s=100, label='Gerçek Veri', zorder=5)
x_line = np.linspace(0, 9, 100)
ax1.plot(x_line, w_normal * x_line + b_normal, 'r-', linewidth=2,
         label=f'Regresyon: y = {w_normal:.1f}x + {b_normal:.1f}')
ax1.set_xlabel('Çalışma Saati', fontsize=12)
ax1.set_ylabel('Sınav Notu', fontsize=12)
ax1.set_title('Lineer Regresyon: Çalışma Saati vs Sınav Notu', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Sağ: Loss convergence karşılaştırması
ax2 = axes[1]
ax2.plot(hist_batch, label='Batch GD', linewidth=2)
ax2.plot(hist_sgd, label='SGD', linewidth=2, alpha=0.7)
ax2.plot(hist_mini, label='Mini-Batch GD', linewidth=2, alpha=0.7)
ax2.axhline(y=0, color='green', linestyle='--', label='Normal Eq. (anlık)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (MSE)', fontsize=12)
ax2.set_title('Convergence Karşılaştırması', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=-0.1)

plt.tight_layout()
plt.savefig('linear_regression_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
