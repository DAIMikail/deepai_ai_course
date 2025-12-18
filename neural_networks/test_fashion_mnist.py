"""
Fashion MNIST Test
==================
OOP framework'unu ann.py ile karsilastir.

Ayni veri, ayni hiperparametreler, benzer sonuclar olmali.
"""

import os
import sys
import numpy as np
import pandas as pd

# Framework importlari
from layers import Dense, ReLU, Softmax
from models import Sequential
from losses import CrossEntropyLoss
from optimizers import SGD

# Tekrarlanabilirlik icin seed
np.random.seed(42)

# ============== VERI YUKLEME ==============
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(DATA_DIR, "fashion-mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "fashion-mnist_test.csv")

print("Veri yukleniyor...")
train_data = pd.read_csv(TRAIN_CSV)
test_data = pd.read_csv(TEST_CSV)

# Etiketler ve pikseller
y_train = train_data.iloc[:, 0].values
X_train = train_data.iloc[:, 1:].values

y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values

# Normalizasyon (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transpose (m, 784) -> (784, m)
X_train = X_train.T
X_test = X_test.T

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")


# ============== ONE-HOT ENCODING ==============
def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot

Y_train = one_hot_encode(y_train)
Y_test = one_hot_encode(y_test)


# ============== ACCURACY HESAPLAMA ==============
def compute_accuracy(model, X, Y):
    y_pred = model.forward(X)
    predictions = np.argmax(y_pred, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels) * 100
    return accuracy


# ============== MODEL OLUSTURMA ==============
print("\n" + "="*60)
print("MODEL OLUSTURULUYOR")
print("="*60)

model = Sequential([
    Dense(784, 128, seed=42),
    ReLU(),
    Dense(128, 10, seed=42),
    Softmax()
])

print(model)

# Loss ve Optimizer (PyTorch tarzi - ayri)
loss_fn = CrossEntropyLoss()
optimizer = SGD(model, lr=0.1)

print(f"\nLoss:      {loss_fn}")
print(f"Optimizer: {optimizer}")


# ============== EGITIM ==============
print("\n" + "="*60)
print("EGITIM BASLIYOR")
print("="*60)

num_epochs = 200
print_interval = 10

print(f"\nHiperparametreler:")
print(f"  Epochs:        {num_epochs}")
print(f"  Learning rate: {optimizer.lr}")
print(f"  Train ornekleri: {X_train.shape[1]}")
print(f"  Test ornekleri:  {X_test.shape[1]}")

print(f"\n{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 50)

for epoch in range(num_epochs):
    # Forward
    y_pred = model.forward(X_train)

    # Loss
    loss = loss_fn.forward(y_pred, Y_train)

    # Backward
    dout = loss_fn.backward()
    model.backward(dout)

    # Update
    optimizer.step()

    # Log
    if (epoch + 1) % print_interval == 0 or epoch == 0:
        train_acc = compute_accuracy(model, X_train, Y_train)
        test_acc = compute_accuracy(model, X_test, Y_test)
        print(f"{epoch+1:>6} | {loss:>10.4f} | {train_acc:>9.2f}% | {test_acc:>9.2f}%")

print("-" * 50)
print("Egitim tamamlandi!")

# Final sonuclar
final_train_acc = compute_accuracy(model, X_train, Y_train)
final_test_acc = compute_accuracy(model, X_test, Y_test)

print(f"\n{'='*60}")
print(f"SONUCLAR")
print(f"{'='*60}")
print(f"Final Train Accuracy: {final_train_acc:.2f}%")
print(f"Final Test Accuracy:  {final_test_acc:.2f}%")
