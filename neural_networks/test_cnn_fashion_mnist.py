"""
CNN Fashion MNIST Test
======================
Conv2D, MaxPool2D, Flatten katmanlarini test et.

NOT: Pure NumPy ile CNN yavas calisir. Gercek uygulamalarda
im2col optimizasyonu veya GPU kullanilir.
"""

import os
import numpy as np
import pandas as pd
import time

# Framework importlari
from layers import Dense, ReLU, Softmax, Conv2D, MaxPool2D, Flatten
from models import Sequential
from losses import CrossEntropyLoss
from optimizers import SGD

# Tekrarlanabilirlik
np.random.seed(42)

# ============== VERI YUKLEME ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "artificial_neural_networks")
TRAIN_CSV = os.path.join(DATA_DIR, "fashion-mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "fashion-mnist_test.csv")

print("Veri yukleniyor...")
train_data = pd.read_csv(TRAIN_CSV)
test_data = pd.read_csv(TEST_CSV)

y_train = train_data.iloc[:, 0].values
X_train = train_data.iloc[:, 1:].values

y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values

# Normalizasyon
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN icin 4D format: (m, C, H, W)
# (60000, 784) -> (60000, 1, 28, 28)
X_train_cnn = X_train.reshape(-1, 1, 28, 28)
X_test_cnn = X_test.reshape(-1, 1, 28, 28)

print(f"X_train_cnn: {X_train_cnn.shape}")
print(f"X_test_cnn:  {X_test_cnn.shape}")

# CNN yavas oldugu icin kucuk subset kullanalim
TRAIN_SIZE = 10000
TEST_SIZE = 1000

X_train_small = X_train_cnn[:TRAIN_SIZE]
y_train_small = y_train[:TRAIN_SIZE]
X_test_small = X_test_cnn[:TEST_SIZE]
y_test_small = y_test[:TEST_SIZE]

print(f"\nKucuk dataset (hiz icin):")
print(f"  Train: {X_train_small.shape[0]} ornek")
print(f"  Test:  {X_test_small.shape[0]} ornek")


# ============== ONE-HOT ENCODING ==============
def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot

Y_train_small = one_hot_encode(y_train_small)
Y_test_small = one_hot_encode(y_test_small)


# ============== ACCURACY ==============
def compute_accuracy(model, X, Y):
    y_pred = model.forward(X)
    predictions = np.argmax(y_pred, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels) * 100
    return accuracy


# ============== MODEL ==============
print("\n" + "="*60)
print("CNN MODEL OLUSTURULUYOR")
print("="*60)

# Hiperparametreler
INPUT_CHANNELS = 1      # Grayscale
INPUT_SIZE = 28         # Fashion MNIST 28x28
CONV_FILTERS = 16       # Konvolüsyon filtre sayısı
KERNEL_SIZE = 3         # Filtre boyutu (3x3)
POOL_SIZE = 2           # MaxPool pencere boyutu
HIDDEN_UNITS = 64       # Dense katman nöron sayısı
NUM_CLASSES = 10        # Sınıf sayısı (0-9)

# Flatten boyutunu otomatik hesapla
conv_output_size = INPUT_SIZE - KERNEL_SIZE + 1  # 28 - 3 + 1 = 26
pool_output_size = conv_output_size // POOL_SIZE  # 26 // 2 = 13
FLATTEN_SIZE = CONV_FILTERS * pool_output_size * pool_output_size  # 16 * 13 * 13 = 2704

print(f"Conv çıktı boyutu: {conv_output_size}x{conv_output_size}")
print(f"Pool çıktı boyutu: {pool_output_size}x{pool_output_size}")
print(f"Flatten boyutu: {FLATTEN_SIZE}")

"""
Mimari:
    Input:    (m, 1, 28, 28)
    Conv2D:   (m, 16, 26, 26)  -> 16 filtre, 3x3 kernel
    ReLU
    MaxPool:  (m, 16, 13, 13)  -> 2x2 pooling
    Flatten:  (2704, m)        -> 16 * 13 * 13 = 2704
    Dense:    (64, m)
    ReLU
    Dense:    (10, m)
    Softmax
"""

model = Sequential([
    Conv2D(in_channels=INPUT_CHANNELS, out_channels=CONV_FILTERS, kernel_size=KERNEL_SIZE),
    ReLU(),
    MaxPool2D(pool_size=POOL_SIZE),
    Flatten(),
    Dense(FLATTEN_SIZE, HIDDEN_UNITS),
    ReLU(),
    Dense(HIDDEN_UNITS, NUM_CLASSES),
    Softmax()
])

print(model)

loss_fn = CrossEntropyLoss()
optimizer = SGD(model, lr=0.01)  # CNN icin daha dusuk lr

print(f"\nLoss:      {loss_fn}")
print(f"Optimizer: {optimizer}")


# ============== EGITIM ==============
print("\n" + "="*60)
print("EGITIM BASLIYOR (CNN yavas, sabir...)")
print("="*60)

num_epochs = 200  # CNN yavas oldugu icin az epoch
print_interval = 10

print(f"\nHiperparametreler:")
print(f"  Epochs:        {num_epochs}")
print(f"  Learning rate: {optimizer.lr}")
print(f"  Train ornekleri: {X_train_small.shape[0]}")
print(f"  Test ornekleri:  {X_test_small.shape[0]}")

print(f"\n{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Sure':>10}")
print("-" * 65)

total_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()

    # Forward
    y_pred = model.forward(X_train_small)

    # Loss
    loss = loss_fn.forward(y_pred, Y_train_small)

    # Backward
    dout = loss_fn.backward()
    model.backward(dout)

    # Update
    optimizer.step()

    epoch_time = time.time() - epoch_start

    # Log
    if (epoch + 1) % print_interval == 0:
        train_acc = compute_accuracy(model, X_train_small, Y_train_small)
        test_acc = compute_accuracy(model, X_test_small, Y_test_small)
        print(f"{epoch+1:>6} | {loss:>10.4f} | {train_acc:>9.2f}% | {test_acc:>9.2f}% | {epoch_time:>8.1f}s")

total_time = time.time() - total_start
print("-" * 65)
print(f"Egitim tamamlandi! Toplam sure: {total_time:.1f}s")

# Final sonuclar
final_train_acc = compute_accuracy(model, X_train_small, Y_train_small)
final_test_acc = compute_accuracy(model, X_test_small, Y_test_small)

print(f"\n{'='*60}")
print(f"SONUCLAR")
print(f"{'='*60}")
print(f"Final Train Accuracy: {final_train_acc:.2f}%")
print(f"Final Test Accuracy:  {final_test_acc:.2f}%")
print(f"\nNOT: Daha iyi sonuclar icin daha fazla veri ve epoch kullanin.")
