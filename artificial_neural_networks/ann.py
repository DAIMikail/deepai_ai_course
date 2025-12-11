"""
Fashion MNIST - NumPy ile Sıfırdan Sinir Ağı
============================================

Ağ Mimarisi:
- Input Layer:  784 nöron (28x28 piksel)
- Hidden Layer: 128 nöron (ReLU aktivasyon)
- Output Layer: 10 nöron (Softmax aktivasyon)

0: T-shirt/top    5: Sandal
1: Trouser        6: Shirt
2: Pullover       7: Sneaker
3: Dress          8: Bag
4: Coat           9: Ankle boot

Boyutlar:
X:  (784, m)    → m örnek, her biri 784 piksel
W1: (128, 784)  → 128 nöron, her biri 784 input'a bağlı
z1: (128, m)    → 128 hidden aktivasyon (pre-activation)
a1: (128, m)    → ReLU sonrası
W2: (10, 128)   → 10 çıktı, her biri 128 hidden'a bağlı
z2: (10, m)     → 10 raw skor
a2: (10, m)     → Softmax sonrası (olasılıklar)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Tekrarlanabilirlik için seed
np.random.seed(42)

# Script'in bulunduğu dizine göre dosya yolları
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(SCRIPT_DIR, "fashion-mnist_train.csv")
TEST_CSV = os.path.join(SCRIPT_DIR, "fashion-mnist_test.csv")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_data = pd.read_csv(TRAIN_CSV)
#print(f"Train CSV boyutu: {train_data.shape}")

test_data = pd.read_csv(TEST_CSV)
#print(f"Test CSV boyutu: {test_data.shape}")

y_train = train_data.iloc[:,0].values
X_train = train_data.iloc[:,1:].values

y_test = test_data.iloc[:,0].values
X_test = test_data.iloc[:,1:].values

# print(f"\nEğitim seti:")
# print(f"  X_train: {X_train.shape}  → (örnek sayısı, piksel sayısı)")
# print(f"  y_train: {y_train.shape}  → (örnek sayısı,)")

# print(f"\nTest seti:")
# print(f"  X_test: {X_test.shape}")
# print(f"  y_test: {y_test.shape}")

# print(f"\nSınıflar: {class_names}")
# print(f"Piksel değer aralığı: {X_train.min()} - {X_train.max()}")


# # ilk 10 resmi gösterelim.
# fig, axes = plt.subplots(2,5,figsize=(12,5))
# axes = axes.flatten()

# for i in range(10):
#     # flatten to img yapalım.
#     img = X_train[i].reshape(28,28)
#     axes[i].imshow(img, cmap='gray')
#     axes[i].set_title(f"{class_names[y_train[i]]}")
#     axes[i].axis('off')

# plt.tight_layout()
# plt.show()

# 0 - 255 arasını -> 0-1 aralığına sıkıştıralım (normalizasyon)
X_train = X_train / 255.0
X_test = X_test / 255.0

"""
X_train: (60000, 784) → (m, n_x)
          ↑      ↑
       örnek  piksel


Ama matris çarpımı formüllerimiz şöyle:

z = W · X + b

W: (n_h, n_x)  →  (128, 784)
X: (n_x, m)    →  (784, 60000)  ← Transpose gerekli!
z: (n_h, m)    →  (128, 60000)
Transpose ile her sütun bir örneği temsil eder
"""

X_train = X_train.T
X_test = X_test.T


def one_hot_encode(y, num_classes=10):
    """
    Etiketleri one-hot vektörlere çevir
    
    y: (m,) şeklinde etiketler [2, 0, 9, 5, ...]
    return: (num_classes, m) şeklinde one-hot matrix
    
    Örnek:
    y = [2, 0, 5]
    
    One-hot (10, 3):
         örnek0  örnek1  örnek2
    0  [   0       1       0   ]  ← T-shirt
    1  [   0       0       0   ]
    2  [   1       0       0   ]  ← Pullover
    3  [   0       0       0   ]
    4  [   0       0       0   ]
    5  [   0       0       1   ]  ← Sandal
    ...
    """

    m = y.shape[0]
    one_hot = np.zeros((num_classes,m))
    one_hot[y,np.arange(m)] = 1

    return one_hot

Y_train = one_hot_encode(y_train)
Y_test = one_hot_encode(y_test)

#forward
#loss
#backward
#gradient update
#tekrar

# print(f"\nDoğrulama (ilk 3 örnek):")
# for i in range(3):
#     print(f"  y={y_train[i]} ({class_names[y_train[i]]:12s}) → {Y_train[:, i].astype(int)}")

def initialize_parameters(n_x, n_h, n_y):
    """
    He initialization ile parametreleri başlat
    
    Parametreler:
        n_x: input boyutu (784)
        n_h: hidden layer boyutu (128)
        n_y: output boyutu (10)
    
    Formül:
        W = randn(n_out, n_in) × √(2 / n_in)
        b = zeros(n_out, 1)
    
    Returns:
        parameters: W1, b1, W2, b2 içeren dictionary
    """

    # Katman 1: Input → Hidden
    # W1 her satırı bir hidden nöronu temsil eder
    # Her hidden nöron 784 input'a bağlı
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h, 1))

    # Katman 2: Hidden → Output
    # W2 her satırı bir output nöronu temsil eder
    # Her output nöron 128 hidden'a bağlı
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2.0 / n_h)
    b2 = np.zeros((n_y,1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2' : W2,
        'b2' : b2
    }

    return parameters    


n_x = 784
n_h = 128
n_y = 10

parameters = initialize_parameters(n_x,n_h,n_y)

# Toplam parametre sayısı
total_params = (parameters['W1'].size + parameters['b1'].size + 
                parameters['W2'].size + parameters['b2'].size)
print(f"\nToplam parametre: {total_params:,}")


def relu(z):
    """
    ReLU Aktivasyonu
    
    Formül: g(z) = max(0, z)
    
    z: herhangi bir boyutta numpy array
    return: aynı boyutta, negatifler sıfırlanmış array
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    ReLU Türevi (Backpropagation için)
    
    Formül: g'(z) = 1 if z > 0
                  = 0 if z ≤ 0
    
    z: forward pass'teki z değerleri
    return: aynı boyutta türev array'i
    """
    return (z > 0).astype(float)

def softmax(z):
    """
    Softmax Aktivasyonu
    
    Formül: softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
    
    z: (n_classes, m) boyutunda raw skorlar
    return: (n_classes, m) boyutunda olasılıklar
    
    Not: Numerical stability için max değeri çıkarıyoruz
    """
    # Her örnek (sütun) için max değeri çıkar
    z_shifted = z - np.max(z, axis=0, keepdims=True)

    #üstel
    exp_z = np.exp(z_shifted)

    #normalize
    return exp_z / np.sum(exp_z,axis=0,keepdims=True)

# print("ReLU Testi:")
# test_input = np.array([-2, -1, 0, 1, 2])
# print(f"  Input:  {test_input}")
# print(f"  ReLU:   {relu(test_input)}")
# print(f"  Türev:  {relu_derivative(test_input)}")

# print("\nSoftmax Testi:")
# test_scores = np.array([[2.0], [1.0], [0.1]])
# softmax_output = softmax(test_scores)
# print(f"  Raw skorlar: {test_scores.flatten()}")
# print(f"  Softmax:     {softmax_output.flatten().round(3)}")
# print(f"  Toplam:      {softmax_output.sum():.4f} (1.0 olmalı)")

# print("\nNumerical Stability Testi:")
# large_scores = np.array([[1000.0], [1001.0], [1002.0]])
# print(f"  Büyük skorlar: {large_scores.flatten()}")
# print(f"  Softmax:       {softmax(large_scores).flatten().round(3)}")
# print(f"  (Overflow olmadan çalıştı ✓)")

# forward_pass
def forward_propagation(X, parameters):
    """
    İleri yayılım - Veriyi ağdan geçir
    
    Formüller:
        Katman 1:
            z₁ = W₁ · X + b₁
            a₁ = ReLU(z₁)
        
        Katman 2:
            z₂ = W₂ · a₁ + b₂
            a₂ = Softmax(z₂) = ŷ
    
    Parametreler:
        X: input verisi (784, m)
        parameters: W1, b1, W2, b2 dictionary
    
    Returns:
        a2: tahminler (10, m)
        cache: backprop için ara değerler {z1, a1, z2, a2}
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    z1 = np.dot(W1,X) + b1
    a1 = relu(z1)

    z2 = np.dot(W2,a1) + b2
    a2 = softmax(z2)

    cache = {
        'z1': z1,
        'a1': a1,
        'z2': z2,
        'a2': a2
    }

    return a2, cache

# İlk 5 örnek üzerinde test
X_sample = X_train[:, :5]  # (784, 5)
print(f"Test input boyutu: {X_sample.shape}")

# Forward pass
a2_test, cache_test = forward_propagation(X_sample, parameters)

print(f"\nCache boyutları:")
print(f"  z1: {cache_test['z1'].shape}")
print(f"  a1: {cache_test['a1'].shape}")
print(f"  z2: {cache_test['z2'].shape}")
print(f"  a2: {cache_test['a2'].shape}")

print(f"\nÇıktı (a2) boyutu: {a2_test.shape}")

# İlk örneğin tahminlerini göster
print(f"\nİlk örnek için tahminler:")
print(f"  Gerçek etiket: {y_train[0]} ({class_names[y_train[0]]})")
print(f"  Tahmin olasılıkları:")
for i, (name, prob) in enumerate(zip(class_names, a2_test[:, 0])):
    bar = "█" * int(prob * 30)
    print(f"    {i}: {name:12s} {prob:.3f} {bar}")

print(f"\n  Olasılıklar toplamı: {a2_test[:, 0].sum():.4f} (1.0 olmalı)")
print(f"  En yüksek olasılık: {class_names[np.argmax(a2_test[:, 0])]}")

def compute_loss(a2, Y):
    """
    Categorical Cross-Entropy Loss
    
    Formül:
        L = -(1/m) × Σⱼ Σᵢ Yᵢⱼ × log(ŷᵢⱼ)
    
    Matris formunda:
        L = -(1/m) × sum(Y ⊙ log(a2))
    
    Parametreler:
        a2: tahminler (softmax çıktısı) (10, m)
        Y:  gerçek etiketler (one-hot) (10, m)
    
    Returns:
        loss: skaler değer
    """

    m = Y.shape[1] #örnek sayısı

    epsilon = 1e-8

    log_probs = np.log(a2 + epsilon)
    loss = -np.sum(Y * log_probs) / m

    return loss

loss_test = compute_loss(a2_test, Y_train[:, :5])
print(f"Test loss (5 örnek): {loss_test:.4f}")

print(f"\nÖrnek Loss Değerleri:")
print(f"  Mükemmel tahmin  (ŷ=1.0): {-np.log(1.0):.4f}")
print(f"  İyi tahmin       (ŷ=0.9): {-np.log(0.9):.4f}")
print(f"  Orta tahmin      (ŷ=0.5): {-np.log(0.5):.4f}")
print(f"  Kötü tahmin      (ŷ=0.1): {-np.log(0.1):.4f}")
print(f"  Çok kötü tahmin  (ŷ=0.01): {-np.log(0.01):.4f}")

def backward_propagation(X, Y, parameters, cache):
    """
    Geri yayılım - Tüm gradyanları hesapla
    
    Formüller:
        Katman 2:
            dz2 = a2 - Y
            dW2 = (1/m) × dz2 · a1ᵀ
            db2 = (1/m) × Σ dz2
        
        Katman 1:
            da1 = W2ᵀ · dz2
            dz1 = da1 ⊙ ReLU'(z1)
            dW1 = (1/m) × dz1 · Xᵀ
            db1 = (1/m) × Σ dz1
    
    Parametreler:
        X: input verisi (784, m)
        Y: gerçek etiketler one-hot (10, m)
        parameters: W1, b1, W2, b2
        cache: forward pass'ten z1, a1, z2, a2
    
    Returns:
        gradients: dW1, db1, dW2, db2 dictionary
    """

    m = X.shape[1]

    z1 = cache['z1']
    a1 = cache['a1']
    a2 = cache['a2']
    W2 = parameters['W2']

    dz2 = a2 - Y
    dW2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T,dz2)
    dz1 = da1 * relu_derivative(z1)
    dW1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis=1,keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients

def update_parameters(parameters, gradients, learning_rate):
    """
    Parametreleri gradient descent ile güncelle
    
    Formül:
        W := W - α × dW
        b := b - α × db
    
    Parametreler:
        parameters: W1, b1, W2, b2 dictionary
        gradients: dW1, db1, dW2, db2 dictionary
        learning_rate: α (öğrenme oranı)
    
    Returns:
        parameters: güncellenmiş parametreler
    """

    parameters['W1'] = parameters['W1'] - learning_rate * gradients['dW1']
    parameters['b1'] = parameters['b1'] - learning_rate * gradients['db1']
    
    # Katman 2
    parameters['W2'] = parameters['W2'] - learning_rate * gradients['dW2']
    parameters['b2'] = parameters['b2'] - learning_rate * gradients['db2']

    return parameters

def compute_accuracy(X, Y, parameters):
    """
    Doğruluk oranını hesapla
    
    Adımlar:
        1. Forward pass ile tahmin yap
        2. argmax ile en olası sınıfı bul
        3. Gerçek etiketlerle karşılaştır
        4. Yüzde hesapla
    
    Parametreler:
        X: input verisi (784, m)
        Y: gerçek etiketler one-hot (10, m)
        parameters: W1, b1, W2, b2
    
    Returns:
        accuracy: yüzde olarak doğruluk (0-100)
    """
    
    # Forward pass
    a2, _ = forward_propagation(X, parameters)
    
    # Tahmin edilen sınıflar (her sütunda en yüksek değerin indeksi)
    predictions = np.argmax(a2, axis=0)    # (m,)
    
    # Gerçek sınıflar (one-hot'tan geri çevir)
    labels = np.argmax(Y, axis=0)          # (m,)
    
    # Doğru tahmin sayısı
    correct = np.sum(predictions == labels)
    
    # Toplam örnek sayısı
    total = Y.shape[1]
    
    # Yüzde hesapla
    accuracy = (correct / total) * 100
    
    return accuracy

def train(X_train, Y_train, X_test, Y_test,
          n_h=128, learning_rate=0.1, num_epochs=100,
          print_interval=10):
    """
    Tam eğitim döngüsü
    
    Parametreler:
        X_train, Y_train: eğitim verisi
        X_test, Y_test: test verisi
        n_h: hidden layer nöron sayısı
        learning_rate: öğrenme oranı (α)
        num_epochs: epoch sayısı
        print_interval: kaç epoch'ta bir sonuç yazdır
    
    Returns:
        parameters: eğitilmiş parametreler
        train_losses: epoch başına loss listesi
        train_accuracies: kaydedilen train accuracy'ler
        test_accuracies: kaydedilen test accuracy'ler
    """
    
    # Ağ boyutları
    n_x = X_train.shape[0]  # 784
    n_y = Y_train.shape[0]  # 10
    
    # 1. PARAMETRELERİ BAŞLAT
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Metrikleri saklamak için listeler
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Hiperparametreleri göster
    print(f"\nHiperparametreler:")
    print(f"  Hidden layer:  {n_h} nöron")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Train örnekleri: {X_train.shape[1]}")
    print(f"  Test örnekleri:  {X_test.shape[1]}")
    
    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10}")
    print(f"{'='*60}")
    
    # 2. EĞİTİM DÖNGÜSÜ
    for epoch in range(num_epochs):
        
        # ===== A. FORWARD PASS =====
        a2, cache = forward_propagation(X_train, parameters)
        
        # ===== B. LOSS HESAPLA =====
        loss = compute_loss(a2, Y_train)
        train_losses.append(loss)
        
        # ===== C. BACKPROPAGATION =====
        gradients = backward_propagation(X_train, Y_train, parameters, cache)
        
        # ===== D. GRADIENT DESCENT (UPDATE) =====
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        # ===== E. ACCURACY HESAPLA VE LOGLA =====
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            train_acc = compute_accuracy(X_train, Y_train, parameters)
            test_acc = compute_accuracy(X_test, Y_test, parameters)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"{epoch+1:>6} | {loss:>10.4f} | {train_acc:>9.2f}% | {test_acc:>9.2f}%")
    
    print(f"{'='*60}")
    print(f"Eğitim tamamlandı!")
    
    return parameters, train_losses, train_accuracies, test_accuracies

parameters, train_losses, train_accs, test_accs = train(
    X_train, Y_train,
    X_test, Y_test,
    n_h=128,
    learning_rate=0.1,
    num_epochs=200,
    print_interval=10
)

def plot_training_history(train_losses, train_accs, test_accs, print_interval):
    """
    Eğitim sürecini görselleştir
    
    İki grafik:
    1. Loss vs Epoch
    2. Accuracy vs Epoch (Train ve Test)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LOSS GRAFİĞİ =====
    ax1 = axes[0]
    ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Başlangıç ve bitiş değerlerini annotate et
    ax1.annotate(f'Start: {train_losses[0]:.2f}', 
                 xy=(0, train_losses[0]),
                 xytext=(10, train_losses[0] - 0.3),
                 fontsize=10, color='blue')
    ax1.annotate(f'End: {train_losses[-1]:.2f}', 
                 xy=(len(train_losses)-1, train_losses[-1]),
                 xytext=(len(train_losses)-30, train_losses[-1] + 0.3),
                 fontsize=10, color='blue')
    
    # ===== ACCURACY GRAFİĞİ =====
    ax2 = axes[1]
    
    # Epoch numaralarını hesapla (print_interval'a göre)
    epochs_recorded = [1]  # İlk epoch
    for i in range(1, len(train_accs)):
        epochs_recorded.append(i * print_interval)
    
    ax2.plot(epochs_recorded, train_accs, 'b-o', linewidth=2, 
             markersize=6, label='Train Accuracy')
    ax2.plot(epochs_recorded, test_accs, 'r-o', linewidth=2, 
             markersize=6, label='Test Accuracy')
    
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_ylim([0, 100])
    
    # Final değerleri göster
    ax2.axhline(y=train_accs[-1], color='blue', linestyle='--', alpha=0.3)
    ax2.axhline(y=test_accs[-1], color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(train_losses, train_accs, test_accs, print_interval=10)

def plot_predictions(X, Y, parameters, class_names, num_samples=10):
    """
    Rastgele örnekleri tahmin et ve görselleştir
    
    Yeşil başlık = Doğru tahmin
    Kırmızı başlık = Yanlış tahmin
    """
    
    m = X.shape[1]
    
    # Rastgele örnekler seç
    indices = np.random.choice(m, num_samples, replace=False)
    
    # Tahminleri al
    a2, _ = forward_propagation(X[:, indices], parameters)
    predictions = np.argmax(a2, axis=0)
    true_labels = np.argmax(Y[:, indices], axis=0)
    
    # Görselleştir
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Görüntüyü 28x28'e reshape et
        img = X[:, indices[i]].reshape(28, 28)
        
        pred = predictions[i]
        true = true_labels[i]
        confidence = a2[pred, i] * 100
        
        # Görüntüyü çiz
        axes[i].imshow(img, cmap='gray')
        
        # Doğru/yanlış rengini belirle
        color = 'green' if pred == true else 'red'
        
        # Başlık ekle
        title = f"Tahmin: {class_names[pred]}\n"
        title += f"Gerçek: {class_names[true]}\n"
        title += f"Güven: {confidence:.1f}%"
        
        axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Model Tahminleri (Yeşil=Doğru, Kırmızı=Yanlış)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# Tahminleri görselleştir
plot_predictions(X_test, Y_test, parameters, class_names, num_samples=10)