import numpy as np

# ========================================
# SORU: Okul Sınıf Ortalamaları
# ========================================
# Bir sınıfta 20 öğrenci var ve 5 dersten (Matematik, Fizik, Kimya, Biyoloji, Türkçe)
# sınav oldular. Notlar 20×5'lik bir array'de tutuluyor.
#
# Görevler:
# 1. Her öğrencinin not ortalamasını hesaplayın.
# 2. Hangi dersin sınıf ortalaması en düşük?
# 3. Her dersin standart sapmasını bulup, notların en çok dağıldığı dersi tespit edin.
# 4. Son olarak, ortalaması 70'in üzerinde olan öğrencileri maskeleme ile bulun.
# ========================================

# Ders isimleri
dersler = np.array(['Matematik', 'Fizik', 'Kimya', 'Biyoloji', 'Türkçe'])

# 20 öğrencinin 5 dersten aldıkları notlar (20×5 array)
# Her satır bir öğrenciyi, her sütun bir dersi temsil ediyor
notlar = np.array([
    [85, 78, 92, 88, 75],  # Öğrenci 1
    [90, 85, 88, 92, 80],  # Öğrenci 2
    [65, 70, 68, 72, 65],  # Öğrenci 3
    [78, 82, 75, 80, 85],  # Öğrenci 4
    [92, 88, 95, 90, 85],  # Öğrenci 5
    [70, 65, 72, 68, 70],  # Öğrenci 6
    [88, 90, 85, 88, 92],  # Öğrenci 7
    [60, 55, 58, 62, 60],  # Öğrenci 8
    [95, 92, 98, 95, 90],  # Öğrenci 9
    [72, 75, 70, 78, 72],  # Öğrenci 10
    [82, 80, 85, 82, 78],  # Öğrenci 11
    [68, 72, 65, 70, 68],  # Öğrenci 12
    [90, 88, 92, 90, 88],  # Öğrenci 13
    [75, 78, 72, 75, 80],  # Öğrenci 14
    [85, 82, 88, 85, 82],  # Öğrenci 15
    [58, 60, 55, 58, 62],  # Öğrenci 16
    [80, 85, 82, 80, 85],  # Öğrenci 17
    [92, 90, 95, 92, 88],  # Öğrenci 18
    [70, 68, 72, 70, 75],  # Öğrenci 19
    [88, 85, 90, 88, 85],  # Öğrenci 20
])

print("Notlar Array'i Oluşturuldu:")
print(f"Shape: {notlar.shape}")
print(f"Dersler: {dersler}")
print("\nİlk 5 öğrencinin notları:")
print(notlar[:5])
print("\n" + "="*50)
print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
print("="*50 + "\n")

# 1. Her öğrencinin not ortalamasını hesaplayın


# 2. Hangi dersin sınıf ortalaması en düşük?


# 3. Her dersin standart sapmasını bulup, notların en çok dağıldığı dersi tespit edin


# 4. Ortalaması 70'in üzerinde olan öğrencileri maskeleme ile bulun
