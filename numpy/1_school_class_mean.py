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
#
# ÖĞRENİLECEK NUMPY KAVRAMLARI:
# ✓ axis parametresi (axis=0: sütunlar, axis=1: satırlar)
# ✓ Agregasyon fonksiyonları (mean, std, min, max)
# ✓ Index bulma (argmin, argmax)
# ✓ Boolean masking (koşullu filtreleme)
# ✓ np.where() fonksiyonu (koşula göre indeks bulma)
# ✓ Standart sapma kavramı
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

# print("Notlar Array'i Oluşturuldu:")
# print(f"Shape: {notlar.shape}")
# print(f"Dersler: {dersler}")
# print("\nİlk 5 öğrencinin notları:")
# print(notlar[:5])
# print("\n" + "="*50)
# print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
# print("="*50 + "\n")

# ============================================================
# 1. Her öğrencinin not ortalamasını hesaplayın
# ============================================================

# ADIM 1: Her öğrencinin 5 dersten aldığı notların ortalamasını hesapla
# axis=1 → Satırlar boyunca ortalama al (her öğrenci için derslerinin ortalaması)
# Sonuç: 20 elemanlı array (her öğrenci için bir ortalama)
her_ogrenci_ort = np.mean(notlar, axis=1)
print(f"Her bir öğrencinin ortalaması: {her_ogrenci_ort}")
print(f"Örnek: Öğrenci-1'in ortalaması: {her_ogrenci_ort[0]:.2f}")

# ============================================================
# 2. Hangi dersin sınıf ortalaması en düşük?
# ============================================================

# ADIM 1: Her dersin sınıf ortalamasını hesapla
# axis=0 → Sütunlar boyunca ortalama al (her ders için tüm öğrencilerin ortalaması)
# Sonuç: 5 elemanlı array (her ders için bir ortalama)
ders_ortalamalari = notlar.mean(axis=0)
print(f"\nDerslerin ortalaması: {ders_ortalamalari}")

# Her dersin ortalamasını ismiyle birlikte göster
for i, ders in enumerate(dersler):
    print(f"  {ders}: {ders_ortalamalari[i]:.2f}")

# ADIM 2: En düşük ortalamayı bul
# min() → Array içindeki en küçük değeri döndürür
en_dusuk_ortalama = np.min(ders_ortalamalari)
print(f"\nEn düşük ortalama: {en_dusuk_ortalama:.2f}")

# ADIM 3: En düşük ortalamanın hangi ders olduğunu bul
# argmin() → En küçük değerin INDEX'ini döndürür
en_dusuk_ders_index = np.argmin(ders_ortalamalari)
print(f"En düşük ortalamalı ders index: {en_dusuk_ders_index}")

# ADIM 4: Index kullanarak ders ismini bul
en_dusuk_ders = dersler[en_dusuk_ders_index]
print(f"En düşük ortalamalı ders: {en_dusuk_ders}")

# ============================================================
# 3. Her dersin standart sapmasını bulup, notların en çok dağıldığı dersi tespit edin
# ============================================================
# STANDART SAPMA NEDİR?
#   - Standart sapma, verilerin ortalamadan ne kadar uzaklaştığını gösteren bir ölçüdür
#   - Yüksek standart sapma = Notlar birbirinden çok farklı (heterojen sınıf)
#   - Düşük standart sapma = Notlar birbirine yakın (homojen sınıf)
#   - Örnek: Matematik [50, 90] → Yüksek sapma, Fizik [75, 77] → Düşük sapma

# ADIM 1: Her dersin standart sapmasını hesapla
# std() → Standart sapma hesaplar
# axis=0 → Sütunlar boyunca (her ders için tüm öğrencilerin sapması)
# Sonuç: 5 elemanlı array (her ders için bir standart sapma)
ders_standart_sapmalari = np.std(notlar, axis=0)
print(f"\nDers standart sapmaları: {ders_standart_sapmalari}")

# Her dersin standart sapmasını ismiyle birlikte göster
for i, ders in enumerate(dersler):
    print(f"  {ders}: {ders_standart_sapmalari[i]:.2f}")

# ADIM 2: En yüksek standart sapmayı bul
# En çok dağılan = En yüksek standart sapma
en_yuksek_std = np.max(ders_standart_sapmalari)
print(f"\nEn yüksek standart sapma: {en_yuksek_std:.2f}")

# ADIM 3: Hangi ders olduğunu bul
# argmax() → En büyük değerin indeksini döndürür
en_dagilan_ders_index = np.argmax(ders_standart_sapmalari)
print(f"En dağılan ders index: {en_dagilan_ders_index}")

# ADIM 4: Ders ismini al
en_dagilan_ders = dersler[en_dagilan_ders_index]
print(f"Notların en çok dağıldığı ders: {en_dagilan_ders}")

# ============================================================
# 4. Ortalaması 70'in üzerinde olan öğrencileri maskeleme ile bulun
# ============================================================

# ADIM 1: Her öğrencinin not ortalamasını hesapla (eğer henüz yoksa)
# axis=1 → Satırlar boyunca (her öğrenci için derslerinin ortalaması)
ogrenci_ortalamalari = np.mean(notlar, axis=1)
print(f"\nÖğrenci ortalamaları: {ogrenci_ortalamalari}")

# ADIM 2: Boolean mask oluştur (70'ten büyük mü?)
# > operatörü → Her elemana uygulanır ve True/False döndürür
# Sonuç: [True, True, False, ...] gibi boolean array
basarili_ogr_mask = ogrenci_ortalamalari > 70
print(f"\nBoolean Mask (True=başarılı): {basarili_ogr_mask}")

# Başarılı öğrenci sayısı
basarili_sayi = basarili_ogr_mask.sum()  # True=1, False=0 olarak sayılır
print(f"Başarılı öğrenci sayısı: {basarili_sayi}")

# ADIM 3: np.where() ile başarılı öğrencilerin indekslerini bul
# np.where(mask) → True olan yerlerin indekslerini döndürür
# [0] → Tuple'dan ilk elemanı (indeks array'ini) al
hangi_ogr_basarili = np.where(basarili_ogr_mask)[0]
print(f"Başarılı öğrencilerin indeksleri: {hangi_ogr_basarili}")

# ADIM 4: Detaylı gösterim
print(f"\nBaşarılı öğrenciler (Ortalama > 70):")
print("-" * 50)
for idx in hangi_ogr_basarili:
    print(f"  Öğrenci-{idx+1}: {ogrenci_ortalamalari[idx]:.2f}")

# Alternatif: Mask ile direkt filtreleme
basarili_ogrenci_ortalamalari = ogrenci_ortalamalari[basarili_ogr_mask]
print(f"\nBaşarılı öğrencilerin ortalamaları: {basarili_ogrenci_ortalamalari}")


