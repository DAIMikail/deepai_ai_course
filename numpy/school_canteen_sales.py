import numpy as np

# ========================================
# SORU: Okul Kantin Günlük Satışlar
# ========================================
# Okul kantininde 8 ürün satılıyor. 5 günlük satış adetleri ve birim fiyatları
# iki ayrı array'de tutuluyor.
#
# Satışlar: 8×5 array (8 ürün × 5 gün)
# Fiyatlar: 8 elemanlı array [2.5, 3.0, 5.0, 1.5, 4.0, 3.5, 2.0, 6.0]
#
# Görevler:
# 1. Dot product kullanarak her günün toplam cirosunu hesaplayın.
# 2. En çok gelir getiren 3 ürünü bulun.
# 3. Fiyatları %15 artırırsanız toplam ciro ne kadar olur?
# ========================================

# Ürün isimleri
urunler = np.array(['Simit', 'Su', 'Sandviç', 'Çikolata',
                     'Meyve Suyu', 'Kurabiye', 'Kek', 'Ayran'])

# Günler
gunler = np.array(['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma'])

# Birim fiyatlar (TL)
fiyatlar = np.array([2.5, 3.0, 5.0, 1.5, 4.0, 3.5, 2.0, 6.0])

# 8 ürünün 5 günlük satış adetleri (8×5 array)
# Her satır bir ürünü, her sütun bir günü temsil ediyor
satislar = np.array([
    [45, 50, 48, 52, 55],  # Simit (2.5 TL) - popüler ve ucuz
    [80, 85, 82, 88, 90],  # Su (3.0 TL) - en çok satan
    [25, 28, 26, 30, 32],  # Sandviç (5.0 TL) - pahalı ama gelir yüksek
    [60, 65, 62, 68, 70],  # Çikolata (1.5 TL) - ucuz
    [35, 38, 36, 40, 42],  # Meyve Suyu (4.0 TL) - orta
    [40, 42, 41, 45, 48],  # Kurabiye (3.5 TL) - orta
    [30, 32, 31, 35, 38],  # Kek (2.0 TL) - ucuz
    [20, 22, 21, 24, 26],  # Ayran (6.0 TL) - pahalı
])

print("Kantin Satış Verileri:")
print(f"Ürünler ({len(urunler)} adet): {urunler}")
print(f"Günler ({len(gunler)} adet): {gunler}")
print(f"\nBirim Fiyatlar (TL):")
for i, urun in enumerate(urunler):
    print(f"  {urun}: {fiyatlar[i]} TL")

print(f"\nSatışlar Array Shape: {satislar.shape} (8 ürün × 5 gün)")
print("\nHer ürünün 5 günlük satış adetleri:")
for i, urun in enumerate(urunler):
    print(f"  {urun}: {satislar[i]}")

print("\n" + "="*50)
print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
print("="*50 + "\n")

# 1. Dot product kullanarak her günün toplam cirosunu hesaplayın
# İpucu: Her gün için, tüm ürünlerin (satış_adedi × fiyat) toplamını bulmalısınız
# İpucu: np.dot() veya @ operatörü kullanılabilir
# İpucu: satislar.T ile transpose alıp her günü bir sütun yapabilirsiniz


# 2. En çok gelir getiren 3 ürünü bulun
# İpucu: Her ürünün toplam gelirini hesaplayın (5 günlük toplam satış × fiyat)


# 3. Fiyatları %15 artırırsanız toplam ciro ne kadar olur?
# İpucu: Broadcasting kullanarak fiyatlar * 1.15 yapabilirsiniz
