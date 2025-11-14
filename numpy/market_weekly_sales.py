import numpy as np

# ========================================
# SORU: Küçük Market Haftalık Satışlar
# ========================================
# Bir marketin 10 ürününün 7 günlük satış miktarları 10×7 array'de saklanıyor.
#
# Görevler:
# 1. Hangi ürün en çok satılmış?
# 2. Hafta sonu (son 2 gün) satışları hafta içi ortalamasından %20 fazla olan ürünleri bulun.
# 3. Satış verilerini normalize edip (0-1 arası), her ürünün performansını karşılaştırın.
# ========================================

# Ürün isimleri
urunler = np.array(['Ekmek', 'Süt', 'Yumurta', 'Su', 'Çay',
                     'Peynir', 'Domates', 'Patates', 'Makarna', 'Deterjan'])

# Günler
gunler = np.array(['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'])

# 10 ürünün 7 günlük satış miktarları (10×7 array)
# Her satır bir ürünü, her sütun bir günü temsil ediyor
# Not: Hafta içi (ilk 5 gün), Hafta sonu (son 2 gün)
satislar = np.array([
    [120, 115, 118, 122, 125, 145, 150],  # Ekmek - hafta sonu artış var
    [85, 88, 82, 90, 87, 95, 92],         # Süt
    [45, 48, 42, 50, 47, 68, 65],         # Yumurta - hafta sonu artış var
    [95, 92, 98, 90, 93, 110, 115],       # Su - hafta sonu artış var
    [30, 28, 32, 29, 31, 35, 33],         # Çay
    [55, 52, 58, 54, 56, 62, 60],         # Peynir
    [72, 70, 68, 75, 73, 95, 98],         # Domates - hafta sonu artış var
    [65, 68, 62, 70, 67, 85, 88],         # Patates - hafta sonu artış var
    [40, 38, 42, 39, 41, 45, 43],         # Makarna
    [25, 22, 28, 24, 26, 30, 32],         # Deterjan
])

print("Satış Array'i Oluşturuldu:")
print(f"Shape: {satislar.shape}")
print(f"Ürünler ({len(urunler)} adet): {urunler}")
print(f"Günler ({len(gunler)} adet): {gunler}")
print("\nİlk 5 ürünün haftalık satışları:")
print(satislar[:5])
print("\n" + "="*50)
print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
print("="*50 + "\n")

# 1. Hangi ürün en çok satılmış?


# 2. Hafta sonu (son 2 gün) satışları hafta içi ortalamasından %20 fazla olan ürünleri bulun
# İpucu: Hafta içi = ilk 5 gün (indeks 0-4), Hafta sonu = son 2 gün (indeks 5-6)


# 3. Satış verilerini normalize edip (0-1 arası), her ürünün performansını karşılaştırın
# İpucu: Normalizasyon formülü = (değer - min) / (max - min)
