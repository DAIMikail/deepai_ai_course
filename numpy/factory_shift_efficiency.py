import numpy as np

# ========================================
# SORU: Fabrika Vardiya Verimliliği
# ========================================
# Bir fabrikada 3 vardiya ve her vardiyada 5 işçi çalışıyor.
# 10 günlük üretim miktarları 3×5×10 array'de tutuluyor.
#
# Görevler:
# 1. Her vardiyada en verimli işçiyi bulun.
# 2. Hangi gün toplam üretim en az olmuş?
# 3. Broadcasting kullanarak tüm işçilerin üretimini %10 artırdığınızda yeni toplam üretimi hesaplayın.
# ========================================

# Vardiya isimleri
vardiyalar = np.array(['Sabah Vardiyası', 'Öğle Vardiyası', 'Gece Vardiyası'])

# İşçi isimleri (her vardiyada 5 işçi)
isciler = np.array(['İşçi-1', 'İşçi-2', 'İşçi-3', 'İşçi-4', 'İşçi-5'])

# Günler
gunler = np.array(['Gün-1', 'Gün-2', 'Gün-3', 'Gün-4', 'Gün-5',
                   'Gün-6', 'Gün-7', 'Gün-8', 'Gün-9', 'Gün-10'])

# 3 vardiya × 5 işçi × 10 gün = (3, 5, 10) boyutunda array
# Boyutlar: [vardiya_index, isci_index, gun_index]
# Her hücre o işçinin o gün o vardiyada ürettiği parça sayısını gösterir

uretim = np.array([
    # Sabah Vardiyası (5 işçi × 10 gün)
    [
        [45, 48, 46, 50, 47, 49, 45, 48, 46, 47],  # İşçi-1
        [52, 55, 53, 56, 54, 55, 52, 54, 53, 55],  # İşçi-2 (en verimli)
        [40, 42, 41, 43, 42, 44, 40, 42, 41, 43],  # İşçi-3
        [48, 50, 49, 51, 50, 52, 48, 50, 49, 51],  # İşçi-4
        [38, 40, 39, 41, 40, 42, 38, 40, 39, 41],  # İşçi-5
    ],
    # Öğle Vardiyası (5 işçi × 10 gün)
    [
        [42, 44, 43, 45, 44, 46, 42, 44, 43, 45],  # İşçi-1
        [46, 48, 47, 49, 48, 50, 46, 48, 47, 49],  # İşçi-2
        [50, 52, 51, 53, 52, 54, 50, 52, 51, 53],  # İşçi-3 (en verimli)
        [44, 46, 45, 47, 46, 48, 44, 46, 45, 47],  # İşçi-4
        [40, 42, 41, 43, 42, 44, 40, 42, 41, 43],  # İşçi-5
    ],
    # Gece Vardiyası (5 işçi × 10 gün)
    [
        [38, 40, 39, 41, 40, 42, 38, 40, 39, 41],  # İşçi-1
        [44, 46, 45, 47, 46, 48, 44, 46, 45, 47],  # İşçi-2
        [36, 38, 37, 39, 38, 40, 36, 38, 37, 39],  # İşçi-3
        [48, 50, 49, 51, 50, 52, 48, 50, 49, 51],  # İşçi-4 (en verimli)
        [42, 44, 43, 45, 44, 46, 42, 44, 43, 45],  # İşçi-5
    ],
])

print("Üretim Array'i Oluşturuldu:")
print(f"Shape: {uretim.shape} -> (3 vardiya, 5 işçi, 10 gün)")
print(f"Vardiyalar: {vardiyalar}")
print(f"İşçiler: {isciler}")
print(f"Günler: {gunler}")
print("\nSabah Vardiyası - İlk 3 işçinin ilk 5 günlük üretimi:")
print(uretim[0, :3, :5])
print("\n" + "="*50)
print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
print("="*50 + "\n")

# 1. Her vardiyada en verimli işçiyi bulun
# İpucu: Her işçinin toplam üretimini hesaplayıp, her vardiya için max değeri bulun


# 2. Hangi gün toplam üretim en az olmuş?
# İpucu: Tüm vardiyalar ve işçiler için günlük toplamları hesaplayın


# 3. Broadcasting kullanarak tüm işçilerin üretimini %10 artırdığınızda yeni toplam üretimi hesaplayın
# İpucu: uretim * 1.10 işlemi broadcasting ile tüm elemanlara uygulanır
