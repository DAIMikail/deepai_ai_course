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
#
# ÖĞRENİLECEK NUMPY KAVRAMLARI:
# ✓ 3 boyutlu array işlemleri
# ✓ axis parametresi (axis=0, axis=1, axis=2)
# ✓ Çoklu axis kullanımı (axis=(0,1))
# ✓ Agregasyon fonksiyonları (sum, max, min)
# ✓ Index bulma (argmax, argmin)
# ✓ Broadcasting (skaler ile array çarpımı)
# ✓ 3D array boyut anlama
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

# print("Üretim Array'i Oluşturuldu:")
# print(f"Shape: {uretim.shape} -> (3 vardiya, 5 işçi, 10 gün)")
# print(f"Vardiyalar: {vardiyalar}")
# print(f"İşçiler: {isciler}")
# print(f"Günler: {gunler}")
# print("\nSabah Vardiyası - İlk 3 işçinin ilk 5 günlük üretimi:")
# print(uretim[0, :3, :5])
# print("\n" + "="*50)
# print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
# print("="*50 + "\n")

# ============================================================
# 1. Her vardiyada en verimli işçiyi bulun
# ============================================================

# ADIM 1: Her işçinin 10 günlük toplam üretimini hesapla
# axis=2 → 3. boyut (günler) boyunca topla
# uretim shape: (3 vardiya, 5 işçi, 10 gün)
# Sonuç: (3, 5) array → 3 vardiya × 5 işçi
# Her işçinin 10 günlük toplam üretimi
isci_toplam_uretim = uretim.sum(axis=2)
print(f"Her işçinin toplam üretimi (3 vardiya × 5 işçi):\n{isci_toplam_uretim}\n")

# ADIM 2: Her vardiyada en yüksek üretimi bul
# max(axis=1) → İşçiler boyunca en büyük değeri bul
# Sonuç: (3,) array → Her vardiya için bir değer
en_yuksek_uretim = isci_toplam_uretim.max(axis=1)

# ADIM 3: Her vardiyada en verimli işçinin indeksini bul
# argmax(axis=1) → Her vardiya için en yüksek değerin indeksini döndür (0-4 arası)
# Sonuç: (3,) array → Her vardiya için bir indeks
en_verimli_isci_index = isci_toplam_uretim.argmax(axis=1)

# ADIM 4: Sonuçları detaylı yazdır
print("Vardiya Bazlı En Verimli İşçiler:")
print("-" * 60)
# enumerate(array, start) → (index, değer) çiftleri döndürür
for vardiya_idx, vardiya_adi in enumerate(vardiyalar):
    isci_idx = en_verimli_isci_index[vardiya_idx]
    toplam = en_yuksek_uretim[vardiya_idx]
    gunluk_ort = toplam / 10
    print(f"{vardiya_adi}:")
    print(f"  En verimli: {isciler[isci_idx]}")
    print(f"  Toplam: {toplam} parça")
    print(f"  Günlük ortalama: {gunluk_ort:.1f} parça\n")


# ============================================================
# 2. Hangi gün toplam üretim en az olmuş?
# ============================================================

# ADIM 1: Her gün için toplam üretimi hesapla (tüm vardiyalar + tüm işçiler)
# axis=(0, 1) → İlk iki boyutu topla (vardiyalar ve işçiler)
# axis=0 → Vardiyaları topla
# axis=1 → İşçileri topla
# Sonuç: (10,) array → Her gün için toplam üretim
gunluk_toplam_uretim = uretim.sum(axis=(0, 1))

print("\nGünlük Toplam Üretim (Tüm vardiyalar + Tüm işçiler):")
print("-" * 60)
# Her günün üretimini yazdır
for gun_idx, gun_adi in enumerate(gunler):
    print(f"{gun_adi}: {gunluk_toplam_uretim[gun_idx]} parça")

# ADIM 2: En az üretim miktarını bul
# min() → Array içindeki en küçük değeri döndürür
en_az_uretim = gunluk_toplam_uretim.min()

# ADIM 3: Hangi gün olduğunu bul
# argmin() → En küçük değerin indeksini döndürür (0-9 arası)
en_az_gun_index = gunluk_toplam_uretim.argmin()
en_az_gun = gunler[en_az_gun_index]

# ADIM 4: Sonuçları yazdır
print(f"\n{'='*60}")
print(f"En az üretim yapılan gün: {en_az_gun}")
print(f"Toplam üretim: {en_az_uretim} parça")
print(f"Günlük ortalama: {gunluk_toplam_uretim.mean():.1f} parça")
print(f"Ortalamadan fark: {gunluk_toplam_uretim.mean() - en_az_uretim:.1f} parça")


# ============================================================
# 3. Broadcasting kullanarak tüm işçilerin üretimini %10 artırdığınızda yeni toplam üretimi hesaplayın
# ============================================================
# BROADCASTING NEDİR?
#   - Tek bir değeri (skaler) tüm array elemanlarına otomatik uygulama
#   - NumPy bunu çok hızlı yapar (döngü gerekmez!)
#   - Örnek: [10, 20, 30] * 2 = [20, 40, 60]
#   - 3D array için de aynı şekilde çalışır
#   - Skaler (1.10) otomatik olarak her elemana uygulanır

# ADIM 1: Tüm üretim değerlerini %10 artır
# Broadcasting: 1.10 skaler değeri, (3, 5, 10) array'in her elemanıyla çarpılır
# %10 artış = değer × 1.10 (mevcut değer + %10'u)
# Sonuç: (3, 5, 10) array → Boyut değişmez, sadece değerler artar
yeni_uretim = uretim * 1.10

print("\nBroadcasting ile %10 Artış:")
print("-" * 60)
print(f"Orijinal array shape: {uretim.shape}")
print(f"Yeni array shape: {yeni_uretim.shape}")  # Boyut değişmez

# Örnek: Bir işçinin verisi
print(f"\nÖrnek - Sabah Vardiyası İşçi-1'in ilk 5 günü:")
print(f"  Eski: {uretim[0, 0, :5]}")
print(f"  Yeni: {yeni_uretim[0, 0, :5]}")

# ADIM 2: Eski ve yeni toplam üretimi hesapla
# sum() parametresiz → Tüm elemanları topla (3×5×10 = 150 eleman)
eski_toplam_uretim = uretim.sum()
yeni_toplam_uretim = yeni_uretim.sum()

# ADIM 3: Artış miktarını hesapla
artis_miktari = yeni_toplam_uretim - eski_toplam_uretim

# ADIM 4: Sonuçları karşılaştır
print(f"\nÜretim Karşılaştırması:")
print("-" * 60)
print(f"Eski toplam üretim: {eski_toplam_uretim:.0f} parça")
print(f"Yeni toplam üretim: {yeni_toplam_uretim:.0f} parça")
print(f"Artış miktarı: {artis_miktari:.0f} parça")
print(f"Artış yüzdesi: %{(artis_miktari/eski_toplam_uretim)*100:.1f}")

# ADIM 5: Vardiya bazlı karşılaştırma
print(f"\nVardiya Bazlı Karşılaştırma:")
print("-" * 60)
for vardiya_idx, vardiya_adi in enumerate(vardiyalar):
    eski = uretim[vardiya_idx].sum()
    yeni = yeni_uretim[vardiya_idx].sum()
    artis = yeni - eski
    print(f"{vardiya_adi}:")
    print(f"  Eski: {eski:.0f} parça → Yeni: {yeni:.0f} parça (+{artis:.0f})")
