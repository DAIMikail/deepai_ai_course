import numpy as np

# ========================================
# SORU: Market Haftalık Satışlar
# ========================================
# Bir marketin 10 ürününün 7 günlük satış miktarları 10×7 array'de saklanıyor.
#
# Görevler:
# 1. Hangi ürün en çok satılmış?
# 2. Hafta sonu (son 2 gün) satışları hafta içi ortalamasından %20 fazla olan ürünleri bulun.
# 3. Satış verilerini normalize edip (0-1 arası), her ürünün performansını karşılaştırın.
#
# ÖĞRENİLECEK NUMPY KAVRAMLARI:
# ✓ axis parametresi (axis=0: sütunlar, axis=1: satırlar)
# ✓ Array slicing ([:, 0:5] gibi dilimleme işlemleri)
# ✓ Agregasyon fonksiyonları (sum, mean, min, max)
# ✓ Index bulma (argmax, argsort)
# ✓ Boolean masking (koşullu filtreleme)
# ✓ Broadcasting (farklı boyutlarda işlemler)
# ✓ keepdims parametresi (boyut koruma)
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

# print("Satış Array'i Oluşturuldu:")
# print(f"Shape: {satislar.shape}")
# print(f"Ürünler ({len(urunler)} adet): {urunler}")
# print(f"Günler ({len(gunler)} adet): {gunler}")
# print("\nİlk 5 ürünün haftalık satışları:")
# print(satislar[:5])
# print("\n" + "="*50)
# print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
# print("="*50 + "\n")

# ============================================================
# 1. Hangi ürün en çok satılmış?
# ============================================================

# ADIM 1: Her ürünün 7 günlük toplam satışını hesapla
# axis=1 → Satırlar boyunca topla (her ürün için tüm günleri topla)
# Sonuç: 10 elemanlı array (her ürün için bir toplam)
toplam_satis = np.sum(satislar, axis=1)
print(f"satislar toplam: {toplam_satis}")

# ADIM 2: En yüksek toplam satış miktarını bul
# max() → Array içindeki en büyük değeri döndürür
en_cok_satis = np.max(toplam_satis)
print(f"en çok satılan ürünün miktarı: {en_cok_satis}")

# ADIM 3: En yüksek satışın hangi sırada olduğunu bul
# argmax() → En büyük değerin INDEX'ini döndürür (değeri değil!)
en_cok_satis_index = np.argmax(toplam_satis)
print(f"en çok satılan ürün index: {en_cok_satis_index}")

# ADIM 4: Index kullanarak ürün ismini bul
# urunler[index] → O indeksteki ürün ismini getirir
en_cok_satilan_urun = urunler[en_cok_satis_index]
print(f"en çok satılan ürün: {en_cok_satilan_urun}")


# ============================================================
# 2. Hafta sonu (son 2 gün) satışları hafta içi ortalamasından %20 fazla olan ürünleri bulun
# ============================================================

# ADIM 1: Hafta içi ve hafta sonu satışlarını ayır (SLICING)
# [:, 0:5] → Tüm satırlar (:), 0-4 arası sütunlar (ilk 5 gün)
# Sonuç: 10×5 array (10 ürün × 5 gün)
hafta_ici_satislar = satislar[:, 0:5]
print(f"hafta ici satislar: \n {hafta_ici_satislar}")

# [:, 5:7] → Tüm satırlar (:), 5-6 arası sütunlar (son 2 gün)
# Sonuç: 10×2 array (10 ürün × 2 gün)
hafta_sonu_satislar = satislar[:, 5:7]
print(f"hafta sonu satislar: \n {hafta_sonu_satislar}")

# ADIM 2: Her ürün için hafta içi ve hafta sonu ortalamalarını hesapla
# axis=1 → Satırlar boyunca ortalama al (her ürün için günlerin ortalaması)
# Sonuç: Her biri 10 elemanlı array (her ürün için bir ortalama)
hafta_ici_ort = np.mean(hafta_ici_satislar, axis=1)
hafta_sonu_ort = np.mean(hafta_sonu_satislar, axis=1)
print(f"Hafta içi ort: {hafta_ici_ort}, Hafta sonu ort: {hafta_sonu_ort}")

# ADIM 3: Boolean mask oluştur (%20 fazla mı kontrolü)
# >= → Büyük veya eşit mi?
# * 1.2 → %20 artış (%100 + %20 = %120 = 1.2 katsayısı)
# Sonuç: [True, False, True, ...] gibi boolean array
kosul_mask = hafta_sonu_ort >= (hafta_ici_ort * 1.2)
print(f"Mask (True=koşul sağlanıyor): {kosul_mask}")

# ADIM 4: Mask ile filtreleme yap
# urunler[mask] → Sadece True olan indekslerdeki ürünleri al
hafta_sonu_populer_urunler = urunler[kosul_mask]
print(f"Hafta sonu daha populer olan ürünler: {hafta_sonu_populer_urunler}")

# ============================================================
# 3. Satış verilerini normalize edip (0-1 arası), her ürünün performansını karşılaştırın
# ============================================================
# NORMALİZASYON NEDİR?
#   - Farklı ölçeklerdeki verileri aynı aralığa (0-1) getirme işlemi
#   - Neden yapılır? Adil karşılaştırma için
#   - Örnek: Ekmek 120 adet, Çay 30 adet satıyor
#            → Direkt karşılaştırma: Ekmek daha iyi görünür
#            → Normalize sonrası: Her ürün kendi kapasitesi içinde değerlendirilir
#   - Formül: (değer - min) / (max - min)
#            → En düşük değer → 0 olur
#            → En yüksek değer → 1 olur
#            → Diğer değerler → 0-1 arası

# ADIM 1: Her ürünün kendi min ve max değerlerini bul
# axis=1 → Her satır (ürün) için ayrı ayrı
# keepdims=True → Boyutu koru (10×1 array yap, broadcasting için gerekli)
#   keepdims olmadan: [120, 85, ...] → (10,) şeklinde
#   keepdims ile:     [[120], [85], ...] → (10, 1) şeklinde
min_vals = satislar.min(axis=1, keepdims=True)  # Her ürünün en düşük satış günü
max_vals = satislar.max(axis=1, keepdims=True)  # Her ürünün en yüksek satış günü

# ADIM 2: Normalizasyon formülünü uygula (her ürün kendi içinde 0-1 arası)
# Broadcasting: (10×7) - (10×1) otomatik olarak her satıra uygulanır
# Formül: (değer - min) / (max - min)
# Sonuç: 0 = o ürünün en kötü günü, 1 = o ürünün en iyi günü
normalize_satislar = (satislar - min_vals) / (max_vals - min_vals)
print(f"Normalize satislar (her ürün kendi içinde 0-1):\n{normalize_satislar}")

# ADIM 3: Her ürünün ortalama performansını hesapla
# mean(axis=1) → Her satır için ortalama (7 günün ortalaması)
# Yüksek ortalama = Genelde yüksek performans göstermiş
# Düşük ortalama = Genelde düşük performans göstermiş
performans = normalize_satislar.mean(axis=1)

# ADIM 4: Performansa göre sırala (en iyiden en kötüye)
# argsort() → Küçükten büyüğe sıralı INDEX'leri döndürür
# [::-1] → Ters çevir (büyükten küçüğe yap)
# Sonuç: En iyi performanstan en kötüye doğru indeksler
sirali_indeksler = performans.argsort()[::-1]

# ADIM 5: Sıralı şekilde yazdır
print("\nPerformans Sıralaması (0-1 arası, yüksek=iyi):")
print("-" * 50)
# enumerate(array, start) → (index, değer) çiftlerini döndürür
# start=1 → Sayımı 1'den başlat
for sira, idx in enumerate(sirali_indeksler, 1):
    print(f"{sira}. {urunler[idx]}: {performans[idx]:.2f}")