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
#
# ÖĞRENİLECEK NUMPY KAVRAMLARI:
# ✓ Dot product (iç çarpım) / Matris çarpımı
# ✓ @ operatörü (matris çarpım operatörü)
# ✓ Transpose (.T) - Satır/sütun değişimi
# ✓ Broadcasting (farklı boyutlarda işlemler)
# ✓ argsort() - Sıralama indeksleri
# ✓ Fancy indexing (indeks dizisi ile seçim)
# ✓ Element-wise çarpma (*) vs dot product farkı
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
print("ÇÖZÜM BÖLÜMÜ")
print("="*50 + "\n")

# ============================================================
# 1. Dot product kullanarak her günün toplam cirosunu hesaplayın
# ============================================================
# DOT PRODUCT (İÇ ÇARPIM) NEDİR?
#   - İki vektörün/matrisin çarpımı
#   - Element-wise çarpım (*) ile FARKLI!
#   - Örnek: [2, 3] dot [4, 5] = 2×4 + 3×5 = 8 + 15 = 23
#   - Matris çarpımında kullanılır
#   - Günlük ciro = Σ(satış_adedi × fiyat) → Dot product ile!

# ADIM 1: Transpose ile satışlar array'ini uygun hale getir
# satislar: (8, 5) → 8 ürün × 5 gün
# satislar.T: (5, 8) → 5 gün × 8 ürün (her satır bir gün)
satislar_transpose = satislar.T
print(f"Satışlar shape: {satislar.shape} (8 ürün × 5 gün)")
print(f"Transpose shape: {satislar_transpose.shape} (5 gün × 8 ürün)")

# ADIM 2: Dot product ile her günün cirosunu hesapla
# Yöntem 1: np.dot() fonksiyonu
gunluk_ciro = np.dot(satislar_transpose, fiyatlar)
# Yöntem 2: @ operatörü (aynı sonuç, daha modern)
# gunluk_ciro = satislar_transpose @ fiyatlar

# Her gün için:
# Gün-1 ciro = Simit_adet×Simit_fiyat + Su_adet×Su_fiyat + ... (8 ürün)
# Gün-1 ciro = [45, 80, 25, 60, 35, 40, 30, 20] dot [2.5, 3.0, 5.0, 1.5, 4.0, 3.5, 2.0, 6.0]

print(f"\nHer Günün Toplam Cirosu:")
print("-" * 60)
for gun_idx, gun_adi in enumerate(gunler):
    print(f"{gun_adi}: {gunluk_ciro[gun_idx]:.2f} TL")

# Toplam 5 günlük ciro
toplam_ciro = gunluk_ciro.sum()
print(f"\n5 Günlük Toplam Ciro: {toplam_ciro:.2f} TL")
print(f"Günlük Ortalama Ciro: {gunluk_ciro.mean():.2f} TL")

# ADIM 3: Alternatif yöntemler (eğitim amaçlı)
print(f"\nAlternatif Yöntemler (aynı sonuç):")
# Yöntem 1: @ operatörü
gunluk_ciro_v2 = satislar.T @ fiyatlar
print(f"@ operatörü: {gunluk_ciro_v2}")

# Yöntem 2: Manuel (döngüsüz)
gunluk_ciro_v3 = (satislar.T * fiyatlar).sum(axis=1)
print(f"Element-wise çarpma + sum: {gunluk_ciro_v3}")


# ============================================================
# 2. En çok gelir getiren 3 ürünü bulun
# ============================================================

# ADIM 1: Her ürünün 5 günlük toplam satış adedini hesapla
# sum(axis=1) → Satırlar boyunca topla (her ürün için 5 günün toplamı)
urun_toplam_satis = satislar.sum(axis=1)
print(f"\nHer Ürünün Toplam Satış Adedi (5 gün):")
print(urun_toplam_satis)

# ADIM 2: Her ürünün toplam gelirini hesapla
# Toplam gelir = Toplam satış adedi × Birim fiyat
# Element-wise çarpma (*) kullanıyoruz (dot product değil!)
urun_gelir = urun_toplam_satis * fiyatlar
print(f"\nHer Ürünün Toplam Geliri:")
print("-" * 60)
for i, urun in enumerate(urunler):
    print(f"{urun:<15}: {urun_toplam_satis[i]:>3} adet × {fiyatlar[i]:>4.2f} TL = {urun_gelir[i]:>6.2f} TL")

# ADIM 3: Gelire göre sıralama indekslerini bul
# argsort() → Küçükten büyüğe sıralı indeksler
# [::-1] → Ters çevir (büyükten küçüğe)
# [-3:] → Son 3 eleman (en yüksek 3 gelir)
sirali_indeksler = urun_gelir.argsort()[::-1]  # Büyükten küçüğe
en_cok_gelir_indeksler = sirali_indeksler[:3]   # İlk 3'ü al

# ADIM 4: En çok gelir getiren 3 ürünü göster
print(f"\nEn Çok Gelir Getiren 3 Ürün:")
print("-" * 60)
for sira, idx in enumerate(en_cok_gelir_indeksler, 1):
    print(f"{sira}. {urunler[idx]}: {urun_gelir[idx]:.2f} TL")
    print(f"   ({urun_toplam_satis[idx]} adet × {fiyatlar[idx]:.2f} TL)")


# ============================================================
# 3. Fiyatları %15 artırırsanız toplam ciro ne kadar olur?
# ============================================================

# ADIM 1: Yeni fiyatları hesapla (Broadcasting)
# Broadcasting: Skaler (1.15) ile array çarpımı
# %15 artış = fiyat × 1.15
yeni_fiyatlar = fiyatlar * 1.15

print(f"\nFiyat Karşılaştırması (%15 artış):")
print("-" * 60)
print(f"{'Ürün':<15} {'Eski Fiyat':>12} {'Yeni Fiyat':>12} {'Artış':>10}")
print("-" * 60)
for i, urun in enumerate(urunler):
    artis = yeni_fiyatlar[i] - fiyatlar[i]
    print(f"{urun:<15} {fiyatlar[i]:>10.2f} TL {yeni_fiyatlar[i]:>10.2f} TL {artis:>8.2f} TL")

# ADIM 2: Yeni fiyatlarla günlük ciroyu hesapla
# Dot product kullanarak (aynı mantık)
yeni_gunluk_ciro = satislar.T @ yeni_fiyatlar

print(f"\nYeni Günlük Cirolar:")
print("-" * 60)
print(f"{'Gün':<15} {'Eski Ciro':>12} {'Yeni Ciro':>12} {'Fark':>10}")
print("-" * 60)
for gun_idx, gun_adi in enumerate(gunler):
    eski = gunluk_ciro[gun_idx]
    yeni = yeni_gunluk_ciro[gun_idx]
    fark = yeni - eski
    print(f"{gun_adi:<15} {eski:>10.2f} TL {yeni:>10.2f} TL {fark:>8.2f} TL")

# ADIM 3: Toplam ciro karşılaştırması
eski_toplam_ciro = gunluk_ciro.sum()
yeni_toplam_ciro = yeni_gunluk_ciro.sum()
ciro_artisi = yeni_toplam_ciro - eski_toplam_ciro

print(f"\nToplam Ciro Karşılaştırması:")
print("=" * 60)
print(f"Eski Toplam Ciro: {eski_toplam_ciro:.2f} TL")
print(f"Yeni Toplam Ciro: {yeni_toplam_ciro:.2f} TL")
print(f"Ciro Artışı: {ciro_artisi:.2f} TL (%{(ciro_artisi/eski_toplam_ciro)*100:.1f})")
