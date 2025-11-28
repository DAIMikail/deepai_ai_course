import numpy as np

# ========================================
# SORU: Sınıf Devamsızlık Takibi
# ========================================
# 25 öğrencinin 20 günlük devamsızlık durumu 25×20 binary array'de tutuluyor.
# 1 = geldi, 0 = gelmedi
#
# Görevler:
# 1. Her öğrencinin toplam devamsızlığını hesaplayın.
# 2. Hangi günlerde 5'ten fazla öğrenci gelmemiş?
# 3. %80'den fazla devam eden öğrencileri bulun.
# 4. where() fonksiyonu ile hiç devamsızlık yapmayan öğrencilerin indekslerini tespit edin.
#
# ÖĞRENİLECEK NUMPY KAVRAMLARI:
# ✓ Binary array işlemleri (0 ve 1 değerleri)
# ✓ axis parametresi ile toplama (axis=0 ve axis=1)
# ✓ Boolean operatörleri (0 sayma, koşullu filtreleme)
# ✓ np.where() fonksiyonu (koşul bazlı indeks bulma)
# ✓ Boolean masking (koşula göre filtreleme)
# ✓ Binary array'de matematiksel işlemler (toplama = sayma)
# ✓ Yüzdelik hesaplamalar
# ========================================

# 25 öğrencinin 20 günlük devamsızlık durumu (25×20 binary array)
# 1 = geldi, 0 = gelmedi
# Her satır bir öğrenciyi, her sütun bir günü temsil ediyor

devamsizlik = np.array([
    # Öğrenci 1-10: İyi devam eden öğrenciler (0-2 devamsızlık)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 1: 0 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 2: 1 devamsızlık
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 3: 1 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 4: 1 devamsızlık
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # Öğrenci 5: 2 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 6: 1 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],  # Öğrenci 7: 1 devamsızlık
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 8: 1 devamsızlık
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],  # Öğrenci 9: 2 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 10: 0 devamsızlık

    # Öğrenci 11-16: İyi devam eden öğrenciler (devam)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # Öğrenci 11: 1 devamsızlık
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 12: 1 devamsızlık
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # Öğrenci 13: 2 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 14: 0 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],  # Öğrenci 15: 1 devamsızlık
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Öğrenci 16: 2 devamsızlık

    # Öğrenci 17-21: Orta düzey devam (3-5 devamsızlık)
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # Öğrenci 17: 3 devamsızlık
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],  # Öğrenci 18: 3 devamsızlık
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # Öğrenci 19: 4 devamsızlık
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # Öğrenci 20: 4 devamsızlık
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],  # Öğrenci 21: 4 devamsızlık

    # Öğrenci 22-24: Kötü devam (6-8 devamsızlık)
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],  # Öğrenci 22: 6 devamsızlık
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],  # Öğrenci 23: 7 devamsızlık
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Öğrenci 24: 7 devamsızlık

    # Öğrenci 25: Mükemmel devam (hiç devamsızlık yok)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Öğrenci 25: 0 devamsızlık
], dtype=int)

print("Devamsızlık Array'i Oluşturuldu:")
print(f"Shape: {devamsizlik.shape} (25 öğrenci × 20 gün)")
print(f"Binary değerler: 1 = geldi, 0 = gelmedi")
print("\nİlk 5 öğrencinin ilk 10 günlük devam durumu:")
print(devamsizlik[:5, :10])
print("\nİlk 5 öğrencinin devam istatistiği:")
for i in range(5):
    gelme_sayisi = np.sum(devamsizlik[i])
    devamsizlik_sayisi = 20 - gelme_sayisi
    print(f"  Öğrenci-{i+1}: {gelme_sayisi} gün geldi, {devamsizlik_sayisi} gün gelmedi")

print("\n" + "="*50)
print("ÇÖZÜM BÖLÜMÜ")
print("="*50 + "\n")

# ============================================================
# 1. Her öğrencinin toplam devamsızlığını hesaplayın
# ============================================================
# BİNARY ARRAY'DE TOPLAMA = SAYMA
#   - 1'leri toplamak = Gelme sayısını bulmak
#   - 0'ları saymak = 20 - toplam 1 sayısı
#   - Binary array'de sum() çok kullanışlı!

# ADIM 1: Her öğrencinin gelme sayısını hesapla
# sum(axis=1) → Satırlar boyunca topla (her öğrenci için 20 günün toplamı)
# Binary array'de sum() = 1'lerin sayısı = Gelme sayısı
ogrenci_gelme_sayisi = devamsizlik.sum(axis=1)

# ADIM 2: Devamsızlık sayısını hesapla
# Devamsızlık = Toplam gün - Gelme sayısı
# Devamsızlık = 20 - Gelme sayısı
ogrenci_devamsizlik_sayisi = 20 - ogrenci_gelme_sayisi

print("Her Öğrencinin Devamsızlık Durumu:")
print("-" * 70)
print(f"{'Öğrenci':<12} {'Gelme':>8} {'Devamsızlık':>12} {'Devam %':>10}")
print("-" * 70)
for i in range(25):
    gelme = ogrenci_gelme_sayisi[i]
    devamsiz = ogrenci_devamsizlik_sayisi[i]
    yuzde = (gelme / 20) * 100
    print(f"Öğrenci-{i+1:<3} {gelme:>8} {devamsiz:>12} {yuzde:>9.1f}%")

# Ortalama devamsızlık
print(f"\nOrtalama devamsızlık: {ogrenci_devamsizlik_sayisi.mean():.2f} gün")
print(f"En fazla devamsızlık: {ogrenci_devamsizlik_sayisi.max()} gün")
print(f"En az devamsızlık: {ogrenci_devamsizlik_sayisi.min()} gün")


# ============================================================
# 2. Hangi günlerde 5'ten fazla öğrenci gelmemiş?
# ============================================================

# ADIM 1: Her gün için gelmeyen öğrenci sayısını hesapla
# sum(axis=0) → Sütunlar boyunca topla (her gün için 25 öğrencinin toplamı)
# devamsizlik = 1 (geldi), biz 0 (gelmedi) sayısını istiyoruz
# Yöntem 1: 25 - gelme sayısı
gunluk_gelme_sayisi = devamsizlik.sum(axis=0)
gunluk_devamsizlik_sayisi = 25 - gunluk_gelme_sayisi

# Yöntem 2 (Alternatif): 0'ları direkt sayma
# gunluk_devamsizlik_sayisi = (devamsizlik == 0).sum(axis=0)

print(f"\nGünlük Devamsızlık Analizi:")
print("-" * 60)
print(f"{'Gün':<8} {'Gelen':>10} {'Gelmeyen':>12} {'Gelmeyen %':>15}")
print("-" * 60)
for gun_idx in range(20):
    gelen = gunluk_gelme_sayisi[gun_idx]
    gelmeyen = gunluk_devamsizlik_sayisi[gun_idx]
    yuzde = (gelmeyen / 25) * 100
    # 5'ten fazla gelmeyen varsa işaretle
    isaret = " ⚠️" if gelmeyen > 5 else ""
    print(f"Gün-{gun_idx+1:<3} {gelen:>10} {gelmeyen:>12} {yuzde:>14.1f}%{isaret}")

# ADIM 2: 5'ten fazla öğrenci gelmeyen günleri bul
# Boolean mask oluştur
fazla_devamsizlik_mask = gunluk_devamsizlik_sayisi > 5

# ADIM 3: Bu günlerin indekslerini bul
fazla_devamsizlik_gunler = np.where(fazla_devamsizlik_mask)[0]

print(f"\n5'ten Fazla Öğrenci Gelmeyen Günler:")
print("-" * 60)
if len(fazla_devamsizlik_gunler) > 0:
    for gun_idx in fazla_devamsizlik_gunler:
        print(f"  Gün-{gun_idx+1}: {gunluk_devamsizlik_sayisi[gun_idx]} öğrenci gelmedi")
else:
    print("  Hiçbir günde 5'ten fazla öğrenci gelmemiş.")

print(f"\nToplam {len(fazla_devamsizlik_gunler)} gün tespit edildi.")


# ============================================================
# 3. %80'den fazla devam eden öğrencileri bulun
# ============================================================
# %80 DEVAM NE DEMEK?
#   - 20 günün %80'i = 16 gün
#   - En az 16 gün gelmiş olmalı
#   - Gelme sayısı >= 16

# ADIM 1: %80'den fazla devam eden öğrencileri filtrele
# %80 devam = en az 16 gün gelme (20 × 0.80 = 16)
min_devam_gunu = int(20 * 0.80)  # 16 gün

# ADIM 2: Boolean mask oluştur
yuksek_devam_mask = ogrenci_gelme_sayisi >= min_devam_gunu

# ADIM 3: Mask'i uygula
yuksek_devam_ogrenci_indeksleri = np.where(yuksek_devam_mask)[0]
yuksek_devam_sayilari = ogrenci_gelme_sayisi[yuksek_devam_mask]

print(f"\n%80'den Fazla Devam Eden Öğrenciler (En az {min_devam_gunu} gün):")
print("-" * 60)
print(f"{'Öğrenci':<15} {'Gelme Sayısı':>15} {'Devam %':>12}")
print("-" * 60)
for idx in yuksek_devam_ogrenci_indeksleri:
    gelme = ogrenci_gelme_sayisi[idx]
    yuzde = (gelme / 20) * 100
    print(f"Öğrenci-{idx+1:<8} {gelme:>15} {yuzde:>11.1f}%")

print(f"\nToplam {len(yuksek_devam_ogrenci_indeksleri)} öğrenci %80'den fazla devam etti.")
print(f"Yüzde: {(len(yuksek_devam_ogrenci_indeksleri) / 25) * 100:.1f}% (sınıfın)")


# ============================================================
# 4. where() fonksiyonu ile hiç devamsızlık yapmayan öğrencilerin indekslerini tespit edin
# ============================================================
# HİÇ DEVAMSIZLIK YAPMAYAN = TÜM GÜNLER 1
#   - 20 gün gelme sayısı = 20
#   - Tüm satırda sadece 1'ler var

# ADIM 1: Hiç devamsızlık yapmayan öğrencileri bul
# Yöntem 1: Gelme sayısı == 20 olanlar
mukemmel_devam_mask = ogrenci_gelme_sayisi == 20

# ADIM 2: np.where() ile indeksleri bul
# np.where(condition) → condition True olan indeksler
# [0] → Tuple'dan array'i çıkar
mukemmel_devam_indeksleri = np.where(mukemmel_devam_mask)[0]

print(f"\nHiç Devamsızlık Yapmayan Öğrenciler:")
print("-" * 60)
if len(mukemmel_devam_indeksleri) > 0:
    print(f"Toplam {len(mukemmel_devam_indeksleri)} öğrenci hiç devamsızlık yapmamış:\n")
    for idx in mukemmel_devam_indeksleri:
        print(f"  Öğrenci-{idx+1}: 20/20 gün devam (Mükemmel! 🌟)")

    # Detaylı kontrol (doğrulama)
    print(f"\nDoğrulama (ilk mükemmel öğrenci):")
    ilk_mukemmel = mukemmel_devam_indeksleri[0]
    print(f"  Öğrenci-{ilk_mukemmel+1} devam durumu:")
    print(f"  {devamsizlik[ilk_mukemmel]}")
    print(f"  Tüm değerler 1 mi? {np.all(devamsizlik[ilk_mukemmel] == 1)}")
else:
    print("  Hiçbir öğrenci mükemmel devam göstermemiş.")

# ADIM 3: Alternatif yöntemler (eğitim amaçlı)
print(f"\nAlternatif Yöntemler (aynı sonuç):")
# Yöntem 1: np.where() ile gelme sayısı kontrolü
method1 = np.where(ogrenci_gelme_sayisi == 20)[0]
print(f"Yöntem 1 (gelme == 20): {method1}")

# Yöntem 2: np.all() ile satır kontrolü
# Her satırın tüm elemanları 1 mi kontrol et
mukemmel_ogrenciler_v2 = []
for i in range(25):
    if np.all(devamsizlik[i] == 1):
        mukemmel_ogrenciler_v2.append(i)
print(f"Yöntem 2 (np.all()): {mukemmel_ogrenciler_v2}")

# Yöntem 3: Devamsızlık sayısı == 0 olanlar
method3 = np.where(ogrenci_devamsizlik_sayisi == 0)[0]
print(f"Yöntem 3 (devamsızlık == 0): {method3}")
