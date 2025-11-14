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
# ========================================

# 25 öğrencinin 20 günlük devamsızlık durumu (25×20 binary array)
# 1 = geldi, 0 = gelmedi
# Her satır bir öğrenciyi, her sütun bir günü temsil ediyor

np.random.seed(42)  # Tekrar üretilebilirlik için

# Gerçekçi devamsızlık verisi oluştur
# Çoğu öğrenci çoğu gün gelir, ama bazıları daha fazla devamsızlık yapar
devamsizlik = np.ones((25, 20), dtype=int)  # Başlangıçta herkes geldi

# Bazı öğrenciler için rastgele devamsızlıklar ekle
# Öğrenci 0-15: İyi devam (0-2 devamsızlık)
for i in range(16):
    devamsiz_gunler = np.random.choice(20, size=np.random.randint(0, 3), replace=False)
    devamsizlik[i, devamsiz_gunler] = 0

# Öğrenci 16-20: Orta devam (3-5 devamsızlık)
for i in range(16, 21):
    devamsiz_gunler = np.random.choice(20, size=np.random.randint(3, 6), replace=False)
    devamsizlik[i, devamsiz_gunler] = 0

# Öğrenci 21-23: Kötü devam (6-8 devamsızlık)
for i in range(21, 24):
    devamsiz_gunler = np.random.choice(20, size=np.random.randint(6, 9), replace=False)
    devamsizlik[i, devamsiz_gunler] = 0

# Öğrenci 24: Hiç devamsızlık yok (mükemmel devam)
# Zaten 1'lerle dolu, değişiklik yok

# Bazı günlerde toplu devamsızlık oluştur (örneğin hastalık salgını, kötü hava)
# Gün 7: 6 öğrenci gelmedi
devamsizlik[np.random.choice(25, size=6, replace=False), 7] = 0

# Gün 14: 7 öğrenci gelmedi
devamsizlik[np.random.choice(25, size=7, replace=False), 14] = 0

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
print("ÇÖZÜM BÖLÜMÜ - Buraya kodunuzu yazın")
print("="*50 + "\n")

# 1. Her öğrencinin toplam devamsızlığını hesaplayın
# İpucu: Her satırda 0 olan değerleri sayın veya (20 - toplam gelme sayısı)


# 2. Hangi günlerde 5'ten fazla öğrenci gelmemiş?
# İpucu: Her sütun (gün) için 0 değerlerini sayın


# 3. %80'den fazla devam eden öğrencileri bulun
# İpucu: %80 devam = en az 16 gün gelme (20 günün %80'i = 16)


# 4. where() fonksiyonu ile hiç devamsızlık yapmayan öğrencilerin indekslerini tespit edin
# İpucu: np.where() kullanarak tüm günlerde 1 olan (20 gün gelme) öğrencileri bulun
