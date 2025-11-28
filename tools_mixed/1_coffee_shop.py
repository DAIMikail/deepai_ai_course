"""
SENARYO: Kafe

GÖREVLER VE KAZANIMLAR:

1.  [SQL] Veri Bütünlüğü ve Ön İşleme
    -> Amaç: İlişkisel veritabanı sorguları (SQL) kullanılarak veri setindeki anomali teşhisi, 
    aykırı değerlerin tespiti ve eksik veri tamamlama yöntemlerinin uygulanması.

2.  [Numpy] İstatistiksel Hesaplama
    -> Amaç: Sayısal veri dizileri üzerinde vektörel işlemler gerçekleştirerek, 
    merkezi eğilim ölçülerinin hesaplanması ve performans metriklerinin analiz edilmesi.

3.  [Pandas] Keşifsel Veri Analizi (EDA)
    -> Amaç: Veri çerçeveleri üzerinde gruplandırma, filtreleme ve toplulaştırma 
    teknikleri ile temel metriklerin çıkarılması.

4.  [Pandas] Zaman Serisi ve Eğilim Analizi
    -> Amaç: Tarihsel verilerin datetime formatına dönüştürülmesi ve zamana bağlı 
    satış trendlerinin (günlük/dönemsel dalgalanmalar) incelenmesi.

5.  [Pandas] Çok Boyutlu Veri Analizi (Pivot & Korelasyon)
    -> Amaç: Değişkenler arasındaki ilişkilerin (korelasyon matrisi) ve kategorik 
    kırılımların (pivot tablolar) incelenerek stratejik içgörülerin elde edilmesi.

6.  [Matplotlib & Seaborn] Karşılaştırmalı Veri Görselleştirme
    -> Amaç: Matplotlib'in temel çizim yetenekleri ile Seaborn'un istatistiksel görselleştirme 
    kolaylığının (otomatik gruplama, renklendirme) karşılaştırmalı olarak analiz edilmesi.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg') # Not: Bazı ortamlarda backend hatası verirse bu satırı aktif edin.
import matplotlib.pyplot as plt
import seaborn as sns
import os

def veritabani_hazirlik(cursor):
    cursor.execute("DROP TABLE IF EXISTS satislar")
    cursor.execute('''
    CREATE TABLE satislar (
        ID INTEGER PRIMARY KEY,
        Tarih TEXT,
        Kahve_Turu TEXT,
        Boyut TEXT,
        Fiyat REAL,
        Kalori INTEGER,
        Sube TEXT,
        Musteri_Puani REAL,
        Gunluk_Satis INTEGER,
        Barista TEXT
    )
    ''')

    veriler = [
        (1, '2023-11-01', 'Latte', 'Orta', 45.0, 220, 'Kadikoy', 4.5, 120, 'Ali'),
        (2, '2023-11-01', 'Espresso', 'Kucuk', 30.0, 10, 'Besiktas', 4.8, 200, 'Ayse'),
        (3, '2023-11-02', 'Mocha', 'Buyuk', 5000.0, 450, 'Kadikoy', 4.9, 90, 'Ali'),
        (4, '2023-11-02', 'Latte', 'Kucuk', 40.0, None, 'Besiktas', 3.5, 110, 'Mehmet'),
        (5, '2023-11-03', 'Americano', 'Orta', 35.0, 15, 'Kadikoy', 4.2, 150, 'Zeynep'),
        (6, '2023-11-03', 'Cappuccino', 'Orta', 45.0, 180, 'Uskudar', 4.6, 130, 'Can'),
        (7, '2023-11-04', 'Latte', 'Buyuk', 55.0, 280, 'Besiktas', 2.1, 50, 'Ayse'), 
        (8, '2023-11-04', 'Filtre', 'Orta', 30.0, 5, 'Uskudar', 4.0, 180, 'Can'),
        (9, '2023-11-05', 'Mocha', 'Kucuk', 48.0, 320, 'Kadikoy', 4.7, 95, 'Zeynep'),
        (10, '2023-11-05', 'Macchiato', 'Kucuk', 42.0, None, 'Besiktas', 4.3, 100, 'Mehmet')
    ]
    cursor.executemany('INSERT INTO satislar VALUES (?,?,?,?,?,?,?,?,?,?)', veriler)
    print("--- [Sistem] Veritabanı Hazırlandı ---")

# --- ANALİZ FONKSİYONLARI ---

def sql_temizlik(conn, cursor):
    print("\n--- 1. GÖREV: SQL Temizlik ---")
    # Hatalı fiyatları gör ve düzelt
    cursor.execute("UPDATE satislar SET Fiyat = 55 WHERE Fiyat > 100")
    
    # Null kalorileri gör ve düzelt
    cursor.execute("UPDATE satislar SET Kalori = 100 WHERE Kalori IS NULL")
    
    conn.commit()
    print("Veri temizliği tamamlandı.")

def sql_veri_analizi(cursor):
    print("\n--- 1.5. GÖREV: SQL Veri Analizi ---")

    # 1. Şube Bazlı Analiz
    print("\n[Şube Bazlı Analiz]")
    cursor.execute("""
        SELECT
            Sube,
            COUNT(*) as Toplam_Islem,
            AVG(Fiyat) as Ort_Fiyat,
            SUM(Gunluk_Satis) as Toplam_Satis,
            AVG(Musteri_Puani) as Ort_Puan
        FROM satislar
        GROUP BY Sube
        ORDER BY Toplam_Satis DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} işlem, Ort.Fiyat: {row[2]:.2f} TL, "
              f"Toplam Satış: {row[3]} adet, Puan: {row[4]:.2f}")

    # 2. Kahve Türü Performans Analizi
    print("\n[Kahve Türü Performans Analizi]")
    cursor.execute("""
        SELECT
            Kahve_Turu,
            COUNT(*) as Satıs_Sayisi,
            AVG(Gunluk_Satis) as Ort_Gunluk_Satis,
            MAX(Fiyat) as Max_Fiyat,
            AVG(Musteri_Puani) as Ort_Puan
        FROM satislar
        GROUP BY Kahve_Turu
        ORDER BY Ort_Gunluk_Satis DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} kayıt, Günlük Ort: {row[2]:.1f} adet, "
              f"Max Fiyat: {row[3]:.2f} TL, Puan: {row[4]:.2f}")

    # 3. Barista Performans Karşılaştırması
    print("\n[Barista Performans Karşılaştırması]")
    cursor.execute("""
        SELECT
            Barista,
            COUNT(*) as Toplam_Siparis,
            AVG(Musteri_Puani) as Ort_Puan,
            SUM(Fiyat * Gunluk_Satis) as Toplam_Ciro
        FROM satislar
        GROUP BY Barista
        ORDER BY Ort_Puan DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} sipariş, Puan: {row[2]:.2f}, "
              f"Ciro: {row[3]:.2f} TL")

    # 4. Boyut Kategorisi Analizi
    print("\n[Boyut Kategorisi Analizi]")
    cursor.execute("""
        SELECT
            Boyut,
            COUNT(*) as Adet,
            AVG(Fiyat) as Ort_Fiyat,
            AVG(Kalori) as Ort_Kalori
        FROM satislar
        GROUP BY Boyut
        ORDER BY
            CASE Boyut
                WHEN 'Kucuk' THEN 1
                WHEN 'Orta' THEN 2
                WHEN 'Buyuk' THEN 3
            END
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} adet, Ort.Fiyat: {row[2]:.2f} TL, "
              f"Ort.Kalori: {row[3]:.0f}")

    # 5. Yüksek Performanslı Ürünler (Puan >= 4.5)
    print("\n[Premium Ürünler - Puan >= 4.5]")
    cursor.execute("""
        SELECT
            Kahve_Turu,
            Sube,
            Musteri_Puani,
            Gunluk_Satis
        FROM satislar
        WHERE Musteri_Puani >= 4.5
        ORDER BY Musteri_Puani DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} ({row[1]}): Puan {row[2]}, Satış: {row[3]} adet")

    # 6. Genel İstatistikler
    print("\n[Genel İstatistikler]")
    cursor.execute("""
        SELECT
            COUNT(*) as Toplam_Kayit,
            SUM(Gunluk_Satis) as Toplam_Satis_Adedi,
            AVG(Fiyat) as Ort_Fiyat,
            MIN(Fiyat) as Min_Fiyat,
            MAX(Fiyat) as Max_Fiyat,
            AVG(Musteri_Puani) as Genel_Ort_Puan
        FROM satislar
    """)
    row = cursor.fetchone()
    print(f"  Toplam Kayıt: {row[0]}")
    print(f"  Toplam Satış Adedi: {row[1]}")
    print(f"  Ortalama Fiyat: {row[2]:.2f} TL")
    print(f"  Fiyat Aralığı: {row[3]:.2f} - {row[4]:.2f} TL")
    print(f"  Genel Ortalama Puan: {row[5]:.2f}")

def numpy_analiz(cursor):
    print("\n--- 2. GÖREV: Numpy Analiz ---")
    cursor.execute("SELECT Fiyat, Gunluk_Satis FROM satislar")
    data = np.array(cursor.fetchall())

    fiyatlar = data[:, 0]
    satis_adetleri = data[:, 1]

    # Merkezi Eğilim Ölçüleri
    print("\n[Merkezi Eğilim Ölçüleri]")
    print(f"Ortalama Fiyat: {np.mean(fiyatlar):.2f} TL")
    print(f"Medyan Fiyat: {np.median(fiyatlar):.2f} TL")
    print(f"Ortalama Satış: {np.mean(satis_adetleri):.2f} Adet")
    print(f"Medyan Satış: {np.median(satis_adetleri):.2f} Adet")

    # Dağılım Ölçüleri
    print("\n[Dağılım Ölçüleri]")
    print(f"Fiyat Standart Sapma: {np.std(fiyatlar):.2f} TL")
    print(f"Fiyat Varyans: {np.var(fiyatlar):.2f}")
    print(f"Satış Standart Sapma: {np.std(satis_adetleri):.2f} Adet")
    print(f"Satış Varyans: {np.var(satis_adetleri):.2f}")

    # Minimum ve Maksimum Değerler
    print("\n[Min-Max Değerler]")
    print(f"Minimum Fiyat: {np.min(fiyatlar):.2f} TL")
    print(f"Maksimum Fiyat: {np.max(fiyatlar):.2f} TL")
    print(f"Fiyat Aralığı: {np.ptp(fiyatlar):.2f} TL")
    print(f"Minimum Satış: {np.min(satis_adetleri)} Adet")
    print(f"Maksimum Satış: {np.max(satis_adetleri)} Adet")
    print(f"Satış Aralığı: {np.ptp(satis_adetleri)} Adet")

    # Yüzdelik Dilimler (Percentiles)
    print("\n[Yüzdelik Dilimler]")
    print(f"Fiyat %25 Dilimi: {np.percentile(fiyatlar, 25):.2f} TL")
    print(f"Fiyat %75 Dilimi: {np.percentile(fiyatlar, 75):.2f} TL")
    print(f"Satış %25 Dilimi: {np.percentile(satis_adetleri, 25):.0f} Adet")
    print(f"Satış %75 Dilimi: {np.percentile(satis_adetleri, 75):.0f} Adet")

    # Toplam ve Korelasyon
    print("\n[Toplam ve İlişkisel Metrikler]")
    print(f"Toplam Satış Adedi: {np.sum(satis_adetleri):.0f} Adet")
    print(f"Tahmini Ciro: {np.sum(fiyatlar * satis_adetleri):.2f} TL")
    print(f"Fiyat-Satış Korelasyonu: {np.corrcoef(fiyatlar, satis_adetleri)[0, 1]:.3f}")

def pandas_analizleri(conn):
    print("\n--- 3, 4 ve 5. GÖREVLER: Pandas İleri Analiz ---")
    df = pd.read_sql_query("SELECT * FROM satislar", conn)

    # 1. Temel Feature Engineering
    df['Ciro'] = df['Fiyat'] * df['Gunluk_Satis']
    
    # 2. Zaman Serisi Analizi (Datetime Dönüşümü)
    print("\n[Zaman Serisi] Günlük Satış Trendi:")
    df['Tarih'] = pd.to_datetime(df['Tarih']) # Object tipinden Datetime'a çevir
    gunluk_trend = df.groupby('Tarih')['Ciro'].sum()
    print(gunluk_trend)

    # 3. Pivot Tablo (Çok Boyutlu Analiz)
    # Satırlarda Şube, Sütunlarda Kahve Türü, Değerlerde Ortalama Puan
    print("\n[Pivot Tablo] Şube ve Kahve Türüne Göre Puanlar:")
    pivot = df.pivot_table(values='Musteri_Puani', index='Sube', columns='Kahve_Turu', aggfunc='mean')
    print(pivot.fillna('-'))

    # 4. Korelasyon Analizi
    print("\n[İstatistik] Değişkenler Arası Korelasyon:")
    korelasyon = df[['Fiyat', 'Kalori', 'Musteri_Puani', 'Gunluk_Satis']].corr()
    print(korelasyon)

    # 5. Veri Segmentasyonu
    print("\n[Segmentasyon] Fiyat Kategorileri:")
    df['Fiyat_Segmenti'] = df['Fiyat'].apply(lambda x: 'Premium' if x >= 45 else 'Ekonomik')
    
    return df

def gorsellestirme(df):
    print("\n--- 6. GÖREV: Görselleştirme (Matplotlib vs Seaborn) ---")
    # Grafik ayarları
    plt.style.use('seaborn-v0_8-whitegrid') # Eski stil ismi değiştiği için güncelleme
    _, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3) # Grafikler arası boşluk
    
    # --- SENARYO 1: KATEGORİK VERİ GÖRSELLEŞTİRME (Şube Puanları) ---
    
    # 1. Matplotlib Yaklaşımı (Manuel Gruplama Gerekir)
    # Matplotlib otomatik toplama (aggregation) yapmaz, önce veriyi hazırlamalıyız.
    sube_puan = df.groupby('Sube')['Musteri_Puani'].mean()
    
    axs[0, 0].bar(sube_puan.index, sube_puan.values, color='#5DADE2', edgecolor='black')
    axs[0, 0].set_title('Matplotlib: Şube Puanları (Manuel Hazırlık)')
    axs[0, 0].set_xlabel('Şube')
    axs[0, 0].set_ylabel('Ortalama Puan')
    
    # 2. Seaborn Yaklaşımı (Otomatik Aggregation)
    # Seaborn "estimator" parametresi ile ortalamayı kendi hesaplar.
    sns.barplot(x='Sube', y='Musteri_Puani', data=df, ax=axs[0, 1], palette='viridis', errorbar=None)
    axs[0, 1].set_title('Seaborn: Şube Puanları (Otomatik Hesaplama)')
    axs[0, 1].set_ylim(0, 5.5)

    # --- SENARYO 2: ÇOK BOYUTLU İLİŞKİSEL GRAFİK (Scatter Plot) ---
    
    # 3. Matplotlib Yaklaşımı (Fiyat vs Satış)
    # Kategorilere göre renk vermek için döngü kurmak veya maplemek gerekir (Zorluk).
    axs[1, 0].scatter(df['Fiyat'], df['Gunluk_Satis'], color='gray', alpha=0.7)
    axs[1, 0].set_title('Matplotlib: Fiyat vs Satış (Tek Renk)')
    axs[1, 0].set_xlabel('Fiyat')
    axs[1, 0].set_ylabel('Günlük Satış')
    
    # 4. Seaborn Yaklaşımı (Boyut + Renk + Kategori)
    # Hue (Renk) ve Size (Boyut) parametreleri ile 4 boyutu tek grafikte gösteririz.
    sns.scatterplot(
        x='Fiyat', 
        y='Gunluk_Satis', 
        hue='Kahve_Turu',    # 3. Boyut: Renk (Kahve Türü)
        size='Kalori',       # 4. Boyut: Büyüklük (Kalori)
        sizes=(50, 300), 
        data=df, 
        ax=axs[1, 1], 
        palette='deep'
    )
    axs[1, 1].set_title('Seaborn: Fiyat vs Satış (Çok Boyutlu Analiz)')
    # Lejantı dışarı taşıyalım
    axs[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    print("Karşılaştırmalı grafikler oluşturuldu ve gösteriliyor...")
    plt.show()

# --- ANA PROGRAM AKIŞI ---
if __name__ == "__main__":
    conn = sqlite3.connect("kafe_verisi.db")
    cursor = conn.cursor()
    
    veritabani_hazirlik(cursor)
    conn.commit() 

    sql_temizlik(conn, cursor)
    sql_veri_analizi(cursor)
    numpy_analiz(cursor)
    df = pandas_analizleri(conn)
    gorsellestirme(df)

    conn.close()
    print("\nDers tamamlandı.")