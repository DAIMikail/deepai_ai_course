# Kafe Veri Analizi - Detaylı Kod Açıklamaları

## İçindekiler
1. [Import'lar ve Kütüphaneler](#imports)
2. [Veritabanı Hazırlık](#veritabani-hazirlik)
3. [SQL Temizlik](#sql-temizlik)
4. [SQL Veri Analizi](#sql-veri-analizi)
5. [Numpy Analiz](#numpy-analiz)
6. [Pandas Analizleri](#pandas-analizleri)
7. [Görselleştirme](#gorsellestirme)

---

## 1. Import'lar ve Kütüphaneler {#imports}

```python
import sqlite3
```
- **sqlite3**: Python'da yerleşik gelen hafif veritabanı kütüphanesi
- Dosya tabanlı SQL veritabanı oluşturma ve sorgulama
- Sunucu gerektirmez, dosya olarak (.db) saklanır

```python
import numpy as np
```
- **numpy**: Sayısal hesaplama kütüphanesi
- Çok boyutlu diziler (arrays) ve matematiksel işlemler
- Vektörel işlemler (tüm diziyi tek seferde işleme)
- `np` aliası yaygın kullanım standardı

```python
import pandas as pd
```
- **pandas**: Veri analizi ve manipülasyonu için ana kütüphane
- DataFrame (tablo) yapısı sunar
- SQL benzeri gruplama, filtreleme, birleştirme işlemleri
- `pd` aliası yaygın kullanım standardı

```python
import matplotlib.pyplot as plt
```
- **matplotlib**: Temel grafik çizim kütüphanesi
- `pyplot` modülü: MATLAB benzeri arayüz
- `plt` aliası: Kısa ve yaygın kullanım

```python
import seaborn as sns
```
- **seaborn**: Matplotlib üzerine kurulu istatistiksel görselleştirme
- Daha estetik ve otomatik stil ayarları
- Kategorik veriler için gelişmiş grafikler
- `sns` aliası: "SeabornS" kısaltması

---

## 2. Veritabanı Hazırlık {#veritabani-hazirlik}

### Fonksiyon İmzası
```python
def veritabani_hazirlik(cursor):
```
- **cursor**: SQLite bağlantısının cursor objesi
- Cursor: Veritabanına komut gönderen "imleç"

### Tablo Silme
```python
cursor.execute("DROP TABLE IF EXISTS satislar")
```
- **DROP TABLE**: Tabloyu siler
- **IF EXISTS**: Tablo yoksa hata verme (güvenli silme)
- **satislar**: Tablo adı

### Tablo Oluşturma
```python
cursor.execute('''
CREATE TABLE satislar (
    ID INTEGER PRIMARY KEY,
    ...
)
''')
```

#### Sütun Tanımlamaları:

**ID INTEGER PRIMARY KEY**
- **INTEGER**: Tam sayı veri tipi
- **PRIMARY KEY**: Benzersiz kimlik, her satırı tanımlar, NULL olamaz

**Tarih TEXT**
- **TEXT**: Metin verisi (SQLite'da string için)
- Tarih bilgisini '2023-11-01' formatında saklar

**Kahve_Turu TEXT**
- İçecek ismini saklar (Latte, Espresso, vb.)

**Boyut TEXT**
- İçecek boyutu (Kucuk, Orta, Buyuk)

**Fiyat REAL**
- **REAL**: Ondalıklı sayı (float)
- Para birimi için uygun

**Kalori INTEGER**
- Tam sayı (kalori kesirli olmaz)

**Sube TEXT**
- Şube adı (Kadikoy, Besiktas, vb.)

**Musteri_Puani REAL**
- 1.0 - 5.0 arası ondalıklı değer

**Gunluk_Satis INTEGER**
- Satış adedi (tam sayı)

**Barista TEXT**
- Çalışan adı

### Veri Ekleme
```python
cursor.executemany('INSERT INTO satislar VALUES (?,?,?,?,?,?,?,?,?,?)', veriler)
```
- **executemany**: Birden fazla satırı tek seferde ekler
- **?**: Parametre yer tutucuları (SQL injection'a karşı güvenli)
- **veriler**: Liste içinde tuple'lar (her tuple = 1 satır)

---

## 3. SQL Temizlik {#sql-temizlik}

### Hatalı Fiyat Düzeltme
```python
cursor.execute("UPDATE satislar SET Fiyat = 55 WHERE Fiyat > 100")
```
- **UPDATE**: Mevcut satırları günceller
- **SET Fiyat = 55**: Fiyat sütununu 55 yap
- **WHERE Fiyat > 100**: Sadece 100'den büyük fiyatlarda (aykırı değer tespiti)

### NULL Değer Tamamlama
```python
cursor.execute("UPDATE satislar SET Kalori = 100 WHERE Kalori IS NULL")
```
- **IS NULL**: NULL değerleri bulur (= NULL çalışmaz!)
- Eksik kalorileri varsayılan değer (100) ile doldurur

### Commit
```python
conn.commit()
```
- **commit()**: Değişiklikleri kalıcı hale getirir
- Commit olmadan UPDATE/INSERT geçici kalır

---

## 4. SQL Veri Analizi {#sql-veri-analizi}

### 4.1 Şube Bazlı Analiz

#### Örnek Ham Veri:
Önce veritabanındaki verilere bakalım:

```
ID | Sube     | Fiyat | Gunluk_Satis | Musteri_Puani
---|----------|-------|--------------|---------------
1  | Kadikoy  | 45.0  | 120          | 4.5
2  | Besiktas | 30.0  | 200          | 4.8
3  | Kadikoy  | 55.0  | 90           | 4.9
4  | Besiktas | 40.0  | 110          | 3.5
5  | Kadikoy  | 35.0  | 150          | 4.2
6  | Uskudar  | 45.0  | 130          | 4.6
7  | Besiktas | 55.0  | 50           | 2.1
8  | Uskudar  | 30.0  | 180          | 4.0
9  | Kadikoy  | 48.0  | 95           | 4.7
10 | Besiktas | 42.0  | 100          | 4.3
```

#### SQL Sorgusu:
```sql
SELECT
    Sube,                           -- Şube adı
    COUNT(*) as Toplam_Islem,       -- Her şubede kaç satır var?
    AVG(Fiyat) as Ort_Fiyat,        -- O şubenin ortalama fiyatı
    SUM(Gunluk_Satis) as Toplam_Satis,  -- Toplam satış adedi
    AVG(Musteri_Puani) as Ort_Puan  -- Ortalama müşteri puanı
FROM satislar
GROUP BY Sube                       -- Şubelere göre grupla
ORDER BY Toplam_Satis DESC          -- En çok satan şube önce
```

#### Sorgu Adım Adım Nasıl Çalışır?

**Adım 1: GROUP BY Sube** - Veriyi 3 gruba ayırır:

**GRUP 1: Kadikoy**
```
ID=1:  Fiyat=45, Satis=120, Puan=4.5
ID=3:  Fiyat=55, Satis=90,  Puan=4.9
ID=5:  Fiyat=35, Satis=150, Puan=4.2
ID=9:  Fiyat=48, Satis=95,  Puan=4.7
```

**GRUP 2: Besiktas**
```
ID=2:  Fiyat=30, Satis=200, Puan=4.8
ID=4:  Fiyat=40, Satis=110, Puan=3.5
ID=7:  Fiyat=55, Satis=50,  Puan=2.1
ID=10: Fiyat=42, Satis=100, Puan=4.3
```

**GRUP 3: Uskudar**
```
ID=6:  Fiyat=45, Satis=130, Puan=4.6
ID=8:  Fiyat=30, Satis=180, Puan=4.0
```

**Adım 2: Agregasyon Fonksiyonları** - Her grup için hesaplama:

**Kadikoy Grubu:**
- COUNT(*) = 4 satır
- AVG(Fiyat) = (45+55+35+48) / 4 = 45.75
- SUM(Gunluk_Satis) = 120+90+150+95 = 455
- AVG(Musteri_Puani) = (4.5+4.9+4.2+4.7) / 4 = 4.575

**Besiktas Grubu:**
- COUNT(*) = 4 satır
- AVG(Fiyat) = (30+40+55+42) / 4 = 41.75
- SUM(Gunluk_Satis) = 200+110+50+100 = 460
- AVG(Musteri_Puani) = (4.8+3.5+2.1+4.3) / 4 = 3.675

**Uskudar Grubu:**
- COUNT(*) = 2 satır
- AVG(Fiyat) = (45+30) / 2 = 37.50
- SUM(Gunluk_Satis) = 130+180 = 310
- AVG(Musteri_Puani) = (4.6+4.0) / 2 = 4.30

**Adım 3: ORDER BY Toplam_Satis DESC** - Toplam satışa göre sırala:

#### Final Sonuç Tablosu:
```
Sube     | Toplam_Islem | Ort_Fiyat | Toplam_Satis | Ort_Puan
---------|--------------|-----------|--------------|----------
Besiktas | 4            | 41.75     | 460          | 3.68
Kadikoy  | 4            | 45.75     | 455          | 4.58
Uskudar  | 2            | 37.50     | 310          | 4.30
```

#### Python'da Sonuç Okuma:
```python
for row in cursor.fetchall():
    # row = ('Besiktas', 4, 41.75, 460, 3.68)
    print(f"{row[0]}: {row[1]} işlem, Ort.Fiyat: {row[2]:.2f} TL")
```

**Çıktı:**
```
Besiktas: 4 işlem, Ort.Fiyat: 41.75 TL, Toplam Satış: 460 adet, Puan: 3.68
Kadikoy: 4 işlem, Ort.Fiyat: 45.75 TL, Toplam Satış: 455 adet, Puan: 4.58
Uskudar: 2 işlem, Ort.Fiyat: 37.50 TL, Toplam Satış: 310 adet, Puan: 4.30
```

#### İş Anlamı:
- **Besiktas** en çok satış yapıyor (460 adet) ama en düşük puana sahip (3.68)
- **Kadikoy** daha yüksek fiyatlarla (45.75 TL) daha az satış yapıyor ama müşteri memnuniyeti yüksek (4.58)
- **Uskudar** en az işlem sayısına sahip (2 kayıt)

### 4.2 Kahve Türü Performans Analizi

#### Örnek Ham Veri:
```
ID | Kahve_Turu  | Gunluk_Satis | Fiyat | Musteri_Puani
---|-------------|--------------|-------|---------------
1  | Latte       | 120          | 45.0  | 4.5
2  | Espresso    | 200          | 30.0  | 4.8
3  | Mocha       | 90           | 55.0  | 4.9
4  | Latte       | 110          | 40.0  | 3.5
5  | Americano   | 150          | 35.0  | 4.2
6  | Cappuccino  | 130          | 45.0  | 4.6
7  | Latte       | 50           | 55.0  | 2.1
8  | Filtre      | 180          | 30.0  | 4.0
9  | Mocha       | 95           | 48.0  | 4.7
10 | Macchiato   | 100          | 42.0  | 4.3
```

#### SQL Sorgusu:
```sql
SELECT
    Kahve_Turu,
    COUNT(*) as Satis_Sayisi,           -- Kaç kez satıldı?
    AVG(Gunluk_Satis) as Ort_Gunluk_Satis,  -- Ortalama günlük satış
    MAX(Fiyat) as Max_Fiyat,            -- En pahalı versiyonu
    AVG(Musteri_Puani) as Ort_Puan      -- Ortalama memnuniyet
FROM satislar
GROUP BY Kahve_Turu
ORDER BY Ort_Gunluk_Satis DESC          -- En popüler kahve önce
```

#### Gruplama Örneği:

**Latte Grubu (3 kayıt):**
```
ID=1: Satis=120, Fiyat=45.0, Puan=4.5
ID=4: Satis=110, Fiyat=40.0, Puan=3.5
ID=7: Satis=50,  Fiyat=55.0, Puan=2.1

COUNT(*) = 3
AVG(Gunluk_Satis) = (120+110+50) / 3 = 93.33
MAX(Fiyat) = 55.0
AVG(Musteri_Puani) = (4.5+3.5+2.1) / 3 = 3.37
```

**Mocha Grubu (2 kayıt):**
```
ID=3: Satis=90,  Fiyat=55.0, Puan=4.9
ID=9: Satis=95,  Fiyat=48.0, Puan=4.7

COUNT(*) = 2
AVG(Gunluk_Satis) = (90+95) / 2 = 92.5
MAX(Fiyat) = 55.0
AVG(Musteri_Puani) = (4.9+4.7) / 2 = 4.8
```

#### Final Sonuç (ORDER BY Ort_Gunluk_Satis DESC):
```
Kahve_Turu  | Satis_Sayisi | Ort_Gunluk_Satis | Max_Fiyat | Ort_Puan
------------|--------------|------------------|-----------|----------
Espresso    | 1            | 200.0            | 30.0      | 4.8
Filtre      | 1            | 180.0            | 30.0      | 4.0
Americano   | 1            | 150.0            | 35.0      | 4.2
Cappuccino  | 1            | 130.0            | 45.0      | 4.6
Macchiato   | 1            | 100.0            | 42.0      | 4.3
Latte       | 3            | 93.3             | 55.0      | 3.37
Mocha       | 2            | 92.5             | 55.0      | 4.8
```

#### İş Anlamı:
- **Espresso** en popüler (200 günlük satış) ve ucuz (30 TL)
- **Latte** 3 kez satılmış ama düşük ortalama puan (3.37) - kalite problemi?
- **Mocha** yüksek fiyata (55 TL) rağmen mükemmel puan (4.8)

---

### 4.3 CASE Statement - Boyut Sıralama

#### Problem:
Alfabetik sıralama yanlış sonuç verir:
```
Buyuk    (B harfi)
Kucuk    (K harfi)
Orta     (O harfi)
```

Mantıksal sıralama istiyoruz: **Küçük → Orta → Büyük**

#### Çözüm: CASE ile Özel Sıralama

```sql
SELECT
    Boyut,
    COUNT(*) as Adet,
    AVG(Fiyat) as Ort_Fiyat,
    AVG(Kalori) as Ort_Kalori
FROM satislar
GROUP BY Boyut
ORDER BY
    CASE Boyut
        WHEN 'Kucuk' THEN 1    -- Küçük = sıra 1
        WHEN 'Orta' THEN 2     -- Orta = sıra 2
        WHEN 'Buyuk' THEN 3    -- Büyük = sıra 3
    END
```

#### CASE Nasıl Çalışır?

Ham sonuç (sırasız):
```
Boyut  | Adet | Ort_Fiyat | Ort_Kalori
-------|------|-----------|------------
Orta   | 5    | 38.0      | 124
Buyuk  | 2    | 51.5      | 365
Kucuk  | 3    | 37.3      | 115
```

CASE sütunu eklendi (görünmez):
```
Boyut  | Adet | Ort_Fiyat | Ort_Kalori | CASE_DEGERI
-------|------|-----------|------------|-------------
Orta   | 5    | 38.0      | 124        | 2
Buyuk  | 2    | 51.5      | 365        | 3
Kucuk  | 3    | 37.3      | 115        | 1
```

ORDER BY CASE_DEGERI sonrası:
```
Boyut  | Adet | Ort_Fiyat | Ort_Kalori
-------|------|-----------|------------
Kucuk  | 3    | 37.3      | 115
Orta   | 5    | 38.0      | 124
Buyuk  | 2    | 51.5      | 365
```

#### CASE Kullanım Senaryoları:
1. **Özel Sıralama**: İş mantığına göre sıralama
2. **Koşullu Değer Atama**:
   ```sql
   CASE
       WHEN Fiyat < 40 THEN 'Ekonomik'
       WHEN Fiyat < 50 THEN 'Orta'
       ELSE 'Premium'
   END as Kategori
   ```
3. **Pivot Benzeri Dönüşüm**:
   ```sql
   SUM(CASE WHEN Boyut = 'Kucuk' THEN Satis ELSE 0 END) as Kucuk_Satis
   ```

---

### 4.4 WHERE Filtreleme - Premium Ürünler

#### Örnek Ham Veri:
```
ID | Kahve_Turu  | Sube     | Musteri_Puani | Gunluk_Satis
---|-------------|----------|---------------|---------------
1  | Latte       | Kadikoy  | 4.5           | 120
2  | Espresso    | Besiktas | 4.8           | 200  ✓
3  | Mocha       | Kadikoy  | 4.9           | 90   ✓
4  | Latte       | Besiktas | 3.5           | 110
5  | Americano   | Kadikoy  | 4.2           | 150
6  | Cappuccino  | Uskudar  | 4.6           | 130  ✓
7  | Latte       | Besiktas | 2.1           | 50
8  | Filtre      | Uskudar  | 4.0           | 180
9  | Mocha       | Kadikoy  | 4.7           | 95   ✓
10 | Macchiato   | Besiktas | 4.3           | 100
```

#### SQL Sorgusu:
```sql
SELECT
    Kahve_Turu,
    Sube,
    Musteri_Puani,
    Gunluk_Satis
FROM satislar
WHERE Musteri_Puani >= 4.5    -- SADECE yüksek puanlı ürünler
ORDER BY Musteri_Puani DESC   -- En yüksek puan önce
```

#### WHERE İşleme Sırası:
```
1. FROM satislar           → 10 satır yüklenir
2. WHERE Musteri_Puani >= 4.5  → 4 satır kalır (ID: 2,3,6,9)
3. SELECT sütunları seç    → İstenilen sütunları al
4. ORDER BY sırala         → Puana göre sırala
```

#### Final Sonuç:
```
Kahve_Turu  | Sube     | Musteri_Puani | Gunluk_Satis
------------|----------|---------------|---------------
Mocha       | Kadikoy  | 4.9           | 90
Espresso    | Besiktas | 4.8           | 200
Mocha       | Kadikoy  | 4.7           | 95
Cappuccino  | Uskudar  | 4.6           | 130
```

#### WHERE vs HAVING Farkı:

**WHERE**: GROUP BY'dan ÖNCE (ham satırları filtreler)
```sql
SELECT Sube, COUNT(*)
FROM satislar
WHERE Fiyat > 40          -- Önce pahalı ürünleri filtrele
GROUP BY Sube             -- Sonra grupla
```

**HAVING**: GROUP BY'dan SONRA (grupları filtreler)
```sql
SELECT Sube, COUNT(*) as Adet
FROM satislar
GROUP BY Sube             -- Önce grupla
HAVING COUNT(*) > 2       -- Sonra 2'den fazla işlem olan şubeleri al
```

#### Örnek Karşılaştırma:

**WHERE örneği:**
```sql
-- "Pahalı ürünler hangi şubelerde satılıyor?"
SELECT Sube, COUNT(*)
FROM satislar
WHERE Fiyat >= 45         -- 5 satır kalır
GROUP BY Sube
```
Sonuç:
```
Sube     | COUNT(*)
---------|----------
Kadikoy  | 3
Besiktas | 1
Uskudar  | 1
```

**HAVING örneği:**
```sql
-- "Hangi şubelerde 3'ten fazla işlem var?"
SELECT Sube, COUNT(*) as Adet
FROM satislar
GROUP BY Sube
HAVING COUNT(*) > 3       -- Grupları filtrele
```
Sonuç:
```
Sube     | Adet
---------|------
Kadikoy  | 4
Besiktas | 4
-- Uskudar yok (sadece 2 işlemi var)
```

---

### 4.5 Barista Performans ve Hesaplanmış Alan

#### SQL Sorgusu:
```sql
SELECT
    Barista,
    COUNT(*) as Toplam_Siparis,
    AVG(Musteri_Puani) as Ort_Puan,
    SUM(Fiyat * Gunluk_Satis) as Toplam_Ciro    -- Hesaplanmış alan
FROM satislar
GROUP BY Barista
ORDER BY Ort_Puan DESC
```

#### Örnek Ham Veri:
```
ID | Barista | Fiyat | Gunluk_Satis | Musteri_Puani
---|---------|-------|--------------|---------------
1  | Ali     | 45.0  | 120          | 4.5
2  | Ayse    | 30.0  | 200          | 4.8
3  | Ali     | 55.0  | 90           | 4.9
4  | Mehmet  | 40.0  | 110          | 3.5
5  | Zeynep  | 35.0  | 150          | 4.2
6  | Can     | 45.0  | 130          | 4.6
7  | Ayse    | 55.0  | 50           | 2.1
8  | Can     | 30.0  | 180          | 4.0
9  | Zeynep  | 48.0  | 95           | 4.7
10 | Mehmet  | 42.0  | 100          | 4.3
```

#### Hesaplanmış Alan Örneği (Ali için):

**Ali'nin kayıtları:**
```
ID=1: Fiyat=45.0, Satis=120  →  Ciro = 45.0 * 120 = 5400
ID=3: Fiyat=55.0, Satis=90   →  Ciro = 55.0 * 90  = 4950

COUNT(*) = 2 sipariş
AVG(Musteri_Puani) = (4.5 + 4.9) / 2 = 4.7
SUM(Fiyat * Gunluk_Satis) = 5400 + 4950 = 10350 TL
```

#### Final Sonuç:
```
Barista | Toplam_Siparis | Ort_Puan | Toplam_Ciro
--------|----------------|----------|-------------
Ali     | 2              | 4.70     | 10350.00
Zeynep  | 2              | 4.45     | 9810.00
Can     | 2              | 4.30     | 11250.00
Mehmet  | 2              | 3.90     | 8600.00
Ayse    | 2              | 3.45     | 8750.00
```

#### SQL İşlem Önceliği:
```
1. FROM satislar                           → 10 satır
2. GROUP BY Barista                        → 5 grup
3. SELECT hesaplamaları:
   - COUNT(*) her grup için satır sayısı
   - AVG(Musteri_Puani) her grup için ortalama
   - SUM(Fiyat * Gunluk_Satis) ← ÖNEMLİ: Önce her satırda çarpım, sonra toplam
4. ORDER BY Ort_Puan DESC                  → Sıralama
```

#### Hesaplanmış Alan Detayı:

**Yanlış yaklaşım:**
```sql
AVG(Fiyat) * AVG(Gunluk_Satis)    -- YANLIŞ! Ortalama fiyat * ortalama satış
```
Ali için: (45+55)/2 * (120+90)/2 = 50 * 105 = 5250 TL ❌

**Doğru yaklaşım:**
```sql
SUM(Fiyat * Gunluk_Satis)         -- DOĞRU! Her satırda çarp, sonra topla
```
Ali için: (45*120) + (55*90) = 5400 + 4950 = 10350 TL ✓

#### İş Anlamı:
- **Ali**: En yüksek puan (4.7) ama orta ciro (10350 TL)
- **Can**: Orta puan (4.3) ama en yüksek ciro (11250 TL) - yüksek satış hacmi
- **Ayse**: En düşük puan (3.45) - eğitim gerekebilir

---

### 4.6 SQL Sorgu Çalıştırma Sırası

SQL sorgularının **yazılış sırası** ile **çalıştırma sırası** farklıdır!

#### Yazılış Sırası:
```sql
SELECT sütunlar
FROM tablo
WHERE koşul
GROUP BY grup
HAVING grup_koşulu
ORDER BY sıralama
LIMIT sayı
```

#### Çalıştırma Sırası:
```
1. FROM      → Tabloyu yükle
2. WHERE     → Satırları filtrele
3. GROUP BY  → Grupla
4. HAVING    → Grupları filtrele
5. SELECT    → Sütunları seç ve hesapla
6. ORDER BY  → Sırala
7. LIMIT     → Sayıyı kısıtla
```

#### Pratik Örnek:

**Sorgu:**
```sql
SELECT Sube, COUNT(*) as Adet, AVG(Fiyat) as Ort_Fiyat
FROM satislar
WHERE Gunluk_Satis > 100
GROUP BY Sube
HAVING AVG(Fiyat) > 40
ORDER BY Adet DESC
LIMIT 2
```

**Adım Adım:**
```
1. FROM satislar
   → 10 satır yüklendi

2. WHERE Gunluk_Satis > 100
   → 6 satır kaldı (ID: 1,2,4,5,6,8)

3. GROUP BY Sube
   → 3 grup oluştu (Kadikoy: 3, Besiktas: 2, Uskudar: 1)

4. HAVING AVG(Fiyat) > 40
   → Kadikoy grubu elendi (ort=38.3), 2 grup kaldı

5. SELECT Sube, COUNT(*), AVG(Fiyat)
   → Sütunlar hesaplandı

6. ORDER BY Adet DESC
   → Besiktas önce (2 adet), Uskudar sonra (1 adet)

7. LIMIT 2
   → Her ikisi de gösteriliyor (zaten 2 grup var)
```

**Final Sonuç:**
```
Sube     | Adet | Ort_Fiyat
---------|------|----------
Besiktas | 2    | 45.0
Uskudar  | 1    | 45.0
```

#### Sık Yapılan Hatalar:

**HATA 1: SELECT'te olmayan sütunu ORDER BY'da kullanma**
```sql
SELECT Sube, AVG(Fiyat)
FROM satislar
GROUP BY Sube
ORDER BY Gunluk_Satis    -- HATA! Gunluk_Satis SELECT'te yok ve gruplanmış
```

**Çözüm:** ORDER BY için sütunu SELECT'e ekle
```sql
SELECT Sube, AVG(Fiyat), MAX(Gunluk_Satis)
FROM satislar
GROUP BY Sube
ORDER BY MAX(Gunluk_Satis)
```

**HATA 2: WHERE'de alias kullanma**
```sql
SELECT Fiyat * 2 as Cift_Fiyat
FROM satislar
WHERE Cift_Fiyat > 100    -- HATA! WHERE, SELECT'ten önce çalışır
```

**Çözüm:** Ham sütunu kullan
```sql
SELECT Fiyat * 2 as Cift_Fiyat
FROM satislar
WHERE Fiyat * 2 > 100     -- DOĞRU
```

**HATA 3: HAVING yerine WHERE kullanma**
```sql
SELECT Sube, COUNT(*)
FROM satislar
WHERE COUNT(*) > 2        -- HATA! WHERE gruplamadan önce çalışır
GROUP BY Sube
```

**Çözüm:** Grup koşulu için HAVING
```sql
SELECT Sube, COUNT(*)
FROM satislar
GROUP BY Sube
HAVING COUNT(*) > 2       -- DOĞRU
```

---

## 5. Numpy Analiz {#numpy-analiz}

### Veri Çekme ve Dönüştürme

```python
cursor.execute("SELECT Fiyat, Gunluk_Satis FROM satislar")
data = np.array(cursor.fetchall())
```
- **np.array()**: Python listesini numpy dizisine çevirir
- **fetchall()**: [(45.0, 120), (30.0, 200), ...]
- **data**: 2 boyutlu numpy array (satır, sütun)

### Sütun Ayırma

```python
fiyatlar = data[:, 0]
satis_adetleri = data[:, 1]
```
- **[:,  0]**: "Tüm satırlar (:), ilk sütun (0)"
- **[:,  1]**: "Tüm satırlar (:), ikinci sütun (1)"
- NumPy slicing (dilimleme) sözdizimi

### 5.1 Merkezi Eğilim Ölçüleri

**np.mean()**
```python
np.mean(fiyatlar)
```
- **Ortalama**: Tüm değerlerin toplamı / eleman sayısı
- `mean([10, 20, 30]) = 20`

**np.median()**
```python
np.median(fiyatlar)
```
- **Medyan**: Sıralanmış verinin ortancası
- Aykırı değerlerden etkilenmez
- `median([10, 20, 100]) = 20` (ortalama 43.33 olurdu)

### 5.2 Dağılım Ölçüleri

**np.std()**
```python
np.std(fiyatlar)
```
- **Standart Sapma**: Verinin ortalamadan ne kadar dağıldığı
- Küçük std = veriler birbirine yakın
- Büyük std = veriler dağınık

**np.var()**
```python
np.var(fiyatlar)
```
- **Varyans**: Standart sapmanın karesi
- `var = std²`
- İstatistiksel hesaplamalarda kullanılır

### 5.3 Min-Max Değerler

**np.min() / np.max()**
```python
np.min(fiyatlar)
np.max(fiyatlar)
```
- En küçük / En büyük değer

**np.ptp()**
```python
np.ptp(fiyatlar)
```
- **Peak to Peak**: Tepe noktası aralığı
- `ptp = max - min`
- Veri aralığını gösterir

### 5.4 Yüzdelik Dilimler

**np.percentile()**
```python
np.percentile(fiyatlar, 25)
np.percentile(fiyatlar, 75)
```
- **25. persentil (Q1)**: Verinin %25'i bundan küçük
- **75. persentil (Q3)**: Verinin %75'i bundan küçük
- **IQR (Inter-Quartile Range)**: Q3 - Q1
- Box plot'larda kullanılır

### 5.5 Korelasyon

**np.corrcoef()**
```python
np.corrcoef(fiyatlar, satis_adetleri)[0, 1]
```
- **Korelasyon Katsayısı**: İki değişken arasındaki ilişki
- **-1**: Tam negatif korelasyon (biri artarken diğeri azalır)
- **0**: İlişki yok
- **+1**: Tam pozitif korelasyon (birlikte artarlar)
- **[0, 1]**: Matrisin (0,1) elemanı = fiyat-satış korelasyonu

---

## 6. Pandas Analizleri {#pandas-analizleri}

### SQL'den DataFrame'e

```python
df = pd.read_sql_query("SELECT * FROM satislar", conn)
```
- **read_sql_query()**: SQL sorgusunu çalıştırıp DataFrame döndürür
- **conn**: SQLite bağlantı objesi (cursor değil!)
- **df**: DataFrame (tablo yapısı)

### 6.1 Hesaplanmış Sütun

```python
df['Ciro'] = df['Fiyat'] * df['Gunluk_Satis']
```
- Yeni sütun oluşturur
- Vektörel işlem (tüm satırlarda aynı anda)
- SQL'deki `SELECT Fiyat * Gunluk_Satis AS Ciro` ile eşdeğer

### 6.2 Datetime Dönüşümü

```python
df['Tarih'] = pd.to_datetime(df['Tarih'])
```
- **to_datetime()**: String'i datetime objesine çevirir
- '2023-11-01' → datetime(2023, 11, 1)
- Zaman serisi analizleri için gerekli
- Tarih aritmetiği yapabilir (fark alma, sıralama)

### 6.3 GroupBy

```python
gunluk_trend = df.groupby('Tarih')['Ciro'].sum()
```
- **groupby('Tarih')**: Tarihe göre grupla
- **['Ciro']**: Sadece Ciro sütununu seç
- **.sum()**: Her grup için topla
- SQL'deki `GROUP BY Tarih` ile aynı mantık

### 6.4 Pivot Table

```python
pivot = df.pivot_table(
    values='Musteri_Puani',  # Gösterilecek değer
    index='Sube',            # Satırlar
    columns='Kahve_Turu',    # Sütunlar
    aggfunc='mean'           # Toplama fonksiyonu
)
```
- **Pivot Table**: Excel pivot table'ı gibi
- Çok boyutlu veriyi 2D tabloya çevirir
- **aggfunc**: Aggregation function (mean, sum, count, vb.)

**Sonuç formatı:**
```
Kahve_Turu  Americano  Cappuccino  Espresso  ...
Sube
Besiktas        -         -         4.8
Kadikoy        4.2        -         -
```

### 6.5 Korelasyon Matrisi

```python
korelasyon = df[['Fiyat', 'Kalori', 'Musteri_Puani', 'Gunluk_Satis']].corr()
```
- **[[liste]]**: Birden fazla sütun seçimi
- **.corr()**: Korelasyon matrisi hesaplar
- Her sütun çifti arasındaki korelasyonu gösterir

**Sonuç formatı:**
```
                Fiyat  Kalori  Musteri_Puani  Gunluk_Satis
Fiyat            1.00    0.75           0.23         -0.45
Kalori           0.75    1.00           0.10         -0.20
...
```

### 6.6 Lambda Fonksiyonu

```python
df['Fiyat_Segmenti'] = df['Fiyat'].apply(lambda x: 'Premium' if x >= 45 else 'Ekonomik')
```
- **apply()**: Her satıra fonksiyon uygular
- **lambda x**: Anonim fonksiyon (tek satırlık fonksiyon)
- **x**: Her bir fiyat değeri
- **if-else**: Koşullu atama (ternary operator)

**Eşdeğer normal fonksiyon:**
```python
def fiyat_kategorisi(x):
    if x >= 45:
        return 'Premium'
    else:
        return 'Ekonomik'

df['Fiyat_Segmenti'] = df['Fiyat'].apply(fiyat_kategorisi)
```

---

## 7. Görselleştirme {#gorsellestirme}

### 7.1 Stil ve Figure Oluşturma

```python
plt.style.use('seaborn-v0_8-whitegrid')
```
- **style.use()**: Grafik stilini belirler
- **seaborn-v0_8-whitegrid**: Beyaz arka plan + grid çizgileri

```python
_, axs = plt.subplots(2, 2, figsize=(14, 10))
```
- **subplots()**: Çoklu grafik oluşturur
- **2, 2**: 2 satır, 2 sütun = 4 grafik
- **figsize=(14, 10)**: Genişlik 14 inch, yükseklik 10 inch
- **_**: Figure objesini kullanmıyoruz (underscore = görmezden gel)
- **axs**: Axes (eksen) dizisi - her grafik bir axes

```python
plt.subplots_adjust(hspace=0.4, wspace=0.3)
```
- **hspace**: Yatay (height) boşluk
- **wspace**: Dikey (width) boşluk
- Grafiklerin üst üste binmemesi için

### 7.2 Matplotlib Bar Chart

```python
sube_puan = df.groupby('Sube')['Musteri_Puani'].mean()
```
- Manuel veri hazırlığı (Matplotlib otomatik toplama yapmaz)

```python
axs[0, 0].bar(sube_puan.index, sube_puan.values, color='#5DADE2', edgecolor='black')
```
- **axs[0, 0]**: İlk satır, ilk sütun (sol üst grafik)
- **.bar()**: Bar chart (sütun grafiği)
- **sube_puan.index**: X ekseni (şube isimleri)
- **sube_puan.values**: Y ekseni (puan değerleri)
- **color='#5DADE2'**: Hex renk kodu (mavi)
- **edgecolor='black'**: Bar kenar rengi

```python
axs[0, 0].set_title('Matplotlib: Şube Puanları (Manuel Hazırlık)')
axs[0, 0].set_xlabel('Şube')
axs[0, 0].set_ylabel('Ortalama Puan')
```
- **set_title()**: Başlık
- **set_xlabel()**: X ekseni etiketi
- **set_ylabel()**: Y ekseni etiketi

### 7.3 Seaborn Bar Chart

```python
sns.barplot(
    x='Sube',
    y='Musteri_Puani',
    data=df,
    ax=axs[0, 1],
    palette='viridis',
    errorbar=None
)
```
- **x, y**: Sütun isimleri (manuel hazırlık gerekmez!)
- **data=df**: Kaynak DataFrame
- **ax=axs[0, 1]**: Hangi alt grafiğe çizilecek (sağ üst)
- **palette='viridis'**: Renk paleti (gradient mavi-sarı-yeşil)
- **errorbar=None**: Hata çubuklarını kaldır (varsayılan: güven aralığı)

### 7.4 Matplotlib Scatter Plot

```python
axs[1, 0].scatter(df['Fiyat'], df['Gunluk_Satis'], color='gray', alpha=0.7)
```
- **scatter()**: Nokta grafiği
- **alpha=0.7**: Şeffaflık (0.0=tamamen şeffaf, 1.0=opak)
- Tek renk (kategorilere göre renklendirme yapamaz)

### 7.5 Seaborn Scatter Plot (Çok Boyutlu)

```python
sns.scatterplot(
    x='Fiyat',
    y='Gunluk_Satis',
    hue='Kahve_Turu',     # 3. Boyut: Renk
    size='Kalori',        # 4. Boyut: Nokta büyüklüğü
    sizes=(50, 300),      # Min-max nokta boyutu
    data=df,
    ax=axs[1, 1],
    palette='deep'
)
```
- **hue**: Kategorik değişkene göre renklendirme
- **size**: Sayısal değişkene göre boyutlandırma
- **sizes=(50, 300)**: Minimum 50, maksimum 300 piksel
- **palette='deep'**: Derin renkler (koyu tonlar)

**Sonuç:** Tek grafikte 4 boyut gösterilir:
1. X ekseni: Fiyat
2. Y ekseni: Satış
3. Renk: Kahve Türü
4. Büyüklük: Kalori

### 7.6 Lejant Konumlandırma

```python
axs[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
```
- **legend()**: Açıklama kutusu
- **bbox_to_anchor=(1.05, 1)**: Lejantın konumu (grafiğin sağ üstü)
  - 1.05: Grafik genişliğinin %105'i (sağa taşar)
  - 1: Grafik yüksekliğinin %100'ü (üst)
- **loc='upper left'**: Lejantın kendi içindeki hizalama
- **borderaxespad=0**: Kenarlık boşluğu

### 7.7 Grafiği Gösterme

```python
plt.show()
```
- **show()**: Pencere açar ve grafikleri gösterir
- Programı bekletir (pencere kapanana kadar)
- Jupyter'da gereksiz (otomatik gösterir)

---

## Özet: Kütüphane Karşılaştırması

### SQL
- **Avantajları**: Hızlı, veritabanı seviyesinde filtreleme, büyük veriler için ideal
- **Dezavantajları**: Sınırlı istatistiksel fonksiyonlar, karmaşık analizler için yetersiz

### NumPy
- **Avantajları**: Çok hızlı matematiksel işlemler, bellek verimli
- **Dezavantajları**: Sadece sayısal veri, tablo yapısı yok

### Pandas
- **Avantajları**: SQL + NumPy = Tablo yapısı + İstatistik, en esnek
- **Dezavantajları**: Büyük verilerde yavaş, bellek tüketimi fazla

### Matplotlib
- **Avantajları**: Tam kontrol, her detayı ayarlayabilirsin
- **Dezavantajları**: Çok kod yazmak gerekir, manuel veri hazırlığı

### Seaborn
- **Avantajları**: Az kod, otomatik estetik, istatistiksel grafikler
- **Dezavantajları**: Daha az kontrol, bazı özelleştirmeler zor

---

## Temel Veri Analizi İş Akışı

```
1. SQL          → Veriyi filtrele, grupla, ön işle (hızlı)
2. NumPy        → Basit istatistikler (ortalama, std)
3. Pandas       → Keşifsel analiz, dönüşümler
4. Matplotlib   → Özel grafikler (tam kontrol)
5. Seaborn      → Hızlı istatistiksel görselleştirme
```

Bu dosya, tüm kullanılan fonksiyonların, parametrelerin ve SQL komutlarının detaylı açıklamalarını içerir.
