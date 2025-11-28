# Pivot Tablo Nedir?

Pivot tablo, **uzun formatdaki** bir tabloyu **çapraz (matrix) formatına** dönüştüren bir araçtır. Excel'de çok popülerdir.

## Basit Örnekle Anlatalım

### ÖNCE: Normal Tablo (Uzun Format)
```
Sube       Kahve_Turu    Musteri_Puani
Kadikoy    Latte         4.5
Besiktas   Espresso      4.8
Kadikoy    Mocha         4.9
Besiktas   Latte         3.5
Uskudar    Cappuccino    4.6
```

### SONRA: Pivot Tablo (Çapraz Format)
```
              Latte  Espresso  Mocha  Cappuccino
Kadikoy        4.5      -      4.9       -
Besiktas       3.5     4.8      -        -
Uskudar         -       -       -       4.6
```

## Ne Değişti?

1. **Satırlar**: Şubeler (Kadikoy, Besiktas, Uskudar)
2. **Sütunlar**: Kahve türleri (Latte, Espresso, ...)
3. **Değerler**: Müşteri puanları
4. **Kesişim**: Her şube-kahve çiftinin puanı

---

## Pandas Pivot Kodu (Satır Satır)

```python
pivot = df.pivot_table(
    values='Musteri_Puani',    # Hangi sayıyı göstereceğiz?
    index='Sube',              # Satırlarda ne olacak?
    columns='Kahve_Turu',      # Sütunlarda ne olacak?
    aggfunc='mean'             # Aynı hücreye düşen değerleri nasıl birleştireceğiz?
)
```

### Her Argüman Ne İşe Yarar?

| Argüman | Açıklama | Örnek Değer |
|---------|----------|-------------|
| `values` | Hücrelerde hangi değer gösterilecek | `'Musteri_Puani'` |
| `index` | **Satır** başlıkları olacak sütun | `'Sube'` |
| `columns` | **Sütun** başlıkları olacak sütun | `'Kahve_Turu'` |
| `aggfunc` | Birden fazla değeri birleştirme yöntemi | `'mean'` (ortalama), `'sum'` (toplam), `'count'` (sayma) |

---

## Neden Kullanılır?

### 1. **Hızlı Karşılaştırma**
Normal tabloda: "Kadıköy'deki Latte puanı ne?" → 5 satır okuyup bulmak gerek
Pivot tabloda: Direkt satır-sütun kesişimine bakarsın

### 2. **Özet Çıkarmak**
```python
# Şubelerin toplam cirosu
pivot = df.pivot_table(
    values='Ciro',
    index='Sube',
    aggfunc='sum'  # Topla
)
```

### 3. **Excel'e Aktarmak**
İş dünyasında pivot tablolar Excel'de çok kullanıldığı için, pandas ile hazırlayıp Excel'e aktarabilirsin.

---

## Pratik Örnek: Farklı Agregasyonlar

```python
# Ortalama
df.pivot_table(values='Fiyat', index='Sube', columns='Kahve_Turu', aggfunc='mean')

# Toplam
df.pivot_table(values='Gunluk_Satis', index='Sube', columns='Kahve_Turu', aggfunc='sum')

# Kaç tane var?
df.pivot_table(values='Fiyat', index='Sube', columns='Kahve_Turu', aggfunc='count')
```

---

## Kafayı Karıştıran Nokta: `aggfunc` Neden Gerekli?

Bazen aynı hücreye birden fazla değer düşer:

```
Sube      Kahve_Turu  Musteri_Puani
Kadikoy   Latte       4.5
Kadikoy   Latte       4.0   ← İkinci Latte!
```

Pivot tablo oluştururken **Kadikoy-Latte hücresine ne yazılacak?**
- `aggfunc='mean'` → Ortalama: `(4.5 + 4.0) / 2 = 4.25`
- `aggfunc='sum'` → Toplam: `4.5 + 4.0 = 8.5`
- `aggfunc='max'` → Maksimum: `4.5`

---

## Özet

Pivot tablo = **Veriyi yeniden düzenleme aracı**

- Normal tablo: Satır satır liste
- Pivot tablo: Satır-sütun çapraz tablo (matris)

**Formül:**
```
Satırlar × Sütunlar = Değerler (agregasyonla birleştirilmiş)
```
