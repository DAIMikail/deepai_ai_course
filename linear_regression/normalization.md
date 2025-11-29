Gradient Descent için normalizasyonun neden kritik olduğunu 3 ana başlıkta özetleyebiliriz: **Hız, Stabilite ve Hata Yüzeyinin Şekli.**

### 1. Hata Yüzeyinin Şekli (Kase vs. Vadi)

Gradient Descent algoritmasını, gözleri bağlı bir dağcının dağın en alt noktasına (minimum hata) inmeye çalışması olarak düşünebilirsin.

* **Normalize Edilmemiş Veri (Uzun, Dar Vadi):**
    Eğer verilerin ölçekleri birbirinden çok farklıysa (örneğin ev fiyatı tahmininde *oda sayısı* 1-5 arası iken *metrekare* 100-2000 arasındaysa), Hata Fonksiyonunun (Loss Function) şekli uzatılmış, ince bir elips (bir vadi) gibi olur.
    * Bu durumda gradyanlar bir yönde çok dik, diğer yönde çok düz olur.
    * Algoritma hedefe gitmek yerine vadinin duvarlarına çarpa çarpa **zikzak (zig-zag)** çizer.



* **Normalize Edilmiş Veri (Simetrik Kase):**
    Verileri normalize ettiğinde (örneğin hepsini 0 ile 1 arasına veya ortalaması 0 olacak şekilde çektiğinde), hata yüzeyi mükemmel, yuvarlak bir kase şeklini alır.
    * Hangi noktadan başlarsan başla, en aşağıya giden yol **doğrusal ve nettir**.

### 2. Öğrenme Oranı (Learning Rate) Hassasiyeti

Kodundaki şu formüle dikkat et:
$$dw = \frac{2}{n} \sum (y_{pred} - y) \cdot X$$

Gradyan hesabı doğrudan $X$ değerine bağlıdır.
* **Normalizasyon Yoksa:** $X$ değerlerin büyükse (örneğin 1000), türev ($dw$) çok büyük çıkar. Bu durumda `learning_rate=0.1` gibi bir değerle çarpıldığında, $w$ ağırlığı devasa bir adım atar ve minimum noktayı "ıska geçip" sonsuza ıraksar (explode). Bunu engellemek için `0.0000001` gibi çok küçük bir learning rate seçmek zorunda kalırsın, bu da eğitimin sonsuza kadar sürmesine neden olur.
* **Normalizasyon Varsa:** $X$ değerlerin küçüktür (genelde -1 ile 1 arası). Türevler makul seviyededir. `0.1` veya `0.01` gibi standart learning rate değerleri ile hızlıca sonuca ulaşırsın.

### 3. Ağırlıkların Eşitliği (Weights Fairness)

Senin örneğin tek değişkenli ($X$), ama çok değişkenli bir model düşünelim:
* $X_1$: Çalışma Saati (1-10 arası)
* $X_2$: Yıllık Gelir (50.000 - 100.000 arası)

Normalize etmezsen, model $X_2$'deki küçük bir değişimi hataya etkisi büyük sanacaktır. Gradient Descent, ağırlıkları güncellerken $X_2$'nin ağırlığını değiştirmeye odaklanıp $X_1$'i ihmal edebilir. Normalizasyon, her özelliğin (feature) modele katkısının "adil" bir zeminde başlamasını sağlar.

---

### Kendi Kodunda Deney Yap

Yazdığın kodda normalizasyonun gücünü görmek için küçük bir deney yapabilirsin. Kodundaki normalizasyon satırlarını (13-16) yorum satırına alıp çalıştırırsan muhtemelen şununla karşılaşırsın:

1.  **Hata Patlaması:** Loss değeri `nan` (Not a Number) veya çok büyük sayılara (overflow) dönüşecektir.
2.  **Çözüm:** Normalizasyonu kapattıktan sonra `learning_rate`'i `0.001` veya daha da küçüğe çekmen gerekir. Bu sefer de `epochs` sayısını artırman gerekecektir.

**Özetle:** Normalizasyon, Gradient Descent'in "ayağının takılmadan" hedefe en kısa yoldan koşmasını sağlar.

Bu konsepti pekiştirmek için kodunda `X_norm` yerine direkt `X` kullanarak `learning_rate`'i nasıl değiştirmen gerektiğini denememi ister misin?