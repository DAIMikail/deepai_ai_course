model: gpt-4.1-nano-2025-04-14
aktarılacak dosyalar: employees.md flow.md product_catalogs.md

# Döküman prompt'u
bir elektronik firmasına ait müşteri hizmetleri ai'ın müşteri ile konuşmasını simule edeceğim. Bunun için dökümanlara ihtiyacım var. Bu dökümanları teknik bir dil kullanarak ve md formatında oluştur. Başlık, altbaşlık dikkat et. Ticari dökümanlara uygun bir formatta hazırla.
  product_catalogs.md -> 30 farklı ürünün adı, kullanma talimatları, hangi durumda garantiye dahil hangi durumda değil
  cargo.md -> 5 adet kargo firmasının iade prosedürü ve ürünlerin kargo politikaları
  employees.md -> çalışanların bir listesi hangi çalışan hangi ürün ile ilgilenir? iletişim bilgileri neler vs

# context window anlat.
Context window, bir LLM'in tek seferde isleyebilecegi maksimum token sayisidir. Bu pencere icerisine sistem promptu, kullanici mesaji, belgeler (RAG) ve konusma gecmisi dahil edilir. Ornegin GPT-4.1-nano 1M token context window'a sahiptir. Token limiti asildiginda eski bilgiler kesilir veya hata alinir. Bu nedenle token sayisini takip etmek ve context'i verimli kullanmak onemlidir.

# uv init ve kütüphanelerin kurulumu
uv init .
uv venv
uv add chromadb openai tiktoken fastapi

# tokenizer
tiktoken kullanarak gpt4.1-nano yani o200k_base ile bir src/tokenizer.py dosyası oluştur. Burda bir sınıf bulunsun sınıfın amacı verilen belgenin veya metnin tokenize edilerek token
  sayısının belirlenmesidir.

# mevcut belgelerin token sayılarının gösterilmesi.
token_count_data.py adında bir dosya oluştur ve @../employees.md @../product_catalogs.md @../cargo.md dosyalarının token sayılarını yazdır.

# mevcut belgelerle ve openai ile soru cevap
## Geçen haftaki openai client'i örnek olarak al.
@../openai_ex.py 'i kullanarak src/model.py adında yeni bir dosya oluştur. Bu dosyada OpenAIClient sınıfı bulunsun openai'a istekte bulunmak için bir fonksiyon barındırsın.    
  model olarak gpt4.1-nano'u kullan. 

# basit bir chatbot arayüzü ve backend
- html css js ve tailwind kullanarak basit bir chatbot arayüzü oluştur. Dosya adı frontend.html olsun.
- fastapi ile chat endpointine sahip ve cors ayarları yapılmış basit bir backend oluştur. Dosya adı backend.py olsun.
- backend ile frontend'i bağlayacak düzenlemeleri frontend.html içerisinde script bölümünde yap.
## TEST ET

# context init
src/context.py adında bir dosya oluştur. Bu dosyanın içinde ContextManager sınıfı bulunsun. Bu sınıf prompt'a eklenecek context'i yönetsin. Sınıfın backend içindeki entegrasyonunu gerçekleştir. Şimdilik context değişkenini basit bir şekilde inşa et.

# tokenizer ile context birleştir
- tokenizer sınıfını contextmanager içinde bir fonksiyonda çağırarak ortaya çıkan prompt'un token sayısını bulan bir işlev ekle. son oluşturulan prompt'u bir nesnede saklayabilmek ve o nesne üzerinden token uzunluğunu alabilmek istiyorum.
- backend içinde oluşturulan prompt'un token uzunluğunu print ile ekrana yazdır.
## TEST ET

# basic RAG
employees.md cargo.md product_catalogs.md dosyalarını prompt içine ekleyerek basit bir RAG işlemi gerçekleştir.
## TEST ET Token sayısı arttı.
belgelerden verinin geldiğini test edebilmem için 2 test prompt'u ver. 
## TEST ET

# basic memory kayan pencere (sliding window)
- src/memory.py adında bir dosya oluştur. Bu dosyanın içine memorymanager adında bir sınıf oluştur. Basit bir liste içinde CRUD işlemleri gerçekleştirsin. Bu sınıfın asıl amacı konuşmanın geçmişini tutmaktır. Kullanıcı mesajları USER, llm cevapları ASISTAN key'i ile saklansın. ilk giren ilk çıkar yapısı ile çalışsın ve 10 mesaja kadar tutsun. Listenin içinde konuşmalar sessionid ile tutulsun. sessionid frontend.html içinde unique olarak oluşturulsun ve backend'e aktarılsın. sessionid değiştiğinde eski konuşmalar silinsin sessionid değişmediği sürece 10 mesaja kadar tutulsun. konuşma geçmişini prompt'a ekleme işlemini contextmanager üzerinden gerçekleştirelim.
- hafızayı test edebilmem için örnek bir test senaryosu ver. Belgelerle ilgili ve biraz karışık olsun.
## TEST ET, system prompt'u, hafızayı ve yanıtların kalitesini incele
## TEST ET, 10 soruya ulaştığında ilk soruyu unutması gerek.

# context compressing
- konuşmaların uzunluğu 10 mesajı geçtiğinde mesajların bir özetini oluşturalım. Bunun için src/summarizer.py adında bir dosya oluştur. Bu dosya içinde Summarizer sınıfı bulunsun. Bu sınıfın amacı bir session da konuşma 10 mesajı geçtiğinde model.py dosyasınıda kullanarak yapay zekaya hap bilgilerden oluşan bir özet oluşturtmak ve contextmanager'a bu özeti iletmek. Bundan sonra context'in içerisine eğer varsa history haricinde summary'i de ekleyelim. 
Kümülatif Özet
  - 10 mesaj → Özet oluştur → Mesajları sil
  - Sonraki 10 mesaj → Önceki özet + yeni mesajlar birlikte özetlenir
## TEST ET, 10 soru sonrasında özet sayesinde token sayısının düştüğünü gözlemle

# vector veritabanı ile context window içindeki tokenı azaltma
# chromadb örneğinden embedding.py'ı oluşturma
@../chromadb_ex.py 'ı kullanarak burda bir src/embedding.py dosyası oluştur. Bu dosya bir sınıftan oluşsun ve sınıfın fonksiyonları aşağıdaki gibi
  olsun:\
  init -> @../cargo.md @../employees.md @../product_catalogs.md dosyalarını alt başlıklarına göre chunklara ayırarak vector veritabanına gömsün. Her bir chunk'ın hangi belgeye ve başlığa ait olduğunu metaveriye ekle.\
  search -> verilen query'e göre vektör veritabanında arama yapsın ve sonuçları md formatında (to_md fonksiyonunu kullan) döndürsün\
  to_md -> chromadb'nin döndürdüğü formatı md formatına çevirsin

# model.py'ı tool kullanabilir halde güncellemek
 @../src/model.py içindeki sınıfı @../openai_tool_ex.py dosyasını inceleyerek tool kullanabilir halde güncelle. Sınıf init sırasında kullanabileceği toolları alsın. Ayrıca bu
  toollar @../src/context.py içindeki ContextManager tarafından sistem promptuna eklensin. Bunun içinde ContextManager sınıfına yeni bir fonksiyon ekle. @../tool_decorator.py
  dosyasını da incele OpenAIClient'e iletilen fonksiyonlar bu decorator ile oluşturulsun. OpenAIClient bir tool çağırısı olduğunda tool çağırısını işlesin fonksiyonu çağırsın
  yanıtı tekrar client'e göndersin bize en son sonuç iletilsin.

-  @../src/embedding.py içindeki search fonksiyonunu bir tool olarak ekle. Prompt'a doğrudan eklenen @../cargo.md @../employees.md @../product_catalogs.md dosyalarını kaldır. Bundan sonrasında model kendisi gerektiğinde search ile bilgi toplayacak. Bunun için gerekli backend değişikliklerini de yap.

- tool çağırma ve sonuç akışını görebilmemiz için ilgili yerlere printler ekle.

## TEST ET

# semantic arama kalitesini artırmak için yapabileceklerinden bahset ve bitir.
- Chunk şeklini değiştirmek,
- Döküman yapısını değiştirmek,
- Daha fazla sonuç çıkmasını sağlamak,
- Daha farklı modeller ile çalışmak.