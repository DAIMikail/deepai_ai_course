# Gerekli kütüphaneler
# pip install chromadb==1.4.0 openai python-dotenv

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from dotenv import load_dotenv

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

# ============================================
# 1. ChromaDB Client Oluşturma
# ============================================

# Persistent (kalıcı) client - veriler diske kaydedilir
client = chromadb.PersistentClient(path="./chroma_db")

# Alternatif: In-memory client (RAM'de, geçici)
# client = chromadb.Client()

# ============================================
# 2. OpenAI Embedding Fonksiyonu
# ============================================

openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_KEY,
    model_name="text-embedding-3-small"  # veya "text-embedding-3-large"
)

# ============================================
# 3. Collection Oluşturma
# ============================================

# Eğer varsa sil, yeniden oluştur (demo için)
try:
    client.delete_collection("documents")
except:
    pass

collection = client.create_collection(
    name="documents",
    embedding_function=openai_ef,
    metadata={"description": "Örnek döküman koleksiyonu"}
)

# ============================================
# 4. Dökümanları Ekleme
# ============================================

documents = [
    "Python, yüksek seviyeli ve genel amaçlı bir programlama dilidir. Okunabilirliği ve basit sözdizimi ile bilinir.",
    "Machine Learning, bilgisayarların açıkça programlanmadan verilerden öğrenmesini sağlayan yapay zeka dalıdır.",
    "Deep Learning, yapay sinir ağlarını kullanarak karmaşık örüntüleri öğrenen bir makine öğrenmesi alt dalıdır.",
    "ChromaDB, vektör veritabanı olarak kullanılan açık kaynaklı bir embedding store'dur.",
    "LangChain, LLM uygulamaları geliştirmek için kullanılan popüler bir Python framework'üdür.",
    "RAG (Retrieval Augmented Generation), LLM'lere harici bilgi kaynakları ekleyerek yanıt kalitesini artırır.",
    "Transformer mimarisi, attention mekanizması kullanan ve NLP'de devrim yaratan bir derin öğrenme modelidir.",
    "Fine-tuning, önceden eğitilmiş bir modeli belirli bir görev için yeniden eğitme sürecidir.",
]

# Metadata ekleyelim
metadatas = [
    {"category": "programming", "difficulty": "beginner"},
    {"category": "ai", "difficulty": "intermediate"},
    {"category": "ai", "difficulty": "advanced"},
    {"category": "database", "difficulty": "intermediate"},
    {"category": "framework", "difficulty": "intermediate"},
    {"category": "ai", "difficulty": "intermediate"},
    {"category": "ai", "difficulty": "advanced"},
    {"category": "ai", "difficulty": "advanced"},
]

# Unique ID'ler
ids = [f"doc_{i}" for i in range(len(documents))]

# Dökümanları ekle
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ {len(documents)} döküman eklendi!")
print(f"📊 Koleksiyon boyutu: {collection.count()}")

# ============================================
# 5. Semantic Arama (Query)
# ============================================

print("\n" + "="*50)
print("🔍 SEMANTIC ARAMA ÖRNEKLERİ")
print("="*50)

# Örnek 1: Basit arama
query = "Yapay zeka nasıl öğrenir?"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(f"\n📌 Sorgu: '{query}'")
print("-" * 40)
for i, (doc, meta, distance) in enumerate(zip(
    results['documents'][0], 
    results['metadatas'][0],
    results['distances'][0]
)):
    print(f"{i+1}. [Skor: {1-distance:.4f}] {doc[:80]}...")
    print(f"   Kategori: {meta['category']}, Seviye: {meta['difficulty']}")

# Örnek 2: Metadata filtreli arama
query2 = "Model eğitimi"
results2 = collection.query(
    query_texts=[query2],
    n_results=3,
    where={"difficulty": "advanced"}  # Sadece advanced dökümanlar
)

print(f"\n📌 Sorgu: '{query2}' (sadece advanced)")
print("-" * 40)
for i, doc in enumerate(results2['documents'][0]):
    print(f"{i+1}. {doc[:80]}...")

# Örnek 3: Birden fazla sorgu
queries = ["veritabanı çözümleri", "LLM framework"]
results3 = collection.query(
    query_texts=queries,
    n_results=2
)

print(f"\n📌 Çoklu Sorgu Sonuçları:")
print("-" * 40)
for q_idx, query in enumerate(queries):
    print(f"\n➡️ '{query}':")
    for doc in results3['documents'][q_idx]:
        print(f"   • {doc[:60]}...")

# ============================================
# 6. Döküman Güncelleme
# ============================================

collection.update(
    ids=["doc_0"],
    documents=["Python, en popüler programlama dillerinden biridir. AI ve ML projelerinde yaygın kullanılır."],
    metadatas=[{"category": "programming", "difficulty": "beginner", "updated": True}]
)
print("\n✅ doc_0 güncellendi!")

# ============================================
# 7. ID ile Döküman Getirme
# ============================================

specific_docs = collection.get(
    ids=["doc_0", "doc_3"],
    include=["documents", "metadatas"]
)
print("\n📄 Belirli dökümanlar:")
for doc, meta in zip(specific_docs['documents'], specific_docs['metadatas']):
    print(f"   • {doc[:50]}... | {meta}")

# ============================================
# 8. Koleksiyon Bilgileri
# ============================================

print("\n" + "="*50)
print("📊 KOLEKSİYON BİLGİLERİ")
print("="*50)
print(f"Ad: {collection.name}")
print(f"Toplam döküman: {collection.count()}")
print(f"Metadata: {collection.metadata}")

# Tüm koleksiyonları listele
print(f"\nMevcut koleksiyonlar: {[c.name for c in client.list_collections()]}")
