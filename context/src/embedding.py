"""Document Embedding Module

Bu modül markdown dosyalarını chunklara ayırarak ChromaDB vektör veritabanına gömer
ve semantic arama işlemleri gerçekleştirir.
"""

import os
import sys
import re
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from pathlib import Path

# tool_decorator'ı import et
sys.path.insert(0, str(Path(__file__).parent.parent))
from tool_decorator import openai_tool


class DocumentEmbedding:
    """Dökümanları vektör veritabanına gömen ve arama yapan sınıf."""

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "documents"):
        """
        Markdown dosyalarını chunklara ayırarak vektör veritabanına gömer.

        Args:
            db_path: ChromaDB veritabanı yolu
            collection_name: Koleksiyon adı
        """
        # .env dosyasından API key'i yükle
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")

        # ChromaDB client oluştur
        self.client = chromadb.PersistentClient(path=db_path)

        # OpenAI embedding fonksiyonu
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=self.openai_key,
            model_name="text-embedding-3-small"
        )

        # Mevcut koleksiyonu sil ve yeniden oluştur
        try:
            self.client.delete_collection(collection_name)
        except:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "TeknoElektronik dokümantasyon koleksiyonu"}
        )

        # Dosya yollarını tanımla
        base_path = Path(__file__).parent.parent
        doc_files = {
            "cargo": base_path / "cargo.md",
            "employees": base_path / "employees.md",
            "product_catalogs": base_path / "product_catalogs.md"
        }

        # Her dosyayı işle ve veritabanına ekle
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc_name, file_path in doc_files.items():
            chunks, headers = self._chunk_markdown(file_path)

            for i, (chunk, header) in enumerate(zip(chunks, headers)):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "document": doc_name,
                    "header": header,
                    "source_file": str(file_path.name)
                })
                all_ids.append(f"{doc_name}_{i}")

        # Tüm dökümanları ekle
        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )

        print(f"Toplam {len(all_chunks)} chunk veritabanına eklendi.")

    def _chunk_markdown(self, file_path: Path) -> tuple[list[str], list[str]]:
        """
        Markdown dosyasını alt başlıklarına göre chunklara ayırır.

        Args:
            file_path: Markdown dosyasının yolu

        Returns:
            chunks: İçerik parçaları listesi
            headers: Her chunk'a karşılık gelen başlık listesi
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = []
        headers = []

        # ## veya ### ile başlayan başlıkları bul
        # Her başlık ve altındaki içeriği bir chunk olarak al
        pattern = r'^(#{2,3})\s+(.+?)$'
        lines = content.split('\n')

        current_header = "Giriş"
        current_chunk_lines = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Önceki chunk'ı kaydet (eğer içerik varsa)
                if current_chunk_lines:
                    chunk_content = '\n'.join(current_chunk_lines).strip()
                    if chunk_content and len(chunk_content) > 50:  # Çok kısa chunkları atla
                        chunks.append(chunk_content)
                        headers.append(current_header)

                # Yeni başlık ile devam et
                current_header = match.group(2).strip()
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)

        # Son chunk'ı da ekle
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines).strip()
            if chunk_content and len(chunk_content) > 50:
                chunks.append(chunk_content)
                headers.append(current_header)

        return chunks, headers

    def search(self, query: str, n_results: int = 5, where: dict = None) -> str:
        """
        Vektör veritabanında semantic arama yapar.

        Args:
            query: Arama sorgusu
            n_results: Döndürülecek sonuç sayısı
            where: Metadata filtresi (opsiyonel)

        Returns:
            Markdown formatında arama sonuçları
        """
        search_params = {
            "query_texts": [query],
            "n_results": n_results
        }

        if where:
            search_params["where"] = where

        results = self.collection.query(**search_params)

        return self.to_md(results)

    def to_md(self, results: dict) -> str:
        """
        ChromaDB sorgu sonuçlarını markdown formatına çevirir.

        Args:
            results: ChromaDB query sonuçları

        Returns:
            Markdown formatında sonuçlar
        """
        if not results or not results.get('documents') or not results['documents'][0]:
            return "Sonuç bulunamadı."

        md_output = []
        md_output.append("# Arama Sonuçları\n")

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results.get('distances', [[]])[0]

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # Benzerlik skoru hesapla (distance'ı skora çevir)
            score = 1 - distances[i] if distances else None

            md_output.append(f"## Sonuç {i + 1}")
            md_output.append(f"**Belge:** {meta.get('document', 'Bilinmiyor')}")
            md_output.append(f"**Bölüm:** {meta.get('header', 'Bilinmiyor')}")
            md_output.append(f"**Kaynak:** {meta.get('source_file', 'Bilinmiyor')}")

            if score is not None:
                md_output.append(f"**Benzerlik Skoru:** {score:.4f}")

            md_output.append(f"\n### İçerik\n")
            # İçeriği kısalt (çok uzunsa)
            content = doc if len(doc) <= 500 else doc[:500] + "..."
            md_output.append(content)
            md_output.append("\n---\n")

        return '\n'.join(md_output)


# Global embedder instance (lazy initialization)
_embedder: DocumentEmbedding = None


def get_embedder() -> DocumentEmbedding:
    """
    Global embedder instance'ı döndürür.
    İlk çağrıda oluşturulur (lazy initialization).
    """
    global _embedder
    if _embedder is None:
        _embedder = DocumentEmbedding()
    return _embedder


@openai_tool
def bilgi_ara(sorgu: str, sonuc_sayisi: int = 3) -> str:
    """Sirket dokumanlarinda bilgi arar. Urunler, kargo, iade, garanti ve calisan bilgileri icin kullan.

    Args:
        sorgu: Aranacak konu veya soru
        sonuc_sayisi: Dondurulecek maksimum sonuc sayisi

    Returns:
        Bulunan bilgiler markdown formatinda
    """
    embedder = get_embedder()
    return embedder.search(sorgu, n_results=sonuc_sayisi)


# Test için
if __name__ == "__main__":
    # Tool schema'sını göster
    print("Tool Schema:")
    print(bilgi_ara.schema)
    print()

    # Örnek kullanım
    print("="*50)
    print("ARAMA: Kargo iade prosedürü")
    print("="*50)
    result = bilgi_ara("Kargo iade prosedürü nasıl işliyor?", sonuc_sayisi=3)
    print(result)

    print("\n" + "="*50)
    print("ARAMA: Garanti kapsamı")
    print("="*50)
    result = bilgi_ara("Garanti kapsamı nedir?", sonuc_sayisi=2)
    print(result)
