# src/rag.py
"""
RAG (Retrieval-Augmented Generation) System

Bu modül OpenWeatherMap API dökümanlarını:
1. Okur ve parçalara böler (chunking)
2. OpenAI ile embedding'lere dönüştürür
3. MongoDB Atlas'a vector olarak kaydeder
4. Semantic search ile ilgili dökümanları bulur

LangSmith ile tüm işlemler otomatik trace edilir.
"""

import os
from typing import List
import dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

from pymongo import MongoClient

# Environment variables yükle
dotenv.load_dotenv()


class RAGSystem:
    """
    Retrieval-Augmented Generation sistemi

    Attributes:
        client: MongoDB client bağlantısı
        db: MongoDB database
        collection: MongoDB collection (documents)
        embeddings: OpenAI embeddings modeli
        vectorstore: MongoDB vector store
    """

    def __init__(self):
        """
        RAG sistemini başlat
        - MongoDB bağlantısı
        - OpenAI embeddings
        - Vector store
        """
        print("\n🚀 RAG System başlatılıyor...")

        # ====================================================================
        # MongoDB Connection
        # ====================================================================
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError(
                "❌ MONGODB_URI environment variable tanımlı değil!\n"
                "   .env dosyasını kontrol edin."
            )

        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client["weather_assistant"]
            self.collection = self.db["documents"]

            # Connection test
            self.client.server_info()
            print("✅ MongoDB bağlantısı başarılı")

        except Exception as e:
            raise ConnectionError(f"❌ MongoDB bağlantı hatası: {e}")

        # ====================================================================
        # OpenAI Embeddings
        # ====================================================================
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "❌ OPENAI_API_KEY environment variable tanımlı değil!\n"
                "   .env dosyasını kontrol edin."
            )

        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",  # 1536 dimensions
                openai_api_key=openai_key
            )
            print("✅ OpenAI Embeddings modeli yüklendi")

        except Exception as e:
            raise ValueError(f"❌ OpenAI Embeddings hatası: {e}")

        # ====================================================================
        # MongoDB Atlas Vector Search
        # ====================================================================
        try:
            self.vectorstore = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embeddings,
                index_name="vector_index",  # Atlas'ta oluşturduğumuz index
                embedding_key="embedding",  # Embedding field adı
                text_key="text"  # Text field adı
            )
            print("✅ Vector Store hazır")

        except Exception as e:
            raise ValueError(f"❌ Vector Store hatası: {e}")

        print("🎉 RAG System başarıyla başlatıldı!\n")

    def load_documents(
            self,
            file_path: str = "/Users/code23-1/PycharmProjects/ai-weather-assistant /data/docs/openweather_api_docs.txt",
            chunk_size: int = 800,
            chunk_overlap: int = 100
    ) -> int:
        """
        Dökümanları yükle, parçala ve MongoDB'ye kaydet

        Args:
            file_path: Döküman dosya yolu
            chunk_size: Her chunk'ın maksimum karakter sayısı
            chunk_overlap: Chunk'lar arası örtüşme (karakter)

        Returns:
            Eklenen chunk sayısı

        Raises:
            FileNotFoundError: Dosya bulunamazsa
            Exception: Diğer hatalar
        """
        print(f"\n📥 Dökümanlar yükleniyor: {file_path}")
        print("-" * 70)

        # ====================================================================
        # Dosyayı Oku
        # ====================================================================
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            print(f"✅ Dosya okundu: {len(text):,} karakter")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Dosya bulunamadı: {file_path}\n"
                f"   Lütfen döküman dosyasını oluşturun."
            )
        except Exception as e:
            raise Exception(f"❌ Dosya okuma hatası: {e}")

        # ====================================================================
        # Text Splitting (Chunking)
        # ====================================================================
        print(f"\n✂️  Metin parçalanıyor...")
        print(f"   Chunk Size: {chunk_size}")
        print(f"   Chunk Overlap: {chunk_overlap}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Önce paragraflardan böl
                "\n",  # Sonra satırlardan
                " ",  # Sonra kelimelerden
                ""  # En son karakterlerden
            ],
            keep_separator=True
        )

        # Document oluştur
        doc = Document(
            page_content=text,
            metadata={
                "source": file_path,
                "type": "api_documentation",
                "total_chars": len(text)
            }
        )

        # Parçala
        chunks = text_splitter.split_documents([doc])

        print(f"✅ {len(chunks)} parçaya bölündü")

        # Chunk bilgileri
        avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"   Ortalama chunk boyutu: {avg_chunk_size:.0f} karakter")

        # ====================================================================
        # MongoDB'ye Ekle (Embedding otomatik oluşur)
        # ====================================================================
        print(f"\n💾 MongoDB'ye kaydediliyor...")

        try:
            # Her chunk için metadata ekle
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content)
                })

            # Vector store'a ekle (embedding'ler otomatik oluşur)
            ids = self.vectorstore.add_documents(chunks)

            print(f"✅ {len(ids)} döküman MongoDB'ye eklendi")
            print(f"   Vector Index: vector_index")
            print(f"   Embedding Boyutu: 1536 (text-embedding-3-small)")

            return len(ids)

        except Exception as e:
            raise Exception(f"❌ MongoDB kayıt hatası: {e}")

    def search(
            self,
            query: str,
            k: int = 3,
            score_threshold: float = 0.7
    ) -> List[Document]:
        """
        Sorguya semantically benzer dökümanları ara

        Args:
            query: Arama sorgusu
            k: Döndürülecek döküman sayısı
            score_threshold: Minimum benzerlik skoru (0-1)

        Returns:
            List of Document objects with similarity scores
        """
        print(f"\n🔍 Arama yapılıyor: '{query}'")
        print(f"   Top-K: {k}")
        print(f"   Score Threshold: {score_threshold}")

        try:
            # Similarity search with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=k
            )

            # Filter by score threshold
            filtered_docs = [
                (doc, score)
                for doc, score in docs_with_scores
                if score >= score_threshold
            ]

            print(f"✅ {len(filtered_docs)} sonuç bulundu")

            # Sadece document'leri döndür (score'ları metadata'ya ekle)
            results = []
            for doc, score in filtered_docs:
                doc.metadata["similarity_score"] = score
                results.append(doc)
                print(f"   📄 Score: {score:.3f} | {doc.page_content[:60]}...")

            return results

        except Exception as e:
            print(f"❌ Arama hatası: {e}")
            return []

    def get_context_for_query(
            self,
            query: str,
            k: int = 3,
            max_chars: int = 2000
    ) -> str:
        """
        Sorgu için context metni oluştur (LLM'e gönderilecek)

        Args:
            query: Arama sorgusu
            k: Kullanılacak döküman sayısı
            max_chars: Maksimum karakter sayısı

        Returns:
            Birleştirilmiş context metni
        """
        docs = self.search(query, k=k)

        if not docs:
            return "İlgili döküman bulunamadı."

        # Dökümanları birleştir
        context_parts = []
        total_chars = 0

        for i, doc in enumerate(docs, 1):
            doc_text = doc.page_content

            # Max char limit kontrolü
            if total_chars + len(doc_text) > max_chars:
                remaining = max_chars - total_chars
                doc_text = doc_text[:remaining] + "..."
                context_parts.append(f"[Döküman {i}]\n{doc_text}")
                break

            context_parts.append(f"[Döküman {i}]\n{doc_text}")
            total_chars += len(doc_text)

        context = "\n\n---\n\n".join(context_parts)

        print(f"📝 Context oluşturuldu: {len(context)} karakter")

        return context

    def get_collection_stats(self) -> dict:
        """MongoDB collection istatistiklerini döndür"""
        doc_count = self.collection.count_documents({})

        # Sample document al
        sample_doc = self.collection.find_one({})

        return {
            "total_documents": doc_count,
            "has_embeddings": bool(sample_doc and "embedding" in sample_doc),
            "embedding_size": len(sample_doc.get("embedding", [])) if sample_doc else 0
        }

    def clear_collection(self):
        """
        Collection'ı temizle (test için)

        ⚠️ DİKKAT: Tüm dökümanları siler!
        """
        result = self.collection.delete_many({})
        print(f"🗑️  {result.deleted_count} döküman silindi")


# ============================================================================
# Test ve Demo Fonksiyonu
# ============================================================================

def main():
    """RAG sistemini test et ve demo yap"""

    print("=" * 70)
    print(" RAG SYSTEM - TEST VE DEMO")
    print("=" * 70)

    # ====================================================================
    # RAG System Oluştur
    # ====================================================================
    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"\n❌ RAG System oluşturulamadı: {e}")
        return

    # ====================================================================
    # Collection İstatistikleri
    # ====================================================================
    print("\n" + "=" * 70)
    print(" MONGODB COLLECTION İSTATİSTİKLERİ")
    print("=" * 70)

    stats = rag.get_collection_stats()
    print(f"📊 Toplam Döküman: {stats['total_documents']}")
    print(f"🔢 Embedding Var mı: {stats['has_embeddings']}")
    print(f"📏 Embedding Boyutu: {stats['embedding_size']}")

    # ====================================================================
    # Döküman Yükleme (Eğer collection boşsa)
    # ====================================================================
    if stats['total_documents'] == 0:
        print("\n" + "=" * 70)
        print(" DÖKÜMAN YÜKLEME")
        print("=" * 70)

        try:
            chunk_count = rag.load_documents(
                file_path="data/docs/openweather_api_docs.txt",
                chunk_size=800,
                chunk_overlap=100
            )
            print(f"\n✅ {chunk_count} döküman chunk'ı başarıyla yüklendi!")
        except Exception as e:
            print(f"\n❌ Döküman yükleme hatası: {e}")
            print("   data/docs/openweather_api_docs.txt dosyasının var olduğundan emin ol")
            return
    else:
        print(f"\n✅ Dökümanlar zaten yüklenmiş (Toplam: {stats['total_documents']})")
        print("   Yeniden yüklemek için önce rag.clear_collection() çalıştır")

    # ====================================================================
    # Test Sorguları
    # ====================================================================
    print("\n" + "=" * 70)
    print(" TEST SORGULARI")
    print("=" * 70)

    test_queries = [
        {
            "query": "How do I get an API key?",
            "description": "API key alma süreci"
        },
        {
            "query": "What is the endpoint for current weather?",
            "description": "Current weather endpoint"
        },
        {
            "query": "What units can I use for temperature?",
            "description": "Sıcaklık birimleri"
        },
        {
            "query": "How to handle 401 error?",
            "description": "Hata yönetimi"
        },
        {
            "query": "API key nasıl alınır?",
            "description": "Türkçe sorgu testi"
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}: {test['description']}")
        print(f"{'─' * 70}")
        print(f"❓ Sorgu: {test['query']}")

        # Arama yap
        docs = rag.search(test['query'], k=2, score_threshold=0.6)

        if docs:
            for j, doc in enumerate(docs, 1):
                score = doc.metadata.get('similarity_score', 0)
                print(f"\n📄 Sonuç {j} (Score: {score:.3f})")
                print("─" * 70)
                # İlk 300 karakteri göster
                content = doc.page_content[:300]
                print(content)
                if len(doc.page_content) > 300:
                    print("...")
        else:
            print("❌ İlgili döküman bulunamadı")

    # ====================================================================
    # Context Oluşturma Demo
    # ====================================================================
    print("\n" + "=" * 70)
    print(" CONTEXT OLUŞTURMA DEMO")
    print("=" * 70)

    demo_query = "API key nereden alınır ve nasıl kullanılır?"
    print(f"\n❓ Sorgu: {demo_query}")

    context = rag.get_context_for_query(
        query=demo_query,
        k=3,
        max_chars=1500
    )

    print("\n📝 Oluşturulan Context:")
    print("─" * 70)
    print(context)

    # ====================================================================
    # Özet
    # ====================================================================
    print("\n" + "=" * 70)
    print(" TEST TAMAMLANDI")
    print("=" * 70)
    print("✅ RAG sistemi başarıyla çalışıyor!")
    print("✅ Dökümanlar MongoDB'de")
    print("✅ Semantic search çalışıyor")
    print("✅ Context oluşturma hazır")
    print("\n💡 Bir sonraki adım: Weather API tool'u oluştur")
    print("=" * 70)


if __name__ == "__main__":
    main()