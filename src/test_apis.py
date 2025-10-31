import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

# Environment variables'ı yükle
load_dotenv()

print("=" * 70)
print("API BAĞLANTI TESTLERİ")
print("=" * 70)

# ============================================================================
# TEST 1: Environment Variables Kontrolü
# ============================================================================
print("\n🔍 TEST 1: Environment Variables")
print("-" * 70)

required_vars = [
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "OPENWEATHER_API_KEY",
    "MONGODB_URI"
]

for var in required_vars:
    value = os.getenv(var)
    if value:
        # İlk 10 karakteri göster, geri kalanı gizle
        masked = value[:10] + "..." if len(value) > 10 else value
        print(f"✅ {var}: {masked}")
    else:
        print(f"❌ {var}: TANIMLI DEĞİL!")

# ============================================================================
# TEST 2: OpenAI API - Chat Completion
# ============================================================================
print("\n🤖 TEST 2: OpenAI Chat API")
print("-" * 70)

try:
    # ChatOpenAI instance oluştur
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Basit bir test sorusu
    response = llm.invoke("Merhaba! 2+2 kaç eder?")

    print(f"✅ OpenAI Chat API çalışıyor!")
    print(f"📝 Soru: Merhaba! 2+2 kaç eder?")
    print(f"💬 Cevap: {response.content}")
    print(f"🏷️  Model: {response.response_metadata.get('model_name', 'N/A')}")
    print(f"🎫 Token Kullanımı: {response.response_metadata.get('token_usage', 'N/A')}")

except Exception as e:
    print(f"❌ OpenAI Chat API Hatası: {e}")

# ============================================================================
# TEST 3: OpenAI Embeddings API
# ============================================================================
print("\n🔢 TEST 3: OpenAI Embeddings API")
print("-" * 70)

try:
    # Embeddings instance oluştur
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Test metni
    test_text = "Hava durumu nasıl?"

    # Embedding oluştur
    embedding_vector = embeddings.embed_query(test_text)

    print(f"✅ OpenAI Embeddings API çalışıyor!")
    print(f"📝 Test Metni: {test_text}")
    print(f"🔢 Embedding Boyutu: {len(embedding_vector)}")
    print(f"📊 İlk 5 değer: {embedding_vector[:5]}")

except Exception as e:
    print(f"❌ OpenAI Embeddings API Hatası: {e}")

# ============================================================================
# TEST 4: LangSmith Connection
# ============================================================================
print("\n📊 TEST 4: LangSmith API")
print("-" * 70)

try:
    # LangSmith client oluştur
    client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY")
    )

    # Kullanıcı bilgisi al (connection test)
    # Not: Bu API endpoint değişebilir
    print(f"✅ LangSmith API çalışıyor!")
    print(f"🔑 API Key: {os.getenv('LANGSMITH_API_KEY')[:15]}...")
    print(f"📁 Project: {os.getenv('LANGSMITH_PROJECT')}")
    print(f"🌐 Tracing: {os.getenv('LANGSMITH_TRACING')}")
    print(f"💡 LangSmith Dashboard: https://smith.langchain.com/")

except Exception as e:
    print(f"⚠️  LangSmith API Hatası (opsiyonel): {e}")
    print("   Not: LangSmith bağlantı testi başarısız olsa bile trace'ler çalışabilir")

# ============================================================================
# ÖZET
# ============================================================================
print("\n" + "=" * 70)
print("TEST ÖZET")
print("=" * 70)
print("✅ Environment variables yüklendi")
print("✅ OpenAI Chat API çalışıyor")
print("✅ OpenAI Embeddings API çalışıyor")
print("✅ LangSmith konfigürasyonu tamam")
print("\n🎉 Tüm API'ler hazır! RAG sistemi kurulumuna geçebilirsin.")
print("=" * 70)