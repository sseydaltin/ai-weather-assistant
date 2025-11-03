## AI Weather Assistant

Akıllı hava durumu asistanı: OpenWeatherMap canlı verisi ile MongoDB Atlas üzerinde kurulu RAG sistemi ve LangGraph ajan iş akışını birleştirir. OpenAI (GPT-4o-mini) ile soruları sınıflandırır, gerekirse doküman bilgisini getirir, gerekirse canlı hava durumunu çeker veya ikisini birleştirir. Tüm süreç LangSmith ile izlenebilir.

### Özellikler
- ✅ RAG (MongoDB Atlas Vector Search)
- ✅ Canlı hava durumu (OpenWeatherMap)
- ✅ Kısa ve uzun süreli bellek (MongoDB)
- ✅ LangGraph ajan iş akışı (classify → rag/weather → respond)
- ✅ LangSmith tracing
- ✅ Context window yönetimi
- ✅ Sağlam hata yönetimi ve timeout

### Mimari
```
┌─────────────────────────────────────────────┐
│           AI Weather Assistant              │
├─────────────────────────────────────────────┤
│                                             │
│  User Input                                 │
│      ↓                                      │
│  ┌──────────────────┐                      │
│  │  LangGraph Agent │                      │
│  └────────┬─────────┘                      │
│           ↓                                 │
│     Classify Query                          │
│           ↓                                 │
│    ┌──────┴──────┐                         │
│    ↓             ↓                          │
│  [RAG]        [Weather API]                 │
│    ↓             ↓                          │
│  MongoDB      OpenWeather                   │
│  (Vector)     (Live Data)                   │
│    ↓             ↓                          │
│  ┌─────────────────┐                       │
│  │  GPT-4o-mini    │                       │
│  │  (Response)     │                       │
│  └────────┬────────┘                       │
│           ↓                                 │
│     User Response                           │
│                                             │
│  📊 LangSmith: Tracing & Monitoring        │
│  💾 MongoDB: Vector Store + Memory         │
│  🤖 OpenAI: LLM + Embeddings               │
└─────────────────────────────────────────────┘
```

### Teknoloji Yığını
- **Dil**: Python 3.11
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: text-embedding-3-small (1536)
- **Framework**: LangGraph, LangChain
- **Vector Store**: MongoDB Atlas
- **Monitoring**: LangSmith
- **API**: OpenWeatherMap

### Proje Yapısı
```
ai-weather-assistant/
├── .env                    # Ortam değişkenleri (GITIGNORE!)
├── .gitignore
├── requirements.txt
├── langgraph.json
├── README.md
├── src/
│   ├── agent.py           # LangGraph agent (CORE)
│   ├── rag.py             # RAG system
│   ├── tools.py           # Weather API tool
│   ├── memory.py          # Memory management
│   ├── main.py            # CLI application
│   ├── rag_helpers.py     # Helper functions
│   ├── test_apis.py       # API tests
│   └── test_mongo.py      # MongoDB tests
└── data/
    └── docs/
        └── openweather_api_docs.txt
```

### Kurulum

Önkoşullar:
```
- Python 3.10+
- MongoDB Atlas (free tier)
- OpenWeatherMap API key
- OpenAI API key
- LangSmith API key
```

Kurulum adımları:
```bash
# 1. Klonla
git clone <repo-url>
cd ai-weather-assistant

# 2. Sanal ortam
python3 -m venv venv
source venv/bin/activate

# 3. Bağımlılıklar
pip install -r requirements.txt

# 4. Ortam değişkenleri
cp env.example .env
# .env dosyasını düzenleyip anahtarları ekleyin
```

MongoDB Atlas Kurulumu:
1. mongodb.com/cloud/atlas → free cluster
2. Database: `weather_assistant`
3. Collections: `documents`, `conversations`
4. Vector index (`documents`):
```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```
5. Connection string'i `.env` içine kopyalayın.

OpenWeatherMap Kurulumu:
1. openweathermap.org → API key alın.
2. Aktivasyon 10-120 dk sürebilir.
3. `.env` → `OPENWEATHER_API_KEY` alanını doldurun.

### Çalıştırma

İlk yükleme (dokümanlar):
```bash
python src/rag.py
```

Bileşen testleri:
```bash
python src/test_apis.py
python src/rag.py
python src/tools.py
python src/memory.py
python src/agent.py
```

Ana uygulama:
```bash
python src/main.py
```

LangGraph Studio (opsiyonel):
```bash
langgraph dev
# Studio'da local gruba bağlanın (localhost:8000)
```

### Örnek Akış
```
🤖 AI Weather Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Session ID: abc123
LangSmith Trace: https://smith.langchain.com/o/.../runs/...

You: API key nasıl alınır?
🤖: OpenWeatherMap API key almak için...

You: Istanbul'da hava nasıl?
🤖: 🌍 Istanbul Hava Durumu: ...

You: exit
👋 Görüşürüz!
```


