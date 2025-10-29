# src/test_mongo.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB'ye bağlan
client = MongoClient(os.getenv("MONGODB_URI"))

# Test
try:
    # Server info
    info = client.server_info()
    print(f"✅ MongoDB bağlantısı başarılı!")
    print(f"📊 MongoDB Version: {info['version']}")

    # Database ve collection test
    db = client["weather_assistant"]
    print(f"✅ Database: {db.name}")

    collections = db.list_collection_names()
    print(f"✅ Collections: {collections}")

except Exception as e:
    print(f"❌ Bağlantı hatası: {e}")