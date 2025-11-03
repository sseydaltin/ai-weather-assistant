import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


def _get_mongo() -> MongoClient:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI tanımlı değil")
    return MongoClient(uri)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 5.1 Short-term Memory (Conversation History)
def save_message(session_id: str, role: str, content: str) -> str:
    client = _get_mongo()
    db = client[os.getenv("MONGODB_DB_NAME", "weather_assistant")]
    col = db[os.getenv("MONGODB_COLLECTION_CONVERSATIONS", "conversations")]
    doc = {
        "type": "message",
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": _now_iso(),
    }
    result = col.insert_one(doc)
    return str(result.inserted_id)


def get_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    client = _get_mongo()
    db = client[os.getenv("MONGODB_DB_NAME", "weather_assistant")]
    col = db[os.getenv("MONGODB_COLLECTION_CONVERSATIONS", "conversations")]
    cursor = (
        col.find({"session_id": session_id, "type": "message"})
        .sort("timestamp", 1)
    )
    items = list(cursor)
    if len(items) > limit:
        items = items[-limit:]
    return [{"role": i["role"], "content": i["content"], "timestamp": i["timestamp"]} for i in items]


# 5.2 Context Window Management
def estimate_tokens(text: str) -> int:
    # Yaklaşık: 4 karakter ≈ 1 token
    return max(1, int(len(text) / 4))


def manage_context_window(messages: List[Dict[str, str]], max_tokens: int = 4000) -> Dict[str, Any]:
    running_tokens = 0
    trimmed: List[Dict[str, str]] = []
    # Son mesajlardan başlayarak ekle (en yeniler sonda varsayılır)
    for msg in reversed(messages):
        tokens = estimate_tokens(msg.get("content", ""))
        if running_tokens + tokens > max_tokens:
            break
        trimmed.append(msg)
        running_tokens += tokens

    trimmed.reverse()
    warning = "" if len(trimmed) == len(messages) else "⚠️ Context window doldu, bazı eski mesajlar çıkarıldı."

    return {
        "messages": trimmed,
        "used_tokens": running_tokens,
        "warning": warning,
    }


# 5.3 Long-term Memory (Conversation Summary)
def save_summary(session_id: str, summary: str) -> str:
    client = _get_mongo()
    db = client[os.getenv("MONGODB_DB_NAME", "weather_assistant")]
    col = db[os.getenv("MONGODB_COLLECTION_CONVERSATIONS", "conversations")]
    doc = {
        "type": "summary",
        "session_id": session_id,
        "summary": summary,
        "timestamp": _now_iso(),
    }
    result = col.insert_one(doc)
    return str(result.inserted_id)


def get_summaries(session_id: str) -> List[Dict[str, Any]]:
    client = _get_mongo()
    db = client[os.getenv("MONGODB_DB_NAME", "weather_assistant")]
    col = db[os.getenv("MONGODB_COLLECTION_CONVERSATIONS", "conversations")]
    cursor = (
        col.find({"session_id": session_id, "type": "summary"})
        .sort("timestamp", 1)
    )
    return [{"summary": i.get("summary", ""), "timestamp": i.get("timestamp", "")} for i in cursor]


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY TESTS")
    print("=" * 60)

    sid = os.getenv("DEFAULT_SESSION_ID", f"test-{int(time.time())}")
    print(f"Session: {sid}")

    # Short-term
    save_message(sid, "user", "Merhaba, Istanbul'un havası nasıl?")
    save_message(sid, "assistant", "Istanbul'da hava güneşli ve 22°C.")
    save_message(sid, "user", "Peki London?")

    hist = get_conversation_history(sid, limit=10)
    print(f"Son {len(hist)} mesaj")
    for h in hist:
        print(f"- {h['role']}: {h['content']}")

    # Context window overflow testi
    big_messages = [{"role": "user", "content": "x" * 1000} for _ in range(30)]
    result = manage_context_window(big_messages, max_tokens=4000)
    print(f"Context tokens: {result['used_tokens']} | Warning: {bool(result['warning'])}")

    # Long-term
    save_summary(sid, "Kullanıcı sık sık Istanbul ve London hava durumunu soruyor.")
    summaries = get_summaries(sid)
    print(f"Toplam özet: {len(summaries)}")


