import os
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

# Ensure project root on path for `from src...` imports when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

from langchain_core.messages import HumanMessage
from src.agent import create_agent


def get_trace_link() -> str:
    proj = os.getenv("LANGSMITH_PROJECT", "ai-weather-assistant")
    return f"https://smith.langchain.com/o/~/projects/{proj}"


st.set_page_config(page_title="AI Weather Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Weather Assistant")
st.caption("RAG + OpenWeather + LangGraph • LangSmith Tracing")

if "session_id" not in st.session_state:
    st.session_state.session_id = os.getenv("DEFAULT_SESSION_ID") or str(uuid4())

if "agent_app" not in st.session_state:
    with st.status("RAG sistemi başlatılıyor...", expanded=True) as status:
        st.write("Ajan derleniyor ve hazır hale getiriliyor…")
        try:
            st.session_state.agent_app = create_agent()
            status.update(label="Sistem hazır ✅", state="complete")
        except Exception as e:
            status.update(label="Başlatma hatası ❌", state="error")
            st.error(f"Ajan başlatılırken hata: {e}")
            st.stop()

st.markdown(f"**Session ID**: `{st.session_state.session_id}`  ")
st.markdown(f"**LangSmith**: [{get_trace_link()}]({get_trace_link()})")
st.divider()

# Chat history using Streamlit chat components
if "chat" not in st.session_state:
    st.session_state.chat = []  # list[dict]: {role: "user"|"assistant", content: str}

for msg in st.session_state.chat:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Chat input anchored at bottom like ChatGPT
user_input = st.chat_input("Şehrini veya sorunuzu yazın…")
if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke agent
    app = st.session_state.agent_app
    state = {
        "messages": [HumanMessage(content=user_input)],
        "context": "",
        "next_action": "rag",
        "session_id": st.session_state.session_id,
    }
    with st.status("Yanıt hazırlanıyor…", expanded=False):
        out = app.invoke(state, config={"configurable": {"thread_id": st.session_state.session_id}})
    answer = out["messages"][ -1 ].content

    st.session_state.chat.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

st.info("İpuçları: 'Istanbul'da hava nasıl?', 'API key nasıl alınır?', 'Paris ve London'ın sıcaklıklarını karşılaştır' ")


