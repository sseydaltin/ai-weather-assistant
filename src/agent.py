from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import TypedDict, List

# Ensure project root is on sys.path when running as `python src/agent.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.rag import RAGSystem
from src.tools import get_current_weather
from src import memory as mem


load_dotenv()


class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str
    next_action: str
    session_id: str


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
rag_system: RAGSystem | None = None


def _ensure_rag() -> RAGSystem:
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system


# 1) classify_query
def classify_query(state: AgentState) -> AgentState:
    user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    prompt = (
        "Kullanıcı sorusu döküman bilgisi mi gerektiriyor yoksa canlı hava durumu API'si mi?\n"
        "Sadece şu yanıtlardan birini ver: rag, weather, both.\n\n"
        f"Soru: {user_msg}"
    )
    res = llm.invoke(prompt)
    label = (res.content or "rag").strip().lower()
    if label not in {"rag", "weather", "both"}:
        label = "rag"
    state["next_action"] = label
    return state


# 2) rag_node
def rag_node(state: AgentState) -> AgentState:
    user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    rag = _ensure_rag()
    context = rag.get_context_for_query(user_msg, k=3, max_chars=2000)
    state["context"] = context
    return state


# 3) weather_node
def _extract_city(question: str) -> str:
    # Basit LLM destekli çıkarım + regex fallback
    sys_prompt = (
        "Aşağıdaki cümlede geçen şehir adını tek kelime olarak döndür.\n"
        "Sadece şehir adını yaz, başka bir şey yazma.\n"
        f"Metin: {question}"
    )
    try:
        out = llm.invoke(sys_prompt).content.strip()
        if out:
            return out.split("\n")[0].strip()
    except Exception:
        pass

    m = re.search(r"in |de |da |'da |'de |\bfor\b|\bof\b|\bin\b\s+([A-ZİIŞĞÜÖ][a-zçıüğöşı]+)", question)
    if m:
        return m.group(1)
    return question.strip()


def weather_node(state: AgentState) -> AgentState:
    user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    city = _extract_city(user_msg)
    weather_text = get_current_weather.invoke(city)
    # Context'e ekle
    prev = state.get("context", "")
    state["context"] = (prev + "\n\n" if prev else "") + weather_text
    return state


# 4) respond_node
def respond_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id") or os.getenv("DEFAULT_SESSION_ID", "local-dev")
    history_items = mem.get_conversation_history(session_id, limit=10)
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history_items])

    user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    context = state.get("context", "")

    prompt = (
        "Aşağıdaki bağlamı ve önceki konuşma geçmişini kullanarak kullanıcıya kısa, net ve Türkçe cevap ver.\n"
        "Gerektiğinde madde işaretleri ve emoji kullan.\n\n"
        f"[Geçmiş]\n{history_text}\n\n[Bağlam]\n{context}\n\n[Soru]\n{user_msg}"
    )

    answer = llm.invoke(prompt).content

    # Memory kaydet
    mem.save_message(session_id, "user", user_msg)
    mem.save_message(session_id, "assistant", answer)

    state["messages"].append(AIMessage(content=answer))
    return state


def create_agent():
    workflow = StateGraph(AgentState)
    workflow.set_entry_point("classify")
    workflow.add_node("classify", classify_query)
    workflow.add_node("rag", rag_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("respond", respond_node)

    workflow.add_conditional_edges(
        "classify",
        lambda s: s["next_action"],
        {"rag": "rag", "weather": "weather", "both": "rag"},
    )

    workflow.add_edge("rag", "respond")
    workflow.add_edge("weather", "respond")
    workflow.add_edge("respond", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


if __name__ == "__main__":
    from uuid import uuid4

    app = create_agent()
    sid = os.getenv("DEFAULT_SESSION_ID", str(uuid4()))

    def run(q: str):
        state = {
            "messages": [HumanMessage(content=q)],
            "context": "",
            "next_action": "rag",
            "session_id": sid,
        }
        out = app.invoke(state, config={"configurable": {"thread_id": sid}})
        last = out["messages"][-1]
        print(f"\nYou: {q}\n🤖: {last.content}\n")

    # Tests
    run("API key nasıl alınır?")
    run("Istanbul'da hava nasıl?")
    run("API kullanarak Istanbul'un havasını nasıl öğrenebilirim?")
    run("Paris ve London'ın sıcaklıklarını karşılaştır")
    run("Daha önce hangi şehrin havasını sormuştum?")


