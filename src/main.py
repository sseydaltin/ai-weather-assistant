import os
import signal
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt

# Ensure project root is on sys.path when running as `python src/main.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith import Client as LSClient
from src.agent import create_agent
from langchain_core.messages import HumanMessage


load_dotenv()


def _get_trace_link() -> str:
    try:
        proj = os.getenv("LANGSMITH_PROJECT", "ai-weather-assistant")
        return f"https://smith.langchain.com/o/~/projects/{proj}"
    except Exception:
        return "https://smith.langchain.com/"


def main():
    app = create_agent()
    session_id = os.getenv("DEFAULT_SESSION_ID") or str(uuid4())

    def handle_sigint(signum, frame):
        rprint("\n👋 Görüşürüz!")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    rprint(Panel.fit("🤖 [bold cyan]AI Weather Assistant[/bold cyan]", width=50))
    rprint(f"Session ID: [bold]{session_id}[/bold]")
    rprint(f"LangSmith Trace: {_get_trace_link()}")
    rprint("—" * 50)

    while True:
        user = Prompt.ask("[bold]You[/bold]")
        if user.strip().lower() in {"exit", "quit", ":q", "q"}:
            rprint("👋 Görüşürüz!")
            break

        state = {
            "messages": [HumanMessage(content=user)],
            "context": "",
            "next_action": "rag",
            "session_id": session_id,
        }

        out = app.invoke(state, config={"configurable": {"thread_id": session_id}})
        answer = out["messages"][-1].content
        rprint(f"🤖: {answer}")


if __name__ == "__main__":
    main()


