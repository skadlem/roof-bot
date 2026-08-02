"""Fakes and helpers for the test suite. Zero network, zero real API calls."""

import threading
import time

from langchain_core.messages import AIMessage, HumanMessage

# --- LLM fakes (instead of ChatGoogleGenerativeAI) ---


def make_response(text="Привет!"):
    """Фейковый ответ модели: текст без function_call (как обычная реплика Gemini)."""
    return AIMessage(content=text)


def make_tool_call_response(name, args, tool_call_id="call_1"):
    """Фейковый ответ модели с function_call (lookup_pricing / create_lead)."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": tool_call_id}]
    )


class FakeLLM:
    """Scripted-ответы вместо ChatGoogleGenerativeAI (интерфейс: bind_tools/invoke)."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.invoked = []          # каждый вызов: список сообщений
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools.append(tools)
        return self

    def invoke(self, messages):
        self.invoked.append(list(messages))
        return self.responses.pop(0) if self.responses else AIMessage(content="Привет!")

    @property
    def sent_messages(self):
        # последний HumanMessage каждого вызова — «что отправил бот» (для test_kb)
        return [
            next(
                (m.content for m in reversed(msgs) if isinstance(m, HumanMessage)),
                None,
            )
            for msgs in self.invoked
        ]


class TrackingFakeLLM(FakeLLM):
    """FakeLLM, замеряющий параллельность invoke."""

    def __init__(self, tracker, delay=0.15):
        super().__init__()
        self.tracker = tracker
        self.delay = delay

    def invoke(self, messages):
        self.tracker.enter()
        try:
            time.sleep(self.delay)  # держим слот занятым, чтобы замерить параллельность
            return super().invoke(messages)
        finally:
            self.tracker.exit()


# --- WhatsApp webhook message builders ---


def make_text_message(phone, text, mid="wamid.test"):
    return {"from": phone, "id": mid, "type": "text", "text": {"body": text}}


def webhook_payload(messages):
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "test-entry",
                "changes": [{"field": "messages", "value": {"messages": messages}}],
            }
        ],
    }


# --- Concurrency tracker ---


class Tracker:
    """Считает одновременные вхождения в критическую секцию."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self):
        with self._lock:
            self.active -= 1
