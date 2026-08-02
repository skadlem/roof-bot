"""Fakes and helpers for the test suite. Zero network, zero real API calls."""

import threading
import types
import time

# --- Gemini fakes ---


def make_response(text="Привет!"):
    """Фейковый ответ модели: текст без function_call (как обычная реплика Gemini)."""
    part = types.SimpleNamespace(text=text, inline_data=None)
    content = types.SimpleNamespace(parts=[part])
    candidate = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(candidates=[candidate], text=text)


def make_function_response(name, args):
    """Фейковый ответ модели с function_call (create_order / end_chat)."""
    fc = types.SimpleNamespace(name=name, args=args)
    part = types.SimpleNamespace(text=None, function_call=fc, inline_data=None)
    content = types.SimpleNamespace(parts=[part])
    candidate = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(candidates=[candidate], text="")


class FakeChat:
    """Имитация gemini ChatSession: scripted responses, история собирается в parts."""

    def __init__(self, responses=None, history=None):
        self.history = list(history) if history else []
        self.responses = list(responses or [])
        self._idx = 0
        self.sent_messages = []

    def send_message(self, *args, **kwargs):
        self.sent_messages.append((args, kwargs))
        # как настоящий Gemini, сессия запоминает сообщение пользователя
        text = args[0] if args else "сообщение"
        self.history.append(
            types.SimpleNamespace(role="user", parts=[types.SimpleNamespace(text=text, inline_data=None)])
        )
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
        else:
            resp = make_response()
        self.history.append(
            types.SimpleNamespace(role="model", parts=[types.SimpleNamespace(text=getattr(resp, "text", ""), inline_data=None)])
        )
        return resp


class FakeModel:
    """Имитация genai.GenerativeModel: start_chat() возвращает FakeChat."""

    def __init__(self, chat=None):
        self._chat = chat

    def start_chat(self, history=None, **kwargs):
        return self._chat if self._chat is not None else FakeChat(history=history)


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


class TrackingChat(FakeChat):
    """FakeChat, замеряющий параллельность send_message."""

    def __init__(self, tracker, delay=0.15):
        super().__init__()
        self.tracker = tracker
        self.delay = delay

    def send_message(self, *args, **kwargs):
        self.tracker.enter()
        try:
            time.sleep(self.delay)  # держим слот занятым, чтобы замерить параллельность
            return super().send_message(*args, **kwargs)
        finally:
            self.tracker.exit()
