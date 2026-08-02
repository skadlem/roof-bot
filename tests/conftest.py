"""Shared fixtures. Everything external is faked — zero network, zero real API calls."""

import os

import pytest
import google.generativeai as genai

import rag.agent
import rag.retrieve
from tests.helpers import FakeLLM


class _NoLimit:
    """Заглушка для gemini_rate_limiter: никогда не блокирует."""

    def acquire(self):
        pass


class _DummyScheduler:
    """Заглушка APScheduler: startup не поднимает фоновый поток."""

    def add_job(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def shutdown(self, *args, **kwargs):
        pass


class _FakeSheet:
    """Записывает append_row в память вместо Google Sheets."""

    def __init__(self):
        self.rows = []

    def append_row(self, row):
        self.rows.append(row)


@pytest.fixture(scope="session")
def sandbox(tmp_path_factory):
    """Изолированная среда: чистый cwd, фейковые env-переменные, фейковая Gemini-модель.

    main.py выполняет работу на этапе import (load_dotenv, чтение данных, создание
    модели), поэтому main импортируется один раз на всю сессию с подменённым
    genai.GenerativeModel ДО import — иначе конструктор уйдёт в сеть.
    """
    cwd = os.getcwd()
    sandbox_dir = tmp_path_factory.mktemp("sandbox")
    os.chdir(sandbox_dir)

    old_env = {
        k: os.environ.get(k)
        for k in (
            "GEMINI_API_KEY", "WA_TOKEN", "WA_PHONE_ID", "WA_VERIFY_TOKEN",
            "WA_APP_SECRET", "OWNER_PHONE_NUMBER", "GEMINI_RPM",
        )
    }
    os.environ.update({
        "GEMINI_API_KEY": "test-gemini-key",
        "WA_TOKEN": "test-wa-token",
        "WA_PHONE_ID": "test-wa-phone-id",
        "WA_VERIFY_TOKEN": "test-verify-token",
        "WA_APP_SECRET": "test-app-secret",
        "OWNER_PHONE_NUMBER": "79990000000",
        "GEMINI_RPM": "999",
    })

    real_model_cls = genai.GenerativeModel
    genai.GenerativeModel = lambda *a, **k: None
    try:
        import main
        yield main
    finally:
        genai.GenerativeModel = real_model_cls
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.chdir(cwd)


@pytest.fixture
def main(sandbox):
    """Импортированный модуль main (один на сессию, состояние чистится в clean_state)."""
    return sandbox


@pytest.fixture(autouse=True)
def clean_state(main, monkeypatch):
    """Чистое состояние между тестами: лиды, сессии, seen-сообщения, таблица, лимитер."""
    with main.state_lock:
        main.open_leads.clear()
        main.chat_sessions.clear()
    with main._seen_messages_lock:
        main.seen_messages.clear()
    main._sheet = None
    main.gemini_rate_limiter = _NoLimit()
    # агент-цикл лимитирует обращения через свой модульный инстанс (main его импортирует)
    rag.agent.gemini_rate_limiter = _NoLimit()
    # retrieve ходит в сеть за эмбеддингами (genai.embed_content), даже когда база
    # пустая, как в sandbox. Заглушка: контекст = пусто, как и отдала бы пустая
    # база — но без HTTP. Иначе сетевой джиттер ронял concurrency-тесты: потоки
    # разъезжались на round-trip раньше, чем попадали в 0.15s-окно замера.
    monkeypatch.setattr(rag.retrieve, "retrieve", lambda *a, **k: [])
    main.scheduler = _DummyScheduler()
    yield


@pytest.fixture(autouse=True)
def sheets(main, monkeypatch):
    """Фейковая Google Sheets: строки копятся в памяти."""
    fake = _FakeSheet()
    monkeypatch.setattr(main, "get_sheet", lambda: fake)
    return fake


@pytest.fixture
def outbox(main, monkeypatch):
    """Перехватывает исходящие сообщения WhatsApp: (тип, номер, ...) записи."""
    sent = []
    monkeypatch.setattr(
        main, "send_whatsapp_message",
        lambda phone, text: sent.append(("text", phone, text)),
    )
    monkeypatch.setattr(
        main, "send_whatsapp_template_message",
        lambda phone, name, params: sent.append(("template", phone, name, params)),
    )
    return sent


@pytest.fixture
def gemini(main, monkeypatch):
    """Фейковая LLM-модель; тесты задают scripted-ответы."""
    chat = FakeLLM()
    monkeypatch.setattr(main, "model", chat)
    return chat


@pytest.fixture
def client(main):
    """TestClient. С контекстным менеджером: event loop живёт между запросами,
    поэтому фоновые asyncio.to_thread-задачи успевают завершиться; scheduler
    заглушен в clean_state, так что startup ничем не вредит."""
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c
