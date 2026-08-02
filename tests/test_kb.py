"""Tests for the RAG knowledge base: chunking (build_kb) and retrieval (main).

All embeddings are faked — zero network, zero real API calls.
"""

import pytest

import build_kb
from rag import retrieve as rag_retrieve
from tests.helpers import make_text_message

GARANTY = "Гарантия на кровельные работы — 5 лет."
DELIVERY = "Доставка материалов по городу бесплатная."


# --- chunking (build_kb) ---


def test_chunk_text_paragraph_boundaries_and_overlap():
    text = ("a\n\n" + "b" * 840)
    chunks = build_kb.chunk_text(text)
    assert len(chunks) == 2
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[0][-40:-30] in chunks[1]


def test_chunk_text_headers_and_comments_are_stripped():
    text = "# Заголовок\n\n<!-- ЗАПОЛНИТЬ: факты -->\n\nЕдинственный факт."
    assert build_kb.chunk_text(text) == ["Единственный факт."]


def test_read_kb_files_skips_readme(tmp_path):
    (tmp_path / "README.md").write_text("инструкция, не факты", encoding="utf-8")
    (tmp_path / "faq.md").write_text(
        "# FAQ\n\n<!-- ЗАПОЛНИТЬ -->\n\nМы делаем монтаж кровли под ключ.",
        encoding="utf-8",
    )
    chunks = [
        c
        for text in build_kb.read_kb_files(str(tmp_path))
        for c in build_kb.chunk_text(text)
    ]
    assert chunks == ["Мы делаем монтаж кровли под ключ."]


# --- retrieval (main → rag/) ---


@pytest.fixture
def fake_rag(main, monkeypatch):
    """Фейковый rag.retrieve.retrieve: факты по ключевым словам вопроса.

    Слово не совпало — чанк со score 0.2 (ниже порога RAG_SIMILARITY_THRESHOLD).
    """
    FACTS = {
        "гарант": ("гарантия: 5 лет", "kb/faq.md"),
        "доставк": ("доставка: бесплатно", "kb/faq.md"),
        "срок": ("срок: 2 недели", "kb/faq.md"),
    }

    def fake_retrieve(question, client_id, k=4):
        word = next((w for w in FACTS if w in question), None)
        if not word:
            return [{"text": "посторонний чанк", "score": 0.2,
                     "source": "kb/faq.md", "client_id": "all"}]
        text, source = FACTS[word]
        return [{"text": text, "score": 0.9, "source": source, "client_id": "all"}]

    monkeypatch.setattr(rag_retrieve, "retrieve", fake_retrieve)


def test_first_turn_sends_kb_context(main, fake_rag, gemini, outbox):
    """Агент-цикл (rag.agent.run_agent) подмешивает контекст в первый ход."""
    main.process_gemini_response("79990000000", user_message="какая гарантия?")
    sent = gemini.sent_messages[0]
    assert "гарантия: 5 лет" in sent
    assert "[kb/faq.md]" in sent
    assert sent.endswith("какая гарантия?")


def test_first_turn_below_threshold_sends_plain_question(main, fake_rag, gemini, outbox):
    main.process_gemini_response("79990000000", user_message="здравствуйте")
    assert gemini.sent_messages[0] == "здравствуйте"


def test_first_turn_retrieve_exception_falls_back_to_plain_question(
    main, gemini, outbox, monkeypatch
):
    def boom(question, client_id, k=4):
        raise RuntimeError("no api")
    monkeypatch.setattr(rag_retrieve, "retrieve", boom)
    main.process_gemini_response("79990000000", user_message="гарантия")
    assert gemini.sent_messages[0] == "гарантия"


def test_system_prompt_defers_to_manager(main):
    assert "уточнишь у менеджера" in main.SYSTEM_PROMPT
    assert "НЕ выдумывай цены" in main.SYSTEM_PROMPT


# --- wiring (webhook → Gemini) ---


def test_text_message_sends_kb_context(main, fake_rag, gemini, outbox):
    main.handle_single_message(
        make_text_message("79990000000", "какая у вас гарантия на кровлю?")
    )
    sent = gemini.sent_messages[0]
    assert "гарантия: 5 лет" in sent


def test_text_message_without_match_is_unchanged(main, fake_rag, gemini, outbox):
    main.handle_single_message(make_text_message("79990000000", "здравствуйте"))
    assert gemini.sent_messages[0] == "здравствуйте"


def test_voice_message_skips_kb(main, fake_rag, gemini, outbox, monkeypatch):
    # сериализация аудио-истории — отдельная история; здесь проверяем только
    # что голосовой путь не ходит в базу знаний
    monkeypatch.setattr(main, "serialize_chat_history", lambda chat: [])
    main.process_gemini_response(
        "79990000000", audio_data=b"audio", mime_type="audio/ogg"
    )
    assert gemini.sent_messages[0] == [
        {"type": "audio", "base64": "YXVkaW8=", "mime_type": "audio/ogg"}
    ]
    assert outbox[0][2] == "Привет!"
