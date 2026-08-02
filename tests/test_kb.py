"""Tests for the RAG knowledge base: chunking (build_kb) and retrieval (main).

All embeddings are faked — zero network, zero real API calls.
"""

import google.generativeai as genai
import pytest

import build_kb
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


# --- retrieval (main) ---


@pytest.fixture
def fake_kb(main):
    """Три ортогональных факта + фейковые эмбеддинги по ключевым словам."""
    kb = {"chunks": ["гарантия: 5 лет", "доставка: бесплатно", "срок: 2 недели"],
          "vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    def fake_embed_content(model, content, **kwargs):
        word = next((w for w in ("гарант", "доставк", "срок") if w in content), None)
        vec = {"гарант": [1, 0, 0], "доставк": [0, 1, 0], "срок": [0, 0, 1]}.get(word, [0, 0, 0])
        return {"embedding": vec}

    monkey = pytest.MonkeyPatch()
    monkey.setattr(genai, "embed_content", fake_embed_content)
    old_kb = main.KB
    main.KB = kb
    yield kb
    main.KB = old_kb
    monkey.undo()


def test_retrieve_kb_returns_matching_fact(main, fake_kb):
    assert "гарантия: 5 лет" in main.retrieve_kb("какая у вас гарантия?")


def test_retrieve_kb_below_threshold_returns_empty(main, fake_kb):
    assert main.retrieve_kb("совершенно посторонний вопрос") == ""


def test_retrieve_kb_none_returns_empty(main, monkeypatch):
    monkeypatch.setattr(main, "KB", None)
    assert main.retrieve_kb("что-нибудь") == ""


def test_retrieve_kb_embed_exception_returns_empty(main, fake_kb, monkeypatch):
    def boom(model, content, **kwargs):
        raise RuntimeError("no api")
    monkeypatch.setattr(genai, "embed_content", boom)
    assert main.retrieve_kb("гарантия") == ""


def test_build_message_with_kb_formats_context(main, fake_kb):
    result = main.build_message_with_kb("какая гарантия?")
    assert result.startswith("БАЗА ЗНАНИЙ КОМПАНИИ")
    assert "гарантия: 5 лет" in result
    assert result.endswith("какая гарантия?")
    assert "\n\n" in result


def test_build_message_with_kb_no_match_returns_original(main, fake_kb):
    assert main.build_message_with_kb("здравствуйте") == "здравствуйте"


def test_system_prompt_defers_to_manager(main):
    assert "уточнишь у менеджера" in main.SYSTEM_PROMPT
    assert "НЕ выдумывай цены" in main.SYSTEM_PROMPT


# --- wiring (webhook → Gemini) ---


def test_text_message_sends_kb_context(main, fake_kb, gemini, outbox):
    main.handle_single_message(
        make_text_message("79990000000", "какая у вас гарантия на кровлю?")
    )
    sent = gemini.sent_messages[0][0][0]
    assert sent.startswith("БАЗА ЗНАНИЙ КОМПАНИИ")
    assert "гарантия: 5 лет" in sent


def test_text_message_without_match_is_unchanged(main, fake_kb, gemini, outbox):
    main.handle_single_message(make_text_message("79990000000", "здравствуйте"))
    assert gemini.sent_messages[0][0][0] == "здравствуйте"


def test_voice_message_skips_kb(main, fake_kb, gemini, outbox, monkeypatch):
    # сериализация аудио-истории — отдельная история; здесь проверяем только
    # что голосовой путь не ходит в базу знаний
    monkeypatch.setattr(main, "serialize_chat_history", lambda chat: [])
    main.process_gemini_response(
        "79990000000", audio_data=b"audio", mime_type="audio/ogg"
    )
    assert gemini.sent_messages[0][0][0] == [{"mime_type": "audio/ogg", "data": b"audio"}]
    assert outbox[0][2] == "Привет!"
