"""Tests for the RAG pipeline: chunking (rag.ingest) and retrieval (rag.retrieve).

Embeddings are faked and ChromaDB is pointed at a temp dir — zero network,
zero real API calls, nothing written to the real chroma_db/.
"""

import google.generativeai as genai

import rag.ingest
from rag.ingest import SHARED_CLIENT_ID, get_collection
from rag.retrieve import retrieve


# --- chunking (rag.ingest) ---


def test_chunk_text_heading_stays_with_content():
    text = "# Услуги\n\nМы делаем монтаж кровли под ключ."
    assert rag.ingest.chunk_text(text) == ["## Услуги\n\nМы делаем монтаж кровли под ключ."]


def test_chunk_text_paragraph_boundaries_and_overlap():
    text = "# Тема\n\n" + "a" * 840
    chunks = rag.ingest.chunk_text(text)
    assert len(chunks) == 2
    assert all(len(c) <= 510 for c in chunks)  # 500 символов + префикс заголовка
    assert all(c.startswith("## Тема") for c in chunks)
    assert chunks[0][-40:-30] in chunks[1]


def test_chunk_text_comments_stripped_empty_sections_dropped():
    text = "# Услуги\n\n<!-- ЗАПОЛНИТЬ: факты -->\n\n## Прайс\n\nЦена: 850 тг."
    assert rag.ingest.chunk_text(text) == ["## Прайс\n\nЦена: 850 тг."]


# --- retrieval (rag.retrieve) ---


def _seed(collection):
    collection.upsert(
        ids=["shared#0", "a#0", "b#0"],
        documents=["гарантия: 5 лет", "заказ клиента A", "заказ клиента B"],
        embeddings=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        metadatas=[
            {"source": "kb/faq.md", "source_hash": "h", "client_id": SHARED_CLIENT_ID},
            {"source": "Sheet1/111", "source_hash": "h", "client_id": "111"},
            {"source": "Sheet1/222", "source_hash": "h", "client_id": "222"},
        ],
    )


def test_retrieve_filters_by_client_id(monkeypatch, tmp_path):
    monkeypatch.setattr(rag.ingest, "CHROMA_DIR", str(tmp_path))

    def fake_embed_content(model, content, **kwargs):
        return {"embedding": [1, 0, 0] if "гарантия" in content else [0, 0, 0]}

    monkeypatch.setattr(genai, "embed_content", fake_embed_content)
    _seed(get_collection())

    hits = retrieve("какая гарантия?", "111", k=4)
    sources = [h["source"] for h in hits]
    assert "kb/faq.md" in sources  # общие факты доступны всем клиентам
    assert "Sheet1/111" in sources  # чанки своего клиента
    assert "Sheet1/222" not in sources  # чужой клиент не виден
    assert hits[0]["score"] > 0.5
    assert hits[0]["text"] == "гарантия: 5 лет"
