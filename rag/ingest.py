"""Сборка базы знаний (RAG): чанкинг kb/*.md и Google Sheets, эмбеддинги, ChromaDB.

Запуск: venv/Scripts/python.exe -m rag.ingest

Нужны GEMINI_API_KEY (или GOOGLE_API_KEY) в .env, google_credentials.json и
SHEET_KEY для таблицы. Скрипт идемпотентен: повторный запуск пересобирает только
изменившиеся источники (совпадение source+sha256), остальные пропускает.
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
import google.generativeai as genai
import gspread
from chromadb.config import Settings
from dotenv import load_dotenv

# консоль Windows (cp1252) не умеет кириллицу в print
sys.stdout.reconfigure(encoding="utf-8")

# google.generativeai (легаси-пакет, как в main.py); gemini-embedding-2 —
# актуальное имя модели (002 в списке моделей нет, 404)
EMBEDDING_MODEL = "models/gemini-embedding-2"
KB_DIR = "kb"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_BATCH = 64
# Общие факты из kb/ видны всем клиентам, у заказов из таблицы client_id = телефон
SHARED_CLIENT_ID = "all"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Делит текст на чанки ~chunk_size символов с перекрытием ~overlap.

    HTML-комментарии (заглушки «ЗАПОЛНИТЬ») удаляются. Заголовок markdown не
    отрывается от своего содержания: текст режется на секции по заголовкам, и
    каждый чанк секции начинается с её заголовка. Границы чанков предпочитают
    границы абзацев (\\n\\n), затем переносы строк.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    # секции: (заголовок, тело); текст до первого заголовка — секция с пустым заголовком
    sections = []
    heading = ""
    for line in text.split("\n"):
        if re.match(r"^\s*#{1,6}\s", line):
            heading = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
            sections.append((heading, ""))
        elif sections:
            sections[-1] = (sections[-1][0], sections[-1][1] + "\n" + line)
        else:
            sections.append(("", line))

    chunks = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        prefix = f"## {heading}\n\n" if heading else ""
        start = 0
        while start < len(body):
            end = min(start + chunk_size, len(body))
            if end < len(body):
                break_pos = body.rfind("\n\n", start, end)
                if break_pos <= start:
                    break_pos = body.rfind("\n", start, end)
                # чанк не короче overlap — иначе start пойдёт назад и цикл не сойдётся
                if break_pos > start + overlap:
                    end = break_pos
            chunks.append(prefix + body[start:end].strip())
            if end >= len(body):
                break
            start = end - overlap
    return [c for c in chunks if c]


def read_kb_files(kb_dir=KB_DIR):
    """Читает файлы фактов из kb/ (README.md — инструкция, не факты).

    Возвращает список (источник, client_id, текст).
    """
    sources = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        if path.name == "README.md":
            continue
        sources.append((str(path).replace("\\", "/"), SHARED_CLIENT_ID, path.read_text(encoding="utf-8")))
    return sources


def read_sheet():
    """Читает заказы из Google Sheets: (имя листа, [(client_id, текст заказа), ...]).

    Ленивая инициализация с try/except — при недоступной таблице бот не падает,
    источник просто пропускается (тот же паттерн, что add_to_google_sheets).
    """
    sheet_key = os.getenv("SHEET_KEY")
    if not sheet_key:
        print("SHEET_KEY не задан в .env — таблица пропущена.")
        return None
    try:
        ws = gspread.service_account(filename="google_credentials.json").open_by_key(sheet_key).sheet1
        rows = ws.get_all_values()
    except Exception as e:
        print(f"[GOOGLE SHEETS ERROR] {e}")
        return None
    if not rows:
        return None
    headers = [h.strip() for h in rows[0]]
    orders = []
    for row in rows[1:]:
        order = dict(zip(headers, [c.strip() for c in row]))
        phone = order.get("Phone", "")
        if not phone:
            continue
        parts = [f"{k}: {v}" for k, v in order.items() if v and k != "Phone"]
        orders.append((phone, "Заказ: " + ", ".join(parts)))
    return ws.title, orders


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Эмбеддинги списка текстов батчами (Gemini лимитирует размер запроса).

    content всегда список → result["embedding"] — список векторов.
    """
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts[i : i + EMBED_BATCH],
            task_type=task_type,
        )
        vectors.extend(result["embedding"])
    return vectors


def get_collection():
    """Открывает (создавая) ChromaDB-коллекцию на диске."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _unchanged(collection, source, source_hash):
    """Не менялся ли источник с прошлого прогона (по hash первого чанка)."""
    existing = collection.get(where={"source": source}, include=["metadatas"])
    return bool(existing["metadatas"]) and existing["metadatas"][0].get("source_hash") == source_hash


def _upsert(collection, source, source_hash, items):
    """Заливает чанки источника (items = [(client_id, текст), ...]).

    Эмбеддинги считаем до удаления старых чанков, чтобы при сбое источник
    не остался пустым. ID чанка — source#номер строки.
    """
    texts = [text for _, text in items]
    vectors = embed_texts(texts)
    collection.delete(where={"source": source})
    collection.upsert(
        ids=[f"{source}#{i}" for i in range(len(items))],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {"source": source, "source_hash": source_hash, "client_id": client_id}
            for client_id, _ in items
        ],
    )
    print(f"[{source}] загружено чанков: {len(items)}")
    return len(items)


def ingest_source(collection, source, client_id, text):
    """Обновляет чанки одного файла (kb/*.md) в коллекции."""
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if _unchanged(collection, source, source_hash):
        print(f"[{source}] без изменений, пропущен")
        return 0
    chunks = chunk_text(text)
    if not chunks:
        # факты удалились из источника (остались только заглушки) — чистим коллекцию
        collection.delete(where={"source": source})
        print(f"[{source}] нет фактов (только заглушки), старые чанки удалены")
        return 0
    return _upsert(collection, source, source_hash, [(client_id, c) for c in chunks])


def ingest_sheet(collection, source, orders):
    """Обновляет чанки заказов таблицы (один источник = один лист).

    Идемпотентность на уровне таблицы: hash всех строк. Изменилась любая
    строка — пересобираются все (строк мало, переэмбеддинг дешёвый).
    """
    source_hash = hashlib.sha256("\n".join(t for _, t in orders).encode("utf-8")).hexdigest()
    if _unchanged(collection, source, source_hash):
        print(f"[{source}] без изменений, пропущен")
        return 0
    if not orders:
        collection.delete(where={"source": source})
        print(f"[{source}] строк нет, старые чанки удалены")
        return 0
    return _upsert(collection, source, source_hash, orders)


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "Ошибка: не задан GEMINI_API_KEY (или GOOGLE_API_KEY) в .env — "
            "без ключа нельзя получить эмбеддинги."
        )
    genai.configure(api_key=api_key)

    collection = get_collection()
    total = 0
    for source, client_id, text in read_kb_files():
        total += ingest_source(collection, source, client_id, text)
    sheet = read_sheet()
    if sheet:
        total += ingest_sheet(collection, *sheet)

    if collection.count() == 0:
        print("Источников нет: в kb/ пусто и таблица не прочитана.")
    print(f"Готово, всего чанков в коллекции: {collection.count()}")


if __name__ == "__main__":
    main()
