"""Сборка базы знаний (RAG): чанкинг kb/*.md и эмбеддинги в kb_embeddings.json.

Запуск: venv/Scripts/python.exe build_kb.py

Нужен GOOGLE_API_KEY (или GEMINI_API_KEY) в .env. Скрипт идемпотентен:
повторный запуск просто перезаписывает kb_embeddings.json заново.
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

# консоль Windows (cp1252) не умеет кириллицу в print
sys.stdout.reconfigure(encoding="utf-8")

EMBEDDING_MODEL = "models/gemini-embedding-001"
KB_DIR = "kb"
OUTPUT_FILE = "kb_embeddings.json"


def chunk_text(text, chunk_size=500, overlap=50):
    """Делит текст на чанки ~chunk_size символов с перекрытием ~overlap.

    HTML-комментарии (заглушки «ЗАПОЛНИТЬ») и markdown-заголовки удаляются —
    в базу попадают только сами факты. Границы чанков предпочитают границы
    абзацев (\n\n), затем переносов строки.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = "\n".join(ln for ln in text.split("\n") if not re.match(r"^\s*#{1,6}\s", ln))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            break_pos = text.rfind("\n\n", start, end)
            if break_pos <= start:
                break_pos = text.rfind("\n", start, end)
            # чанк не короче overlap — иначе start пойдёт назад и цикл не сойдётся
            if break_pos > start + overlap:
                end = break_pos
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def read_kb_files(kb_dir=KB_DIR):
    """Читает все файлы фактов из kb/ (README.md — инструкция, не факты)."""
    texts = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        if path.name == "README.md":
            continue
        texts.append(path.read_text(encoding="utf-8"))
    return texts


def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "Ошибка: не задан GOOGLE_API_KEY (или GEMINI_API_KEY) в .env — "
            "без ключа нельзя получить эмбеддинги."
        )
    genai.configure(api_key=api_key)

    chunks = []
    for text in read_kb_files():
        chunks.extend(chunk_text(text))

    if not chunks:
        print("В kb/ пока нет готовых фактов (заглушки «ЗАПОЛНИТЬ» не индексируются).")
        vectors = []
    else:
        print(f"Чанков: {len(chunks)}")
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT",
        )
        # батч возвращает {"embedding": [[...], [...]]}, одиночный — {"embedding": [...]}
        embedding = result["embedding"]
        vectors = embedding if (embedding and isinstance(embedding[0], list)) else [embedding]

    with open(OUTPUT_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "vectors": vectors}, f, ensure_ascii=False)
    os.replace(OUTPUT_FILE + ".tmp", OUTPUT_FILE)
    print(f"Записано {len(chunks)} чанков в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
