"""Поиск по базе знаний (RAG).

Проверка вручную: venv/Scripts/python.exe -m rag.retrieve "вопрос" <client_id>

Нужны GEMINI_API_KEY (или GOOGLE_API_KEY) в .env и собранная коллекция
(запустите python -m rag.ingest до этого).
"""

import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv

from rag.ingest import EMBEDDING_MODEL, SHARED_CLIENT_ID, get_collection

# консоль Windows (cp1252) не умеет кириллицу в print
sys.stdout.reconfigure(encoding="utf-8")


def retrieve(question, client_id, k=4):
    """Возвращает до k чанков базы знаний клиента, близких к вопросу.

    Ищет по чанкам клиента и общим (client_id="all", факты из kb/).
    score — косинусное сходство (0..1), chroma возвращает distance = 1 - cos
    для cosine-пространства. Перед вызовом нужно load_dotenv() и
    genai.configure(api_key=...) — как в main() ниже.
    """
    collection = get_collection()
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=question,
        task_type="RETRIEVAL_QUERY",
    )
    query_vec = result["embedding"]

    hits = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        where={"client_id": {"$in": [client_id, SHARED_CLIENT_ID]}},
        include=["documents", "metadatas", "distances"],
    )

    found = []
    for i, doc in enumerate(hits["documents"][0]):
        meta = hits["metadatas"][0][i]
        found.append(
            {
                "text": doc,
                "score": 1 - hits["distances"][0][i],
                "source": meta["source"],
                "client_id": meta["client_id"],
            }
        )
    return found


def main():
    if len(sys.argv) < 3:
        sys.exit('Использование: python -m rag.retrieve "вопрос" <client_id>')
    question, client_id = sys.argv[1], sys.argv[2]

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "Ошибка: не задан GEMINI_API_KEY (или GOOGLE_API_KEY) в .env — "
            "без ключа нельзя получить эмбеддинги."
        )
    genai.configure(api_key=api_key)

    hits = retrieve(question, client_id)
    if not hits:
        print("Ничего не найдено.")
        return
    for hit in hits:
        print(f"[{hit['score']:.3f}] {hit['source']} (client: {hit['client_id']})")
        print(hit["text"])
        print()


if __name__ == "__main__":
    main()
