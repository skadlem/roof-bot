"""CLI для проверки RAG-ответа: python -m rag.ask "вопрос" <client_id>

Делает то же, что бот на текстовое сообщение: retrieve → сборка сообщения
с контекстом → вызов Gemini. Печатает ответ модели и использованные источники.
"""

import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv

from rag.prompts import (
    RAG_SIMILARITY_THRESHOLD,
    build_no_context_message,
    build_rag_message,
)
from rag.retrieve import retrieve

# консоль Windows (cp1252) не умеет кириллицу в print
sys.stdout.reconfigure(encoding="utf-8")

# та же модель, что у бота (main.py)
GEMINI_MODEL = "models/gemini-3.1-flash-lite"


def main():
    if len(sys.argv) < 3:
        sys.exit('Использование: python -m rag.ask "вопрос" <client_id>')
    question, client_id = sys.argv[1], sys.argv[2]

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "Ошибка: не задан GEMINI_API_KEY (или GOOGLE_API_KEY) в .env — "
            "без ключа нельзя получить эмбеддинги."
        )
    genai.configure(api_key=api_key)

    hits = retrieve(question, client_id, k=3)
    used = [h for h in hits if h["score"] >= RAG_SIMILARITY_THRESHOLD]
    if used:
        print("Контекст:")
        for h in used:
            print(f"  [{h['score']:.3f}] {h['source']} (client: {h['client_id']})")
    else:
        print("Контекст: пусто (ни один чанк не набрал порога)")

    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    message = (
        build_rag_message(question, hits) if used
        else build_no_context_message(question)
    )
    response = model.generate_content(message)

    print(f"\nОтвет:\n{response.text}\n")
    if used:
        print("Источники ответа: " + ", ".join(h["source"] for h in used))
    else:
        print("Источники ответа: нет (без контекста)")


if __name__ == "__main__":
    main()
