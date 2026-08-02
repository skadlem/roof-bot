"""Интерактивный агент-цикл в консоли: python -m rag.agent_cli <client_id>

Несколько сообщений в одном процессе (одна сессия). Печатает каждый вызов
инструмента и финальный ответ. Запись лида заглушается: Google Sheets не
трогается (тестовый контур, не продакшен-таблица).
"""

import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv

from rag.agent import DEFAULT_SYSTEM_PROMPT, get_model, run_agent
from rag.prompts import RAG_SIMILARITY_THRESHOLD

# консоль Windows (cp1252) не умеет кириллицу в print
sys.stdout.reconfigure(encoding="utf-8")


def _record_lead_stub(order_data):
    """Заглушка вместо main.add_to_google_sheets: лид печатается, не пишется."""
    print(f"  [ЛЕД → Google Sheets] {order_data}")


def _print_result(result):
    if result.hits:
        used = [h for h in result.hits if h["score"] >= RAG_SIMILARITY_THRESHOLD]
        if used:
            print("Контекст базы знаний:")
            for h in used:
                print(f"  [{h['score']:.3f}] {h['source']}")
        else:
            print("Контекст базы знаний: пусто (ни один чанк не набрал порога)")
    for name, args, outcome in result.tool_calls:
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"[Инструмент] {name}({arg_str})")
        print(f"[Результат] {outcome}")
    if result.lead_saved:
        print("[Лид сохранён — сессия будет закрыта]")
    print(f"\nБот: {result.text}\n")


def main():
    if len(sys.argv) < 2:
        sys.exit("Использование: python -m rag.agent_cli <client_id>")
    client_id = sys.argv[1]

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "Ошибка: не задан GEMINI_API_KEY (или GOOGLE_API_KEY) в .env — "
            "без ключа нельзя вызвать Gemini."
        )
    genai.configure(api_key=api_key)

    chat = get_model(DEFAULT_SYSTEM_PROMPT).start_chat(history=[])
    print(f"Агент готов (client_id={client_id}). Пустая строка или 'exit' — выход.")
    while True:
        try:
            question = input("\nВы: ").strip()
        except EOFError:
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        _print_result(run_agent(chat, client_id, user_message=question,
                                record_lead=_record_lead_stub))


if __name__ == "__main__":
    main()
